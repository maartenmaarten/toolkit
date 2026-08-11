#!/usr/bin/env python3
"""
Download nucleotide GenBank (gbff) records for a list of protein accessions.

Rationale
---------
Protein accessions are mapped to the *specific* nucleotide record they are
annotated on, so that genomic context is preserved. Non-redundant RefSeq
proteins (WP_*) are skipped by design: an identical protein sequence does not
imply identical gene neighbourhood, and a WP_ accession maps to many unrelated
genomes with no principled way to pick one.

Mapping strategy
----------------
1. Primary: Identical Protein Groups (efetch -db protein -rettype ipg).
   Batchable, and returns nucleotide accession + coordinates + assembly.
   Rows are filtered to the queried accession itself, not the whole group.
2. Fallback: per-accession elink protein -> nuccore, for proteins that have
   no IPG row (older INSDC entries, eukaryotic records, some CDS-less entries).

Records are fetched with rettype="gbwithparts" so that CON/scaffold records
come back with sequence rather than a join() stub, and one file is written per
nucleotide accession for flexible downstream use.

Usage
-----
    python fetch_genomic_context.py ids.txt -o out/
    cat ids.txt | python fetch_genomic_context.py -o out/

Environment
-----------
    NCBI_EMAIL    required by NCBI (or pass --email)
    NCBI_API_KEY  optional; raises the rate limit from 3/s to 10/s
"""

from Bio import Entrez
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable


# --------------------------------------------------------------------------
# Input parsing
# --------------------------------------------------------------------------

def parse_ids(text):
    """Parse IDs in JSON-array or one-per-line format."""
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
            return [str(i).strip() for i in items if str(i).strip()]
        except json.JSONDecodeError:
            tokens = re.findall(r"['\"]([^'\"]+)['\"]", stripped)
            if tokens:
                return tokens
    ids = []
    for line in stripped.splitlines():
        token = line.strip().strip("\",'")
        if token:
            ids.append(token)
    return ids


def strip_version(acc):
    return acc.rsplit(".", 1)[0] if re.search(r"\.\d+$", acc) else acc


def is_wp(acc):
    return strip_version(acc).upper().startswith("WP_")


# --------------------------------------------------------------------------
# NCBI helpers
# --------------------------------------------------------------------------

def _retry(fn, retries, retry_sleep, label, log):
    """Call fn() with retries. Returns (result, error_string_or_None)."""
    last = None
    for attempt in range(retries + 1):
        try:
            return fn(), None
        except Exception as e:  # noqa: BLE001 - network/parse errors are varied
            last = e
            if attempt < retries:
                log(f"  {label} failed ({e}); retry {attempt + 1}/{retries} "
                    f"in {retry_sleep}s")
                time.sleep(retry_sleep)
    return None, f"{type(last).__name__}: {last}"


def fetch_ipg_table(batch):
    handle = Entrez.efetch(db="protein", id=",".join(batch),
                           rettype="ipg", retmode="text")
    data = handle.read()
    handle.close()
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    if not data.strip():
        raise ValueError("empty IPG response")
    return data


def parse_ipg(text):
    """Parse an IPG TSV into {protein_acc_noversion: [row_dict, ...]}.

    Columns are resolved by header name, not by position, because NCBI has
    changed the column order of this report before.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    header = [h.strip() for h in lines[0].split("\t")]
    idx = {name.lower(): i for i, name in enumerate(header)}

    def col(row, *names):
        for n in names:
            i = idx.get(n.lower())
            if i is not None and i < len(row):
                return row[i].strip()
        return ""

    out = defaultdict(list)
    for line in lines[1:]:
        row = line.split("\t")
        protein = col(row, "Protein")
        nuc = col(row, "Nucleotide Accession")
        if not protein or not nuc:
            continue
        out[strip_version(protein)].append({
            "protein": protein,
            "nucleotide": nuc,
            "start": col(row, "Start"),
            "stop": col(row, "Stop"),
            "strand": col(row, "Strand"),
            "assembly": col(row, "Assembly"),
            "source": col(row, "Source"),
            "organism": col(row, "Organism"),
            "strain": col(row, "Strain"),
        })
    return out


def elink_protein_to_nuccore(protein_id):
    """Fallback single-accession mapping. Returns a nucleotide accession."""
    handle = Entrez.elink(dbfrom="protein", db="nuccore", id=protein_id,
                          linkname="protein_nuccore")
    result = Entrez.read(handle)
    handle.close()
    uids = []
    for linkset in result:
        for db in linkset.get("LinkSetDb", []):
            uids.extend(link["Id"] for link in db.get("Link", []))
    if not uids:
        raise ValueError("no protein_nuccore link")
    handle = Entrez.esummary(db="nuccore", id=uids[0])
    summary = Entrez.read(handle)
    handle.close()
    acc = summary[0].get("AccessionVersion") or summary[0].get("Caption")
    if not acc:
        raise ValueError("esummary returned no accession")
    return str(acc)


def fetch_nucleotide_records(accessions):
    handle = Entrez.efetch(db="nuccore", id=",".join(accessions),
                           rettype="gbwithparts", retmode="text")
    data = handle.read()
    handle.close()
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    if not data.strip() or data.lstrip().startswith("Error") or "<ERROR>" in data:
        raise ValueError(f"unexpected response: {data[:200]!r}")
    return data


def split_genbank(text):
    """Split a multi-record GenBank stream into {version_accession: record}."""
    records = {}
    for chunk in text.split("\n//\n"):
        if not chunk.strip():
            continue
        m = re.search(r"^VERSION\s+(\S+)", chunk, re.MULTILINE)
        if not m:
            continue
        records[m.group(1)] = chunk.strip() + "\n//\n"
    return records


def record_is_complete(path):
    """Cheap integrity check: file exists and terminates with a record end."""
    try:
        if os.path.getsize(path) < 100:
            return False
        with open(path, "rb") as fh:
            fh.seek(-8, os.SEEK_END)
            return fh.read().strip().endswith(b"//")
    except OSError:
        return False


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    genome_dir = os.path.join(args.outdir, "genomes")
    os.makedirs(genome_dir, exist_ok=True)

    log_path = os.path.join(args.outdir, "download_report.txt")
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    started = datetime.now()
    log(f"# fetch_genomic_context.py report")
    log(f"# started: {started.isoformat(timespec='seconds')}")
    log(f"# command: {' '.join(sys.argv)}")
    log("")

    # ---- input --------------------------------------------------------
    if args.id_list:
        with open(args.id_list) as fh:
            raw_ids = parse_ids(fh.read())
    elif not sys.stdin.isatty():
        raw_ids = parse_ids(sys.stdin.read())
    else:
        sys.exit("error: provide an id list file or pipe IDs via stdin")

    seen = set()
    all_ids = []
    n_dupes = 0
    for i in raw_ids:
        key = strip_version(i)
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)
        all_ids.append(i)

    wp_ids = [i for i in all_ids if is_wp(i)]
    query_ids = [i for i in all_ids if not is_wp(i)]

    log(f"input accessions        : {len(raw_ids)}")
    log(f"  duplicates removed    : {n_dupes}")
    log(f"  WP_ skipped           : {len(wp_ids)}")
    log(f"  to map                : {len(query_ids)}")
    log("")
    if not query_ids:
        log("nothing to do.")
        write_report(log_path, log_lines)
        return

    # ---- 1. map protein -> nucleotide ---------------------------------
    mapping = {}          # input_id -> row dict
    unmapped = []
    ambiguous = {}        # input_id -> n alternative rows

    batches = [query_ids[i:i + args.map_batch]
               for i in range(0, len(query_ids), args.map_batch)]
    log(f"[1/2] mapping proteins to nucleotide records "
        f"({len(batches)} IPG batch(es) of {args.map_batch})")

    ipg_hits = {}
    for batch in tqdm(batches, desc="IPG mapping", unit="batch"):
        text, err = _retry(lambda b=batch: fetch_ipg_table(b),
                           args.retries, args.retry_sleep,
                           f"IPG batch starting {batch[0]}", log)
        if text is not None:
            ipg_hits.update(parse_ipg(text))
        else:
            log(f"  IPG batch starting {batch[0]} gave up: {err}")
        time.sleep(args.sleep)

    for pid in query_ids:
        rows = ipg_hits.get(strip_version(pid), [])
        # keep only rows describing this exact protein, not the whole group
        rows = [r for r in rows if strip_version(r["protein"]) == strip_version(pid)]
        if not rows:
            unmapped.append(pid)
            continue
        if len(rows) > 1:
            ambiguous[pid] = len(rows)
        mapping[pid] = rows[0]

    # fallback for anything IPG did not cover
    if unmapped and not args.no_elink_fallback:
        log(f"  {len(unmapped)} accession(s) missing from IPG; "
            f"trying elink fallback")
        still_unmapped = []
        for pid in tqdm(list(unmapped), desc="elink fallback", unit="id"):
            acc, err = _retry(lambda p=pid: elink_protein_to_nuccore(p),
                              args.retries, args.retry_sleep,
                              f"elink {pid}", log)
            if acc:
                mapping[pid] = {"protein": pid, "nucleotide": acc,
                                "start": "", "stop": "", "strand": "",
                                "assembly": "", "source": "elink",
                                "organism": "", "strain": ""}
            else:
                still_unmapped.append(pid)
            time.sleep(args.sleep)
        unmapped = still_unmapped

    log(f"  mapped                : {len(mapping)}")
    log(f"  unmapped              : {len(unmapped)}")
    log(f"  multi-locus proteins  : {len(ambiguous)} (first hit used)")
    log("")

    # write the mapping table regardless of what happens downstream
    map_path = os.path.join(args.outdir, "protein_to_nucleotide.tsv")
    with open(map_path, "w") as fh:
        fh.write("query_protein\tipg_protein\tnucleotide\tstart\tstop\t"
                 "strand\tassembly\tsource\torganism\tstrain\n")
        for pid, r in mapping.items():
            fh.write("\t".join([pid, r["protein"], r["nucleotide"], r["start"],
                                r["stop"], r["strand"], r["assembly"],
                                r["source"], r["organism"], r["strain"]]) + "\n")

    # ---- 2. download nucleotide records -------------------------------
    nuc_to_proteins = defaultdict(list)
    for pid, r in mapping.items():
        nuc_to_proteins[r["nucleotide"]].append(pid)

    wanted = sorted(nuc_to_proteins)
    already = [n for n in wanted
               if record_is_complete(os.path.join(genome_dir, f"{n}.gbff"))]
    todo = [n for n in wanted if n not in set(already)]

    log(f"[2/2] downloading nucleotide records (rettype=gbwithparts)")
    log(f"  unique records        : {len(wanted)}")
    log(f"  already on disk       : {len(already)}")
    log(f"  to download           : {len(todo)}")

    fetched, failed = [], {}
    nbatches = [todo[i:i + args.fetch_batch]
                for i in range(0, len(todo), args.fetch_batch)]
    for batch in tqdm(nbatches, desc="Fetching gbff", unit="batch"):
        text, err = _retry(lambda b=batch: fetch_nucleotide_records(b),
                           args.retries, args.retry_sleep,
                           f"fetch batch starting {batch[0]}", log)
        if text is None:
            for acc in batch:
                failed[acc] = err
            time.sleep(args.sleep)
            continue
        records = split_genbank(text)
        # index returned records by versionless accession for matching
        by_key = {strip_version(k): (k, v) for k, v in records.items()}
        for acc in batch:
            hit = by_key.get(strip_version(acc))
            if hit is None:
                failed[acc] = "not present in response"
                continue
            version, body = hit
            path = os.path.join(genome_dir, f"{acc}.gbff")
            with open(path, "w") as fh:
                fh.write(body)
            if not record_is_complete(path):
                failed[acc] = "truncated record written"
            else:
                fetched.append((acc, version, os.path.getsize(path)))
        time.sleep(args.sleep)

    # ---- report -------------------------------------------------------
    total_bytes = sum(s for _, _, s in fetched)
    elapsed = (datetime.now() - started).total_seconds()

    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"input accessions        : {len(raw_ids)}")
    log(f"duplicates removed      : {n_dupes}")
    log(f"WP_ skipped             : {len(wp_ids)}")
    log(f"mapped to nucleotide    : {len(mapping)} / {len(query_ids)}")
    log(f"unique nucleotide recs  : {len(wanted)}")
    log(f"downloaded this run     : {len(fetched)}")
    log(f"reused from disk        : {len(already)}")
    log(f"failed                  : {len(failed)}")
    log(f"bytes written           : {total_bytes:,}")
    log(f"elapsed                 : {elapsed:.1f}s")
    log("")

    if wp_ids:
        log(f"-- WP_ accessions skipped ({len(wp_ids)}) "
            f"[non-redundant: genomic context undefined] --")
        for i in wp_ids:
            log(f"  {i}")
        log("")
    if unmapped:
        log(f"-- proteins with no nucleotide mapping ({len(unmapped)}) --")
        for i in unmapped:
            log(f"  {i}")
        log("")
    if ambiguous:
        log(f"-- proteins on >1 nucleotide record ({len(ambiguous)}), "
            f"first hit used --")
        for i, n in ambiguous.items():
            log(f"  {i}\t{n} loci\tused {mapping[i]['nucleotide']}")
        log("")
    if failed:
        log(f"-- failed downloads ({len(failed)}) --")
        for acc, reason in failed.items():
            log(f"  {acc}\t{reason}\t(proteins: "
                f"{','.join(nuc_to_proteins[acc])})")
        log("")

    log(f"mapping table : {map_path}")
    log(f"genomes       : {genome_dir}")
    write_report(log_path, log_lines)
    print(f"\nreport written to {log_path}")

    # machine-readable companion
    with open(os.path.join(args.outdir, "download_report.json"), "w") as fh:
        json.dump({
            "started": started.isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 1),
            "n_input": len(raw_ids),
            "n_duplicates": n_dupes,
            "wp_skipped": wp_ids,
            "unmapped": unmapped,
            "ambiguous": {k: v for k, v in ambiguous.items()},
            "failed": failed,
            "downloaded": [{"accession": a, "version": v, "bytes": s}
                           for a, v, s in fetched],
            "reused": already,
        }, fh, indent=2)

    if failed or unmapped:
        sys.exit(1)


def write_report(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(
        description="Download nucleotide gbff records for protein accessions "
                    "(WP_ accessions are skipped).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("id_list", nargs="?",
                   help="file with protein accessions (one per line or JSON "
                        "array); omit to read from stdin")
    p.add_argument("-o", "--outdir", default="genomic_context",
                   help="output directory")
    p.add_argument("--map-batch", type=int, default=200,
                   help="protein accessions per IPG request")
    p.add_argument("--fetch-batch", type=int, default=5,
                   help="nucleotide records per efetch request; keep small, "
                        "gbwithparts records can be tens of MB each")
    p.add_argument("--sleep", type=float, default=None,
                   help="seconds between requests (default: 0.15 with an API "
                        "key, 0.4 without)")
    p.add_argument("--retries", type=int, default=2,
                   help="retries per failed request")
    p.add_argument("--retry-sleep", type=float, default=10,
                   help="seconds before retrying")
    p.add_argument("--email", default="maarten.boneschansker@wur.nl",
                   help="contact email (env: NCBI_EMAIL)")
    p.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"),
                   help="NCBI API key (env: NCBI_API_KEY)")
    p.add_argument("--no-elink-fallback", action="store_true",
                   help="do not fall back to elink for proteins missing "
                        "from IPG")
    args = p.parse_args()

    if not args.email:
        sys.exit("error: set NCBI_EMAIL or pass --email; NCBI requires it")
    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key
    if args.sleep is None:
        args.sleep = 0.15 if args.api_key else 0.4

    run(args)


if __name__ == "__main__":
    main()