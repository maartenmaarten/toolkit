#!/usr/bin/env python3
"""
Download every biological assembly for the PDB entries listed in a CAZy-style
structures TSV.

Usage:
    python3 fetch_gh43_assemblies.py GH43_structures.tsv -o assemblies

Stdlib only. Writes <outdir>/<PDBID>-assembly<N>.cif.gz and a manifest.tsv.
"""
import argparse, csv, gzip, json, os, re, sys, time, urllib.error, urllib.request

GRAPHQL = "https://data.rcsb.org/graphql"
FILES = "https://files.rcsb.org/download"
UA = {"User-Agent": "gh43-assembly-fetch/1.0 (academic use)"}

PDB_RE = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")


def parse_ids(tsv_path, column="PDB/3D"):
    """Return {pdb_id: [source rows]}. Tolerates truncated chain lists."""
    found, skipped = {}, []
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            raw = (row.get(column) or "").strip()
            m = PDB_RE.match(raw)
            if not m:
                if raw:
                    skipped.append(raw)
                continue
            found.setdefault(m.group(1).upper(), []).append(row)
    return found, skipped


def fetch_json(url, payload=None, tries=4):
    for attempt in range(tries):
        try:
            data = json.dumps(payload).encode() if payload else None
            hdr = dict(UA)
            if payload:
                hdr["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=data, headers=hdr)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def assembly_ids(pdb_ids, batch=50):
    """Ask the RCSB GraphQL API how many assemblies each entry has."""
    out = {}
    ids = sorted(pdb_ids)
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        q = ("{entries(entry_ids:%s){rcsb_id "
             "rcsb_entry_container_identifiers{assembly_ids}}}" % json.dumps(chunk))
        res = fetch_json(GRAPHQL, {"query": q})
        entries = (res.get("data") or {}).get("entries") or []
        for e in entries:
            if not e:
                continue
            aid = ((e.get("rcsb_entry_container_identifiers") or {})
                   .get("assembly_ids") or ["1"])
            out[e["rcsb_id"].upper()] = aid
        time.sleep(0.2)
    for p in ids:
        out.setdefault(p, None)  # None = entry not found in the PDB
    return out


def download(url, dest, tries=4):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "cached"
    tmp = dest + ".part"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as fh:
                while True:
                    buf = r.read(1 << 16)
                    if not buf:
                        break
                    fh.write(buf)
            os.replace(tmp, dest)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "404"
            if attempt == tries - 1:
                return f"http{e.code}"
            time.sleep(2 ** attempt)
        except Exception:
            if attempt == tries - 1:
                return "error"
            time.sleep(2 ** attempt)
    return "error"


def het_codes(path):
    """Non-water, non-polymer chem comp IDs present in a gzipped mmCIF."""
    codes = set()
    try:
        with gzip.open(path, "rt", errors="replace") as fh:
            for line in fh:
                if line.startswith("HETATM"):
                    codes.add(line[17:20].strip())
                elif line.startswith(("ATOM ", "_", "loop_")) and len(codes) and line.startswith("_"):
                    continue
    except Exception:
        return set()
    return {c for c in codes if c not in {"HOH", "DOD", ""}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("-o", "--outdir", default="assemblies")
    ap.add_argument("--column", default="PDB/3D")
    ap.add_argument("--also-asymmetric-unit", action="store_true",
                    help="additionally fetch the deposited AU, which contains "
                         "every modelled ligand copy")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    entries, skipped = parse_ids(args.tsv, args.column)
    print(f"{len(entries)} unique PDB IDs parsed; {len(skipped)} unparseable "
          f"field(s): {skipped}", file=sys.stderr)

    print("querying RCSB for assembly counts...", file=sys.stderr)
    asm = assembly_ids(entries)
    missing = [p for p, v in asm.items() if v is None]
    if missing:
        print(f"not found in the PDB: {missing}", file=sys.stderr)

    manifest = []
    for pdb in sorted(entries):
        aids = asm.get(pdb)
        if aids is None:
            manifest.append((pdb, "-", "-", "not_in_pdb", ""))
            continue
        targets = [(f"{pdb}-assembly{a}.cif",
                    f"{FILES}/{pdb}-assembly{a}.cif", a) for a in aids]
        if args.also_asymmetric_unit:
            targets.append((f"{pdb}.cif", f"{FILES}/{pdb}.cif", "AU"))
        for fname, url, label in targets:
            dest = os.path.join(args.outdir, fname)
            status = download(url, dest)
            ligs = ",".join(sorted(het_codes(dest))) if status in ("ok", "cached") else ""
            manifest.append((pdb, label, fname, status, ligs))
            print(f"{pdb}\tassembly {label}\t{status}\t{ligs}", file=sys.stderr)
            time.sleep(args.delay)

    mpath = os.path.join(args.outdir, "manifest.tsv")
    with open(mpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["pdb_id", "assembly", "file", "status", "het_codes"])
        w.writerows(manifest)
    print(f"\nmanifest written to {mpath}", file=sys.stderr)


if __name__ == "__main__":
    main()