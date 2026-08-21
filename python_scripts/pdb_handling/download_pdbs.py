#!/usr/bin/env python3
"""
Download the full mmCIF entry for every PDB ID listed in a CAZy-style
structures TSV.

Full entry files are used deliberately, not per-assembly files: assembly
files are stripped of _struct_ref/_struct_ref_seq/_struct_ref_seq_dif, which
downstream tools need for UniProt numbering and mutation detection.

Usage:
    python3 download_pdbs.py GH43_structures.tsv -o structures

Stdlib only. Writes <outdir>/<PDBID>.cif and a manifest.tsv.
"""
import argparse, csv, gzip, os, re, sys, time, urllib.error, urllib.request

FILES = "https://files.rcsb.org/download"
UA = {"User-Agent": "gh43-structure-fetch/1.0 (academic use)"}

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
    """Non-water, non-polymer chem comp IDs present in an mmCIF file.

    mmCIF _atom_site records are whitespace-delimited with a per-file column
    order (declared by the preceding _atom_site.* loop header), not fixed
    character columns like legacy PDB format -- so the comp_id column index
    is read from that header rather than assumed.
    """
    codes = set()
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", errors="replace") as fh:
            tags, comp_idx = [], None
            for line in fh:
                s = line.rstrip("\n")
                if s.startswith("_atom_site."):
                    tags.append(s.split()[0].split(".", 1)[1])
                    continue
                if s.startswith("HETATM"):
                    if comp_idx is None:
                        idx = {t: i for i, t in enumerate(tags)}
                        key = "auth_comp_id" if "auth_comp_id" in idx else "label_comp_id"
                        comp_idx = idx.get(key)
                        if comp_idx is None:
                            return set()
                    fields = s.split()
                    if comp_idx < len(fields):
                        codes.add(fields[comp_idx])
    except Exception:
        return set()
    return {c for c in codes if c not in {"HOH", "DOD", ""}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("-o", "--outdir", default="structures")
    ap.add_argument("--column", default="PDB/3D")
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    entries, skipped = parse_ids(args.tsv, args.column)
    print(f"{len(entries)} unique PDB IDs parsed; {len(skipped)} unparseable "
          f"field(s): {skipped}", file=sys.stderr)

    manifest = []
    for pdb in sorted(entries):
        fname = f"{pdb}.cif"
        dest = os.path.join(args.outdir, fname)
        status = download(f"{FILES}/{fname}", dest)
        ligs = ",".join(sorted(het_codes(dest))) if status in ("ok", "cached") else ""
        manifest.append((pdb, fname, status, ligs))
        print(f"{pdb}\t{status}\t{ligs}", file=sys.stderr)
        time.sleep(args.delay)

    mpath = os.path.join(args.outdir, "manifest.tsv")
    with open(mpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["pdb_id", "file", "status", "het_codes"])
        w.writerows(manifest)
    print(f"\nmanifest written to {mpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
