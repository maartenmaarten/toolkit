#!/usr/bin/env python3
"""
Emit the RECORDED sequence of every polymer chain in a PDB or mmCIF file as
FASTA -- i.e. SEQRES (legacy .pdb) / _entity_poly.pdbx_seq_one_letter_code
(.cif), not the sequence built by walking ATOM records. The two differ
whenever any residue in the construct wasn't resolved in the density (a
disordered loop, an unmodelled tag, ...): the recorded sequence still lists
it, the atom-derived one silently skips it. Use this when you want the full
deposited construct; if you specifically want only the residues that are
actually present with coordinates, this is the wrong tool.

Usage:
    python emit_fasta_from_pdb.py 9X3M.cif
    python emit_fasta_from_pdb.py 1abc.pdb -o 1abc.fasta
    python emit_fasta_from_pdb.py structure.cif.gz --format cif-seqres
"""
import argparse
import gzip
import os
import sys

from Bio import SeqIO

FORMAT_BY_EXT = {
    ".cif": "cif-seqres", ".mmcif": "cif-seqres",
    ".pdb": "pdb-seqres", ".ent": "pdb-seqres", ".brk": "pdb-seqres",
}


def guess_format(path):
    base = path[:-3] if path.endswith(".gz") else path
    ext = os.path.splitext(base)[1].lower()
    return FORMAT_BY_EXT.get(ext)


def open_maybe_gzip(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def emit_fasta(in_handle, fmt, out_handle):
    count = 0
    for record in SeqIO.parse(in_handle, fmt):
        pdb_id, _, chain = record.id.partition(":")
        mol_type = record.annotations.get("molecule_type", "?")
        header = f"{pdb_id}_{chain or record.id} mol:{mol_type} length:{len(record.seq)}  {record.description}"
        out_handle.write(f">{header}\n{record.seq}\n")
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="input .pdb/.ent (SEQRES) or .cif/.mmcif (_entity_poly), optionally .gz")
    ap.add_argument("-o", "--output", help="output FASTA path (default: stdout)")
    ap.add_argument("--format", choices=["pdb-seqres", "cif-seqres"],
                     help="override format auto-detection (by default guessed from the file extension)")
    args = ap.parse_args()

    fmt = args.format or guess_format(args.structure)
    if fmt is None:
        ap.error(f"could not guess format from extension of '{args.structure}'; pass --format explicitly")

    out_handle = open(args.output, "w") if args.output else sys.stdout
    try:
        with open_maybe_gzip(args.structure) as in_handle:
            count = emit_fasta(in_handle, fmt, out_handle)
    finally:
        if args.output:
            out_handle.close()

    if count == 0:
        print(f"warning: no {'SEQRES' if fmt == 'pdb-seqres' else '_entity_poly'} records found "
              f"in {args.structure} -- wrote an empty FASTA", file=sys.stderr)
    elif args.output:
        print(f"wrote {count} chain(s) to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
