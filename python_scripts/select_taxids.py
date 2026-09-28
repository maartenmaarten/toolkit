#!/usr/bin/env python3
"""Select rows of NCBI prot.accession2taxid(.FULL) whose accession is in an ID list.

Usage:
    python select_taxids.py n1000_acc.txt prot.accession2taxid.FULL.gz selected.tsv

- ID list: one accession per line, no header, no version (e.g. AAA02557).
- TSV: header 'accession.version<TAB>taxid' (may be .gz).
- Matching is on the accession with any ':PDB=...' suffix and '.N' version removed,
  so 'AAA02557.1' matches 'AAA02557'.
- Writes matched rows plus a '<output>.missing.txt' list of IDs without a taxid.
"""
import sys
import polars as pl

ids_path, tsv_path, out_path = sys.argv[1:4]

# 1. ID list: read as strings (IDs like 333361328 must not become ints), clean, dedupe
ids = (
    pl.read_csv(ids_path, has_header=False, new_columns=["accession"],
                infer_schema=False, separator="\t")
    .select(pl.col("accession").str.strip_chars())   # removes \r and spaces
    .filter(pl.col("accession") != "")               # empty lines
    .unique()
)

# 2. Lazy scan of the big table; derive a version-free match key
matched = (
    pl.scan_csv(tsv_path, separator="\t", has_header=True,
                infer_schema=False, quote_char=None)
    .with_columns(
        pl.col("accession.version")
          .str.replace(r":.*$", "")          # 0403181A:PDB=1BP2 -> 0403181A
          .str.replace(r"\.\d+$", "")        # WP_000001.1       -> WP_000001
          .alias("accession")
    )
    .join(ids.lazy(), on="accession", how="semi")
    .unique(subset="accession", keep="first")  # PRF entries can occur with and without :PDB suffix
    .select("accession", "accession.version", "taxid")
    .collect(engine="streaming")   # older polars (<1.23): .collect(streaming=True)
)

matched.write_csv(out_path, separator="\t")

# 3. Report IDs that found no taxid
missing = ids.join(matched.select("accession"), on="accession", how="anti")
missing.write_csv(out_path + ".missing.txt", include_header=False)

print(f"IDs: {ids.height} | matched: {matched.height} | missing: {missing.height}",
      file=sys.stderr)
