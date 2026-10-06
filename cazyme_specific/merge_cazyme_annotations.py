"""
Merge CAZyme annotation sources into a single TSV keyed by genbank_id.

Sources merged:
  - CAZy data     (family, taxonomy, lineage)
  - Unified headers (Mechanism, Clan)
  - Characterized proteins
  - dbCAN domain annotations

Output: one row per unique genbank_id, ready to left-join onto any ID list.

Usage:
    python merge_cazyme_annotations.py --output merged_annotations.tsv
    python merge_cazyme_annotations.py --cazy_data /path/to/cazy.txt --output out.tsv
"""

import argparse
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Merge CAZyme annotation sources.")
parser.add_argument("--cazy_data",    default="/Users/maartenboneschansker/Documents/PhD/collabs/alejandro/cazy_data_20250926_uniq_taxonomy_annotated.txt")
parser.add_argument("--headers",      default="/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/annotations/unified_headers_2026-05-18.tsv")
parser.add_argument("--char",         default="/Users/maartenboneschansker/Documents/PhD/collabs/alejandro/all_char_cazymes_2026-05-18.tsv")
parser.add_argument("--domain_ann",   default="/Users/maartenboneschansker/Documents/PhD/collabs/alejandro/dbcan_parsed_output/dbcan_domain_annotations.tsv")
parser.add_argument("--output",       default="merged_cazyme_annotations.tsv")
args = parser.parse_args()

# ----------------------------------------------------------------------
# A. CAZy data: family + full taxonomy
# ----------------------------------------------------------------------
print("Loading CAZy data…")
annot_df = pd.read_csv(
    args.cazy_data, sep="\t",
    usecols=["genbank_id", "cazy_family", "domain", "organism", "taxid", "source", "lineage"],
    low_memory=False,
)
annot_df["cazy_class"]       = annot_df["cazy_family"].str.extract(r"^([A-Za-z]+)",    expand=False)
annot_df["cazy_family_base"] = annot_df["cazy_family"].str.extract(r"^([A-Za-z]+\d+)", expand=False)

# Keep only GH43 and its subfamilies (GH43_1, GH43_2, …)
gh43_ids = annot_df.loc[annot_df["cazy_family_base"] == "GH43", "genbank_id"].unique()
annot_df = annot_df[annot_df["genbank_id"].isin(gh43_ids)].copy()
print(f"GH43 proteins: {annot_df['genbank_id'].nunique():,}")

LINEAGE_LEVELS = ["tax_kingdom", "tax_phylum", "tax_class", "tax_order",
                  "tax_family",  "tax_genus",  "tax_species"]
lin = (annot_df["lineage"].fillna("")
       .str.split(";", expand=True)
       .reindex(columns=range(7), fill_value="")
       .replace("", np.nan))
lin.columns = LINEAGE_LEVELS
annot_df = pd.concat([annot_df.drop(columns="lineage"), lin], axis=1)

# Deduplicate: prefer non-CBM rows; if only CBMs exist, null out family fields
_is_cbm = annot_df["cazy_family"].str.match(r"^CBM", na=False)
annot_dedup = (
    annot_df
    .assign(_cbm_sort=_is_cbm.astype(int))
    .sort_values(["genbank_id", "_cbm_sort"])
    .drop(columns="_cbm_sort")
    .drop_duplicates("genbank_id")
)
_dedup_is_cbm = annot_dedup["cazy_family"].str.match(r"^CBM", na=False)
annot_dedup.loc[_dedup_is_cbm, ["cazy_family", "cazy_class", "cazy_family_base"]] = np.nan

# Collect CBM families per protein as a semicolon-separated string
cbm_annot = (
    annot_df[_is_cbm]
    .groupby("genbank_id")["cazy_family"]
    .apply(lambda x: ";".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"cazy_family": "cazy_cbm_family"})
)

# GH43 subfamilies per protein (e.g. "GH43_9", or "GH43_1;GH43_16" if multiple)
gh43_annot = (
    annot_df[annot_df["cazy_family"].str.match(r"^GH43_\d+$", na=False)]
    .groupby("genbank_id")["cazy_family"]
    .apply(lambda x: ";".join(sorted(set(x))))
    .reset_index()
    .rename(columns={"cazy_family": "gh43_subfamily"})
)

# ----------------------------------------------------------------------
# B. Unified headers: Mechanism + Clan
# ----------------------------------------------------------------------
print("Loading unified headers…")
header_df = pd.read_csv(args.headers, sep="\t", usecols=["cazyme_family", "Mechanism", "Clan"])

# ----------------------------------------------------------------------
# C. Characterized proteins
# ----------------------------------------------------------------------
print("Loading characterized proteins…")
char_df = pd.read_csv(args.char, sep="\t").rename(columns={
    "Domain":       "char_domain",
    "Protein Name": "char_protein_name",
    "EC":           "char_ec",
    "Organism":     "char_organism",
    "Uniprot":      "char_uniprot",
    "PDB/3D":       "char_pdb",
    "Subf":         "char_subf",
})
char_df["is_characterized"] = True
char_dedup = (
    char_df[["GenBank", "char_domain", "char_protein_name", "char_ec",
             "char_organism", "char_uniprot", "char_pdb", "char_subf", "is_characterized"]]
    .drop_duplicates("GenBank")
)

# ----------------------------------------------------------------------
# D. dbCAN domain annotations
# ----------------------------------------------------------------------
print("Loading dbCAN domain annotations…")
domain_df = pd.read_csv(args.domain_ann, sep="\t")

# ----------------------------------------------------------------------
# E. Merge all sources
# ----------------------------------------------------------------------
print("Merging…")
meta = (
    annot_dedup[["genbank_id", "domain", "cazy_class", "cazy_family",
                 "cazy_family_base", "organism", "taxid", "source", *LINEAGE_LEVELS]]
    .merge(cbm_annot,   on="genbank_id", how="left")
    .merge(gh43_annot,  on="genbank_id", how="left")
    .merge(header_df,  left_on="cazy_family_base", right_on="cazyme_family", how="left")
    .merge(char_dedup, left_on="genbank_id", right_on="GenBank", how="left")
    .merge(domain_df.rename(columns={"protein_id": "genbank_id_dom"}),
           left_on="genbank_id", right_on="genbank_id_dom", how="left")
)

meta["is_characterized"] = meta["is_characterized"].fillna(False)

meta.drop(columns=[c for c in ["cazyme_family", "GenBank", "genbank_id_dom"]
                   if c in meta.columns], inplace=True)

# Belt-and-suspenders: CBM entries must not appear in cazy_family/class/base
_cbm_mask = meta["cazy_family"].str.match(r"^CBM", na=False)
if _cbm_mask.any():
    meta.loc[_cbm_mask, ["cazy_family", "cazy_class", "cazy_family_base"]] = np.nan
    print(f"Stripped {_cbm_mask.sum()} residual CBM entries from cazy_family.")

# ----------------------------------------------------------------------
# F. Summary + write
# ----------------------------------------------------------------------
print(f"\nRows: {len(meta):,}  |  Columns: {len(meta.columns)}")
print(f"  Mechanism match rate:   {meta['Mechanism'].notna().mean():.1%}")
print(f"  Clan match rate:        {meta['Clan'].notna().mean():.1%}")
print(f"  Characterized rate:     {meta['is_characterized'].mean():.1%}")
print(f"  Domain annotation rate: {meta['domain_class'].notna().mean():.1%}" if "domain_class" in meta.columns else "  domain_class: not present")
print(f"  CBM annotation rate:    {meta['cazy_cbm_family'].notna().mean():.1%}")
print(f"  GH43 subfamily rate:    {meta['gh43_subfamily'].notna().mean():.1%}")

meta.to_csv(args.output, sep="\t", index=False)
print(f"\nSaved → {args.output}")
