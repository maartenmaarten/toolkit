#!/usr/bin/env python3
"""Merge CAZy family assignments with NCBI/JGI taxonomy lineage; report match stats.

Usage:
    python check_cazy_taxonomy.py [--cazy PATH] [--taxonomy PATH] [--outdir DIR]

Both inputs are headerless, tab-separated:
    cazy file      family, domain, organism, accession(.version), source
    taxonomy file  accession(no version), accession(.version), taxid, domain,
                    kingdom, phylum, class, order, family, genus, species

Join key: the accession with its version suffix stripped (e.g. CAE01911.1 and
CAE01911.2 both key on CAE01911), since the two files are frequently pinned to
different record versions of the same protein. The original versioned
accession from each side is kept in the output so version drift stays visible.

The taxonomy file only covers NCBI accessions, so cazy rows labelled "JGI"
(source column) are dropped before merging; their count is still reported.

Writes to --outdir:
    cazy_taxonomy_merged.tsv   matched rows (cazy columns + taxonomy lineage)
    cazy_unmatched.tsv         NCBI cazy rows with no taxonomy hit
    taxonomy_unmatched.tsv     taxonomy rows with no cazy hit
    domain_mismatch.tsv        matched rows where cazy domain != taxonomy domain
                                (cazy "unclassified" vs taxonomy "NA" is not
                                flagged - that pairing is correct, not a mismatch)
    version_mismatch.tsv       matched rows where the two accession versions differ
    genus_mismatch.tsv         matched rows where cazy organism's first word
                                (its apparent genus) != taxonomy genus. Only
                                checked for Eukaryota/Bacteria - genus doesn't
                                apply cleanly to viruses (and is skipped for
                                anything else, incl. Archaea, to stay conservative).
                                Also skips "<Family> bacterium" organism names
                                (e.g. "Flavobacteriaceae bacterium"), where the
                                first word is a family, not a genus
    species_not_in_organism.tsv matched rows where taxonomy's full species
                                binomial is not a substring of the cazy organism
                                name
Prints row-count stats (incl. NCBI/JGI split) to stderr.
"""
import argparse
import sys
from pathlib import Path

import polars as pl

CAZY_COLUMNS = ["cazy_family", "cazy_domain", "organism", "cazy_accession", "source"]
TAXONOMY_COLUMNS = [
    "accession_base", "taxonomy_accession", "taxid", "taxonomy_domain",
    "kingdom", "phylum", "class", "order", "taxonomic_family", "genus", "species",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cazy",
        default="/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/cazy_data_20260910.txt",
    )
    ap.add_argument(
        "--taxonomy",
        default="/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/cazy.taxonomy.tsv",
    )
    ap.add_argument("--outdir", default=".")
    arguments = ap.parse_args()

    outdir = Path(arguments.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cazy_raw = pl.scan_csv(
        arguments.cazy, has_header=False, separator="\t",
        new_columns=CAZY_COLUMNS, infer_schema=False, quote_char=None,
    )
    taxonomy = pl.scan_csv(
        arguments.taxonomy, has_header=False, separator="\t",
        new_columns=TAXONOMY_COLUMNS, infer_schema=False, quote_char=None,
    )

    source_counts = (
        cazy_raw.group_by(pl.col("source").str.to_lowercase())
        .len()
        .collect(engine="streaming")
    )
    source_totals = dict(zip(source_counts["source"], source_counts["len"]))
    total_raw_cazy_rows = sum(source_totals.values())
    jgi_rows = source_totals.get("jgi", 0)
    ncbi_rows = source_totals.get("ncbi", 0)
    other_source_rows = total_raw_cazy_rows - jgi_rows - ncbi_rows

    # Taxonomy only covers NCBI accessions, so drop JGI-sourced cazy rows before merging.
    # Join on the version-stripped accession: the two files are frequently pinned
    # to different record versions of the same protein (e.g. CAE01911.1 vs .2).
    cazy = (
        cazy_raw.filter(pl.col("source").str.to_lowercase() == "ncbi")
        .with_columns(
            pl.col("cazy_accession").str.replace(r"\.\d+$", "").alias("accession_base")
        )
    )

    merged = (
        cazy.join(taxonomy, on="accession_base", how="full", coalesce=True)
        .collect(engine="streaming")
    )

    in_cazy = pl.col("cazy_family").is_not_null()
    in_taxonomy = pl.col("taxonomy_accession").is_not_null()

    matched = merged.filter(in_cazy & in_taxonomy).with_columns(
        pl.col("organism").str.extract(r"^(\S+)").alias("cazy_genus"),
        pl.col("organism").str.extract(r"^\S+\s+(\S+)").alias("cazy_organism_word2"),
    )
    cazy_unmatched = merged.filter(in_cazy & ~in_taxonomy).select(CAZY_COLUMNS)
    taxonomy_unmatched = merged.filter(~in_cazy & in_taxonomy).select(TAXONOMY_COLUMNS)

    # cazy_domain "unclassified" paired with taxonomy_domain "NA" is not a real
    # mismatch: taxonomy correctly has no lineage to report for an unclassified organism.
    unclassified_vs_na = (
        pl.col("cazy_domain").str.to_lowercase().str.contains("unclassified")
        & (pl.col("taxonomy_domain") == "NA")
    )
    domain_mismatch = matched.filter(
        (pl.col("cazy_domain") != pl.col("taxonomy_domain")) & ~unclassified_vs_na
    )
    version_mismatch = matched.filter(
        pl.col("cazy_accession") != pl.col("taxonomy_accession")
    )

    # Genus only applies cleanly to Eukaryota/Bacteria (viruses in particular
    # don't have a real genus rank), so restrict the check to those domains
    # and skip rows where taxonomy has no genus to compare against. Also skip
    # "<Family> bacterium"-style organism names (e.g. "Flavobacteriaceae
    # bacterium") - there the first word is a family, not a genus, so it's not
    # comparable to taxonomy's genus column.
    family_bacterium_pattern = (
        pl.col("cazy_organism_word2").str.to_lowercase() == "bacterium"
    )
    genus_checkable = matched.filter(
        pl.col("taxonomy_domain").is_in(["Eukaryota", "Bacteria"])
        & (pl.col("genus") != "NA")
        & ~family_bacterium_pattern
    )
    genus_mismatch = genus_checkable.filter(pl.col("cazy_genus") != pl.col("genus"))

    # Taxonomy's full species binomial should be findable inside cazy's organism string.
    species_checkable = matched.filter(pl.col("species") != "NA")
    species_mismatch = species_checkable.filter(
        ~pl.col("organism").str.contains(pl.col("species"), literal=True)
    )

    matched.write_csv(outdir / "cazy_taxonomy_merged.tsv", separator="\t")
    cazy_unmatched.write_csv(outdir / "cazy_unmatched.tsv", separator="\t")
    taxonomy_unmatched.write_csv(outdir / "taxonomy_unmatched.tsv", separator="\t")
    domain_mismatch.write_csv(outdir / "domain_mismatch.tsv", separator="\t")
    version_mismatch.write_csv(outdir / "version_mismatch.tsv", separator="\t")
    genus_mismatch.write_csv(outdir / "genus_mismatch.tsv", separator="\t")
    species_mismatch.write_csv(outdir / "species_not_in_organism.tsv", separator="\t")

    # cazy is many-to-one on accession_base (a protein can hit several CAZy
    # families), so `matched.height` counts annotation rows, not distinct
    # proteins. Taxonomy accession_base is unique, so unique-accession_base
    # counts give the true protein-level picture and must sum back to the
    # real taxonomy row total.
    ncbi_cazy_rows = matched.height + cazy_unmatched.height
    matched_proteins = matched.select(pl.col("accession_base").n_unique()).item()
    total_taxonomy_rows = matched_proteins + taxonomy_unmatched.height
    domain_mismatch_proteins = domain_mismatch.select(pl.col("accession_base").n_unique()).item()
    version_mismatch_proteins = version_mismatch.select(pl.col("accession_base").n_unique()).item()
    genus_mismatch_proteins = genus_mismatch.select(pl.col("accession_base").n_unique()).item()
    species_mismatch_proteins = species_mismatch.select(pl.col("accession_base").n_unique()).item()

    print(
        f"cazy source rows: {total_raw_cazy_rows} total "
        f"(ncbi: {ncbi_rows}, jgi: {jgi_rows}, other: {other_source_rows})\n"
        f"jgi rows dropped before merge (no JGI taxonomy available): {jgi_rows}\n"
        f"ncbi cazy annotation rows: {ncbi_cazy_rows} "
        f"(matched: {matched.height}, unmatched: {cazy_unmatched.height})\n"
        f"taxonomy rows (unique proteins): {total_taxonomy_rows} "
        f"(matched: {matched_proteins}, unmatched: {taxonomy_unmatched.height})\n"
        f"match rate: {matched.height / ncbi_cazy_rows:.1%} of ncbi cazy rows, "
        f"{matched_proteins / total_taxonomy_rows:.1%} of taxonomy proteins\n"
        f"domain mismatches: {domain_mismatch.height} rows / "
        f"{domain_mismatch_proteins} distinct proteins\n"
        f"accession version mismatches (matched on base, differ by version): "
        f"{version_mismatch.height} rows / {version_mismatch_proteins} distinct proteins\n"
        f"genus mismatches (Eukaryota/Bacteria only, checked {genus_checkable.height} rows): "
        f"{genus_mismatch.height} rows / {genus_mismatch_proteins} distinct proteins\n"
        f"species not found in cazy organism name (checked {species_checkable.height} rows): "
        f"{species_mismatch.height} rows / {species_mismatch_proteins} distinct proteins",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
