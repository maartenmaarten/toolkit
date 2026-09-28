#!/usr/bin/env python3
"""Check CAZy family agreement between the general cazy_data dump and a consolidated per-family file.

Usage:
    python check_cazy_family_consistency.py [--general PATH] [--family-specific PATH] [--outdir DIR]

Inputs:
    general file           headerless, tab-separated: family, domain, organism,
                            accession(.version), source. One row per (accession, family).
    family-specific file   tab-separated with header (Family, Domain, Species,
                            GenBank, Source). One row per accession; Family (and
                            Species) is either a single value or a Python-list-repr
                            string (e.g. "['GH0', 'GH43']") when a protein hits
                            multiple families.

Both files are compared by GenBank accession, matched exactly as given (no
version stripping - the two dumps already agree on accession version for
~97% of the accession overlap, so stripping would add noise, not signal).

The family-specific file also has a handful of malformed rows (an embedded
HTML 404 page ended up inline in the TSV); these are dropped by requiring
Source to be "ncbi" or "jgi".

Note: JGI protein IDs are only unique within a single JGI genome, not
globally, so the same numeric ID can legitimately refer to unrelated proteins
across different genomes. Both files share this ambiguity equally (an ID's
family set is the union across every genome that reused it), so it doesn't
by itself create false mismatches between the two files - but it does mean
a handful of JGI-sourced family lists represent several unrelated proteins,
not one.

Writes to --outdir:
    family_mismatch.tsv        accessions present in both files where the sorted
                                family set disagrees
    general_only.tsv           accessions in the general dump, absent from the
                                family-specific file (coverage gap)
    family_specific_only.tsv   accessions in the family-specific file, absent
                                from the general dump (coverage gap)
Prints row-count stats (incl. per-source coverage) to stderr.
"""
import argparse
import sys
from pathlib import Path

import polars as pl

GENERAL_COLUMNS = ["family", "domain", "organism", "accession", "source"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--general",
        default="/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/cazy_data_20260910.txt",
    )
    ap.add_argument(
        "--family-specific",
        default="/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/2026-09-18_download/consolidated/sequences_ALL.tsv",
    )
    ap.add_argument("--outdir", default=".")
    arguments = ap.parse_args()

    outdir = Path(arguments.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    general = (
        pl.scan_csv(
            arguments.general, has_header=False, separator="\t",
            new_columns=GENERAL_COLUMNS, infer_schema=False, quote_char=None,
        )
        .group_by("accession")
        .agg(
            pl.col("family").unique().sort().alias("general_families"),
            pl.col("domain").first().alias("general_domain"),
            pl.col("source").first().alias("general_source"),
        )
    )

    family_specific_raw = pl.scan_csv(
        arguments.family_specific, has_header=True, separator="\t",
        infer_schema=False, quote_char=None,
    )
    # Drop malformed rows (an embedded HTML 404 page ended up inline in this file).
    family_specific = (
        family_specific_raw.filter(
            pl.col("Source").str.to_lowercase().is_in(["ncbi", "jgi"])
        )
        .rename({"GenBank": "accession"})
        .with_columns(
            # extract_all returns the full match, not a capture group, so match the
            # bare code directly rather than trying to strip the surrounding quotes.
            pl.when(pl.col("Family").str.starts_with("["))
            .then(pl.col("Family").str.extract_all(r"[A-Za-z0-9_]+"))
            .otherwise(pl.col("Family").str.split(","))
            .alias("fs_family_list")
        )
        .explode("fs_family_list")
        .group_by("accession")
        .agg(
            pl.col("fs_family_list").unique().sort().alias("family_specific_families"),
            pl.col("Domain").first().alias("fs_domain"),
            pl.col("Source").first().alias("fs_source"),
        )
    )

    merged = general.join(
        family_specific, on="accession", how="full", coalesce=True
    ).collect(engine="streaming")

    in_general = pl.col("general_families").is_not_null()
    in_family_specific = pl.col("family_specific_families").is_not_null()

    matched = merged.filter(in_general & in_family_specific)
    general_only = merged.filter(in_general & ~in_family_specific).select(
        "accession", "general_domain", "general_source", "general_families"
    )
    family_specific_only = merged.filter(~in_general & in_family_specific).select(
        "accession", "fs_domain", "fs_source", "family_specific_families"
    )

    family_mismatch = matched.filter(
        pl.col("general_families") != pl.col("family_specific_families")
    )

    # CSV can't hold list-typed columns; flatten the family sets to ";"-joined strings for output.
    family_mismatch.with_columns(
        pl.col("general_families").list.join(";"),
        pl.col("family_specific_families").list.join(";"),
    ).write_csv(outdir / "family_mismatch.tsv", separator="\t")
    general_only.with_columns(
        pl.col("general_families").list.join(";")
    ).write_csv(outdir / "general_only.tsv", separator="\t")
    family_specific_only.with_columns(
        pl.col("family_specific_families").list.join(";")
    ).write_csv(outdir / "family_specific_only.tsv", separator="\t")

    total_general = matched.height + general_only.height
    total_family_specific = matched.height + family_specific_only.height

    def source_breakdown(df, source_col):
        counts = df.group_by(source_col).len()
        return dict(zip(counts[source_col].to_list(), counts["len"].to_list()))

    general_only_by_source = source_breakdown(general_only, "general_source")
    family_specific_only_by_source = source_breakdown(family_specific_only, "fs_source")

    print(
        f"general dump accessions: {total_general}\n"
        f"family-specific file accessions: {total_family_specific}\n"
        f"matched accessions: {matched.height}\n"
        f"coverage gap - in general dump only: {general_only.height} "
        f"(by source: {general_only_by_source})\n"
        f"coverage gap - in family-specific file only: {family_specific_only.height} "
        f"(by source: {family_specific_only_by_source})\n"
        f"family set mismatches (of matched accessions): {family_mismatch.height}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
