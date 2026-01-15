#!/usr/bin/env python3

"""
clean_species_cazy.py
A script to clean species names in CAZy data.
"""

import sys
import pandas as pd
import re
import subprocess
import os


def clean_species_names(df):
    """
    Cleans species names in the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with cleaned species names.
    """
    df["lookup_name"] = df.apply(
        lambda row: " ".join(row.iloc[2].split()[:2]) 
            if row.iloc[1] in ["Eukaryota", "Bacteria", "Archaea"]
            else row.iloc[2],
        axis=1
    )
    # remove any parentheses and stuff.
    df["lookup_name"] = df["lookup_name"].apply(lambda name: re.sub(r"\s*\(.*?\)", "", name))


    return df


def run_taxonkit(cleaned_file):
    """
    Runs taxonkit on the cleaned file to get taxids.
    """
    try:
        print(f"Running taxonkit on {cleaned_file}...")
        result = subprocess.run(
            ["taxonkit", "name2taxid", "--name-field", "6", cleaned_file, "|",
             "taxonkit", "lineage", "--id-field", "7"],
            text=True,
            capture_output=True,
            check=True
        )
        print("Taxonkit run successfully.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running taxonkit: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main function to execute the script.
    """
    cazy_txt_df = pd.read_csv(sys.argv[1], sep="\t", header=None)
    print(f"Input file {sys.argv[1]} read successfully.")
    print(cazy_txt_df)
    
    # Clean species names
    cazy_txt_df = clean_species_names(cazy_txt_df)
    print(cazy_txt_df)  # Print the full DataFrame for verification

    # Write the updated DataFrame to a new file
    output_file = sys.argv[2]
    cazy_txt_df.sort_values(by=cazy_txt_df.columns[0], inplace=True)
    cazy_txt_df.to_csv(output_file, sep="\t", index=False, header=False)


if __name__ == "__main__":
    main()