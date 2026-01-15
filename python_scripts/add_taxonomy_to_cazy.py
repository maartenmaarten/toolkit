#!/usr/bin/env python3

"""
add_taxonomy_to_cazy.py

A script to add taxonomy information to CAZy data.
"""

import sys
import pandas as pd
import subprocess
import os
import re

def read_input_file(input_file):
    """
    Reads a CAZy text file and returns it as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(input_file, sep="\t", header=None)
        print(f"Input file {input_file} read successfully.")
        # Check if the DataFrame is empty
        print(f"DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)


def run_name_to_taxid(species_list, fuzzy=False):
    """
    Converts a list of species names to their corresponding NCBI Taxonomy IDs.
    """
    species_input = "\n".join(species_list)
    try:
        print(f"Running taxonkit name2taxid with fuzzy={fuzzy}...")
        # Run taxonkit name2taxid command
        result = subprocess.run(
            ["taxonkit", "name2taxid"] + (["--fuzzy"] if fuzzy else []),
            input=species_input,
            text=True,
            capture_output=True,
            check=True
        )
        taxid_dict = {}

        for line in result.stdout.strip().split("\n"):
            if not line:  # Skip empty lines < is this not a risk?
                continue
                
            parts = line.split("\t")
            if len(parts) == 2:
                species, taxid = parts
                taxid_dict[species] = taxid
            else:
                # No taxid found for this species
                print(f"Warning: No taxid found for '{line}'")
                taxid_dict[line] = ""  # Store empty taxid

        return taxid_dict
    except subprocess.CalledProcessError as e:
        print(f"Taxonkit name2taxid error: {e}", file=sys.stderr)
        sys.exit(1)



def get_taxid_online():
    """
    Retrieves taxid information from NCBI online.
    """
    pass


def taxid_to_lineage(taxid_list):
    """
    Converts a list of taxids to their corresponding lineage.
    """
    taxids_input = "\n".join(taxid_list)
    try:
        print("Running taxonkit lineage...")
        
        # First run taxonkit lineage
        result = subprocess.run(
            ["taxonkit lineage | taxonkit reformat -f '{k};{p};{c};{o};{f};{g};{s}'"],
            input=taxids_input,
            text=True,
            capture_output=True,
            check=True,
            shell=True
        )
        
        lineage_dict = {}

        for line in result.stdout.strip().split("\n"):
            if not line:  # Skip empty lines
                continue
                
            parts = line.split("\t")
            if len(parts) == 3: # Expecting taxid, lineage, reformatted lineage
                taxid, lineage, reformatted = parts
                lineage_dict[taxid] = lineage
            else:
                print(f"Warning: No lineage found for '{line}'")
                lineage_dict[line] = ""  # Store empty lineage

        return lineage_dict
    
    except subprocess.CalledProcessError as e:
        print(f"Taxonkit name2taxid error: {e}", file=sys.stderr)
        sys.exit(1)


def create_lookup_name(cazy_txt_df):
    """
    Creates a lookup name based on the taxonomy type.
    Cleans the species name before assigning to the lookup name.
    Removes everything in parentheses and the strings 'cf.' and 'afn.' 
    from the species name.
    """
    # Function to clean species names
    def clean_species_name(name):
        name = re.sub(r"\s*\(.*?\)", "", name)  # Remove parentheses
        # Remove 'cf.' and 'afn.' even if not surrounded by spaces
        name = re.sub(r"\bcf\.\b", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\bafn\.\b", "", name, flags=re.IGNORECASE)
        name = re.sub(r"cf\.", "", name, flags=re.IGNORECASE)
        name = re.sub(r"afn\.", "", name, flags=re.IGNORECASE)
        return name.strip()

    # Clean the species name column before creating lookup_name
    cleaned_species = cazy_txt_df.iloc[:, 2].apply(clean_species_name)

    # Create lookup_name column based on taxonomy type using cleaned species names
    cazy_txt_df["lookup_name"] = cazy_txt_df.apply(
        lambda row: " ".join(
            cleaned_species.iloc[row.name].split()[:2]
        )
        if row.iloc[1] in ["Eukaryota", "Bacteria", "Archaea"]
        else cleaned_species.iloc[row.name],  # when virus use full name
        axis=1
    )

    return cazy_txt_df


def check_for_missing_taxids(cazy_txt_df):
    """
    Checks for missing taxids in the DataFrame and returns a count of missing taxids.
    """
    missing_taxid_count = (cazy_txt_df["taxid"] == "").sum()
    if missing_taxid_count > 0:
        print(f"Number of missing taxids: {missing_taxid_count}", file=sys.stderr)
        return True
    

def sanity_check(cazy_txt_df):
    """    Performs a sanity check on the DataFrame to ensure it has the expected structure.
    """
    # Check for rows where column 1 is not contained in the lineage at column 7
    # mostly to prevent Ecoli Phage as being classified as bacteria
    # Check for rows where the value in column 1 (second column) is not present in the corresponding lineage (eighth column)
    invalid_rows = cazy_txt_df[~cazy_txt_df.apply(lambda row: str(row.iloc[1]) in str(row.iloc[8]), axis=1)]
    
    if not invalid_rows.empty:
        print(f"Sanity check failed: {len(invalid_rows)} rows have an invalid lineage.", file=sys.stderr)
        print("Invalid rows:")
        print(invalid_rows)
        sys.exit(1)
    else:
        print("Sanity check passed: All rows have valid lineages.")


def main():
    """
    Main function to execute the script.
    """
    cazy_txt_df = read_input_file(sys.argv[1])  # read cazy.txt
    cazy_txt_df = create_lookup_name(cazy_txt_df)  # create lookup_name column

    taxid_dict = run_name_to_taxid(cazy_txt_df["lookup_name"].tolist())  # convert lookup names to taxids
    print(f"Taxid dictionary:")
    for species, taxid in taxid_dict.items():
        print(f"{species}: {taxid}")
    cazy_txt_df["taxid"] = cazy_txt_df["lookup_name"].map(taxid_dict)  # update the "taxid" column with taxids mapped from "lookup_name"

    # Check if there are any missing taxids
    if check_for_missing_taxids(cazy_txt_df):
        # run name2taxid --fuzzy where taxid is missing
        taxid_dict = run_name_to_taxid(
            cazy_txt_df.loc[cazy_txt_df["taxid"] == "", 2].tolist(), # the '2' refers to the species name column
            fuzzy=True)  # Convert species names to taxids with fuzzy matching

        cazy_txt_df.loc[cazy_txt_df["taxid"] == "", "taxid"] = cazy_txt_df.loc[
        cazy_txt_df["taxid"] == "", 2].map(
            taxid_dict
        )  # Update missing taxids in the DataFrame

    if check_for_missing_taxids(cazy_txt_df):
        print("There are still missing taxids, running taxonkit on genus only...")
        # run name2taxid --fuzzy on first element of the lookup name where taxid is still missing
        genus_only = cazy_txt_df.loc[cazy_txt_df["taxid"] == "", "lookup_name"].apply(lambda x: x.split()[0])
        taxid_dict = run_name_to_taxid(genus_only.tolist(), fuzzy=True)
        cazy_txt_df.loc[cazy_txt_df["taxid"] == "", "taxid"] = genus_only.map(taxid_dict)


    # Check if there are still any missing taxids
    if check_for_missing_taxids(cazy_txt_df):
        print("Missing taxids:")
        missing_taxids_df = cazy_txt_df.loc[cazy_txt_df["taxid"] == "", [1, 2, "lookup_name"]]
        missing_taxids_df.to_csv("missing_taxids.csv", index=False, header=["accession","species_name", "lookup_name"])
        print(f"Missing taxids written to missing_taxids.csv")
        #print("There are still missing taxids after fuzzy matching. Exiting.", file=sys.stderr)
        # sys.exit(1)


    print(cazy_txt_df.head())  # Print the first few rows of the DataFrame for verification
    
    # run lineage
    print("Running taxonkit lineage...")
    lineage_dict = taxid_to_lineage(cazy_txt_df["taxid"].tolist())  # Convert taxids to lineages
    cazy_txt_df["lineage"] = cazy_txt_df["taxid"].map(lineage_dict)  # Update the DataFrame with lineages
    
    #sanity_check(cazy_txt_df)  # Perform sanity check on the DataFrame

    # Write the updated DataFrame to a new file
    output_file = sys.argv[2]
    cazy_txt_df.to_csv(output_file, sep="\t", index=False, header=False)

    #TODO almost all organisms, just some viruses and moths (?) left.

if __name__ == "__main__":
    main()