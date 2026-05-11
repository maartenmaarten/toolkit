import pandas as pd
import sys
import os

def load_cazy_tsv(file_path):
    """Load CAZy TSV file into a pandas DataFrame."""
    column_names = [
        'cazy_family', 'domain', 'organism', 'genbank_id', 'source'
    ]
    cazy_df = pd.read_csv(file_path, sep='\t', header=None,
                          names=column_names).drop_duplicates().reset_index(drop=True)
    print(cazy_df.head())
    print(f"Loaded CAZy data from {file_path} with {len(cazy_df)} entries.")
    return cazy_df


def add_lookup_name_column(cazy_df):
    """Add a 'lookup_name' column for taxonkit input."""
    print("Adding 'lookup_name' column for taxonkit...")
    cazy_df['lookup_name'] = cazy_df.apply(
        lambda row: ' '.join(str(row['organism']).split(' ')[0:2])
        if row['domain'] != 'Viruses' else row['organism'],
        axis=1
    )
    print(cazy_df[['organism', 'lookup_name']].head())
    return cazy_df

def main():
    if len(sys.argv) < 2:
        print("Usage: python write_lookup_column.py <cazy.txt>")
        sys.exit(1)
    input_file = sys.argv[1]
    cazy_df = load_cazy_tsv(input_file)
    cazy_df = add_lookup_name_column(cazy_df)
    cazy_df.to_csv(f"{input_file}_lookup_names_for_taxonkit.tsv", sep='\t', index=False, header=False)
    print(f"Wrote {input_file}_lookup_names_for_taxonkit.tsv for TaxonKit input.")

if __name__ == "__main__":
    main()
