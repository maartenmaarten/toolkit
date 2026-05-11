import pandas as pd
import sys
import os


def load_cazy_tsv(file_path):
    """Load CAZy TSV file into a pandas DataFrame."""
    column_names = [
        'cazy_family', 'domain', 'organism', 'genbank_id', 'source'
    ]
    cazy_df = pd.read_csv(file_path, sep='\t', header=None, skiprows=1,
                          names=column_names).drop_duplicates().reset_index(drop=True)

    print(cazy_df.head())
    print(f"Loaded CAZy data from {file_path} with {len(cazy_df)} entries.")
    return cazy_df


def add_lookup_name_column(cazy_df):
    """Add 'lookup_name' column for taxonkit input; skip Viruses."""
    print("Adding 'lookup_name' column for taxonkit...")

    def make_lookup(row):
        org = str(row.get('organism', ''))
        dom = str(row.get('domain', ''))
        if dom.lower() in ('viruses', 'virus'):
            return ''
        parts = org.split()
        return ' '.join(parts[:2]) if parts else ''

    cazy_df['lookup_name'] = cazy_df.apply(make_lookup, axis=1)

    print(cazy_df[['organism', 'domain', 'lookup_name']].head())
    cazy_df.to_csv("lookup_names_for_taxonkit.tsv", sep='\t', index=False)
    return cazy_df


def annotate_taxonomy(cazy_df):
    """Annotate CAZy data with taxonomy information."""
    # Placeholder for taxonomy annotation logic
    unannotated_df = cazy_df["d__domain"].isnull()
    print(f"{len(unannotated_df)} entries have no valid taxid.")
    return cazy_df


def check_annotation_sanity(cazy_df):
    """check whether CAZy Domain column corresponds to expected d__{domain} and species matches."""
    for index, row in cazy_df.iterrows():
        domain = str(row.get("domain", "")).upper()
        if domain == 'UNCLASSIFIED':
            continue
        superkingdom = str(row.get("d__domain", "")).replace('d__', '').upper()
        if pd.notnull(row.get("d__domain")) and domain != superkingdom:
            print(f"Warning: domain Mismatch at {row['genbank_id']}: expected {row['domain']}, found {row['d__domain']}")

    for index, row in cazy_df.iterrows():
        species = normalize_species_name(row.get("s__species", ""))
        organism = normalize_species_name(row.get("organism", ""))
        if pd.notnull(species) and pd.notnull(organism) and species not in organism:
            print(f"Warning: species Mismatch at {row['genbank_id']}: expected {organism}, found {species}")
    return


def run_taxonkit_annotation(cazy_df):
    # placeholder but it has to be from file in the future.
    # python bindings fuck shit up.
    # just run the add_taxonomy_to_cazy.sh shell script, do not capture the output, it writes it itself.
    # use the lookup list as input
    os.system("/Users/maartenboneschansker/Documents/bunch_of_code/toolkit/shell_scripts/add_taxonomy_to_cazy.sh lookup_names_for_taxonkit.tsv")
    return


def normalize_species_name(name):
    """Remove brackets and extra whitespace from species names for matching."""
    if pd.isnull(name):
        return ""
    # Remove square brackets and strip whitespace
    return str(name).replace('[', '').replace(']', '').replace('s__', '').strip()


def annotation_statistics(cazy_df):
    """Print annotation statistics."""
    total_entries = len(cazy_df)
    annotated_entries = cazy_df["taxid"].notnull().sum()
    print(f"Total entries: {total_entries}")
    print(f"Annotated entries: {annotated_entries} ({(annotated_entries/total_entries)*100:.2f}%)")
    return


def print_bad_lines(function_file, expected_fields=18, num_to_show=20):
    """Print lines with incorrect field counts."""
    bad_lines = []
    
    print(f"\n{'='*80}")
    print(f"Analyzing {function_file} for bad lines (expected {expected_fields} fields)...")
    print(f"{'='*80}\n")
    
    with open(function_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num == 1:  # Skip header
                continue
            
            fields = line.rstrip('\n').split('\t')
            num_fields = len(fields)
            
            if num_fields != expected_fields:
                bad_lines.append((line_num, num_fields, line))
    
    if not bad_lines:
        print(f"✓ All lines have correct field count ({expected_fields} fields)")
        return
    
    print(f"Found {len(bad_lines)} bad lines:\n")
    for line_num, num_fields, line_content in bad_lines[:num_to_show]:
        print(f"Line {line_num}: {num_fields} fields (expected {expected_fields}, diff: {num_fields - expected_fields})")
        print(f"  Content (first 150 chars): {line_content[:150]}...")
        print()


def main():
    if len(sys.argv) < 3:
        print("Usage: python annotate_cazy.py <cazy.txt> <function_file.tsv>")
        sys.exit(1)
    input_file = sys.argv[1]
    function_file = sys.argv[2]
    cazy_df = load_cazy_tsv(input_file)
    # taxonomy annotation
    cazy_df = add_lookup_name_column(cazy_df)
    run_taxonkit_annotation(cazy_df)



    names = ['cazy_family',	'domain', 'organism', 'genbank_id', 'source', # cazy base
             'lookup_name', 'taxid', # taxonkit input output
             'd__domain', 'p__phylum', 'c__class', 'o__order', 
             'f__family', 'g__genus', 's__species', 'str__strain']

    annotated_cazy_df = pd.read_csv("lookup_names_for_taxonkit.tax.tsv", sep='\t', 
                                    header=0, names=names)

    #check_annotation_sanity(annotated_cazy_df)

    # Check for bad lines before trying to read
    print_bad_lines(function_file, expected_fields=18)

    # function
    try:
        function_names = ['Domain', 'Protein Name', 'EC', 'Reference', 'Organism', 'GenBank',
            'Uniprot', 'PDB/3D', 'Activity Id', 'Activity Name', 'Residue -2',
            'Bond -1', 'Residue -1', 'Reacting Bond', 'Residue +1', 'Bond +1',
            'Residue +2', 'Reactant'
        ]

        function_df = pd.read_csv(function_file, sep='\t', header=0, dtype=str, 
                                 quotechar='"', quoting=1, index_col=False, names=function_names)  # Don't use any column as index
        print(f"Loaded function annotation data from {function_file} with {len(function_df)} entries.")
        # Merge function annotations
        print(annotated_cazy_df["genbank_id"].head())
        print(function_df['GenBank'].head())


        annotated_cazy_df = annotated_cazy_df.merge(function_df, left_on='genbank_id', right_on='GenBank', how='left')
        print(f"After merging function annotations, data has {len(annotated_cazy_df[annotated_cazy_df['EC'].notnull()])} entries.")
    except Exception as e:
        print(f"Error function annotation failed: {e}")
        import traceback
        traceback.print_exc()

    annotation_statistics(annotated_cazy_df)

    annotated_cazy_df.to_csv(f"{os.path.splitext(os.path.basename(input_file))[0]}.tax.fun.tsv", sep='\t', index=False)
if __name__ == "__main__":
    main()