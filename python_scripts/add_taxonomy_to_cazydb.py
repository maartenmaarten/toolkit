import pandas as pd
import subprocess
import sys

'''
goal is to stick taxonomy to the right side of the cazydb
Do this based on species, which can be converted to taxid, to taxonomy
problem: species is not formatted universally

Bacteria & Eukaryota are mostly fine (always two words species)
Virusses are more annoying, for example:
Siamese algae-eater influenza-like virus SAEILV/fish-T1K
(takes the fish instead of the virus with [0:2])

proposed solution:
iterate over the max length of the species column (will usually be three)
if not, exclude one word from the end. Try again.
Only do on those with an empty taxonomy column

finally, for all those that could still not be mapped
do entrez search based on the ncbi/jgi number.
report final statistics

should also resolve some 'candidatus' cases.

(maybe add a test/checksum?)

question is if the first column is not virusses, can you not just default to [0:2]?
Only one causing trouble is virusses right? Do it this way might be a lot slower
but it is safer. 

quite a lot of virusses too btw
174118 Viruses
3493890 Bacteria
1011334 Eukaryotes
23629 Archaea




input: cazy_data.txt
output: cazy_data_taxonomy.txt
note that an intermediate file of taxonomy is probably needed.
only IF it has exactly the same size and the columns are the same
you can merge with join sideways.

'''


def get_species_list(cazy_data_df, iteration):
    # extract the species column
    species = cazy_data_df["species"]
    species_list = []
    # its very important to keep the order of the species
    for sp in species:
        sp_split = sp.split()
        # make sure that the species is at least two words
        if len(sp_split) - iteration >= 2:
            species_list.append(" ".join(sp_split[:len(sp_split)-iteration]))
        else:
            species_list.append(" ".join(sp_split[:2]))
    return species_list
    


def run_taxonkit(species_list):
    result = subprocess.run(
        ["taxonkit", "name2taxid"],
        input="\n".join(species_list),
        capture_output=True,
        text=True,
        check=True
    )
    taxid_list = []
    for line in result.stdout.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) > 1:
            taxid_list.append(parts[1])
        else:
            taxid_list.append("N/A")  # Handle cases where taxid is not found
    return list(zip(species_list, taxid_list))


def main():
    cazy_data_df = pd.read_csv(sys.argv[1], delimiter="\t", 
                               names=["cazy_family", "kingdom", "species", "protein_id", "source"])
    species_list = get_species_list(cazy_data_df, 0)
    taxid_list = run_taxonkit(species_list)
    print(taxid_list)
    cazy_data_df['taxid'] = taxid_list
    print(cazy_data_df["species","taxid"])
    #new_cazy_data_df = cazy_data_df[cazy_data_df['taxid'] != '']
    #print(new_cazy_data_df)


if __name__ == "__main__":
    main()
