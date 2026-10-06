import re
import pandas as pd


def merge_cazac_char(cazac_path: str, char_path: str) -> pd.DataFrame:
    cazac_df = pd.read_csv(cazac_path, sep='\t').drop_duplicates()
    characterized_df = pd.read_csv(char_path, sep='\t').drop_duplicates()

    multi_ec_mask = characterized_df['EC'].apply(lambda x: isinstance(x, list))
    cazac_multi_ec_mask = cazac_df['EC'].apply(lambda x: isinstance(x, list))
    cazac_df_clean = cazac_df[~cazac_multi_ec_mask]

    # ECs with multiple distinct cazac rows are ambiguous — treat like wildcards
    ec_counts = cazac_df_clean[~cazac_df_clean['EC'].str.endswith('.-', na=False)].groupby('EC').size()
    ambiguous_ecs = set(ec_counts[ec_counts > 1].index)
    if ambiguous_ecs:
        print(f"Ambiguous ECs (multiple distinct cazac entries, sent to name-matching pass): {sorted(ambiguous_ecs)}")

    unambiguous_cazac = cazac_df_clean[
        ~cazac_df_clean['EC'].str.endswith('.-', na=False) &
        ~cazac_df_clean['EC'].isin(ambiguous_ecs)
    ]

    combined_df = pd.merge(
        characterized_df[~multi_ec_mask],
        unambiguous_cazac,
        on=['EC'],
        how='left'
    )

    if multi_ec_mask.any():
        n = multi_ec_mask.sum()
        print(f"Warning: {n} entr{'y' if n == 1 else 'ies'} with multiple EC numbers skipped during merge and kept as unmatched.")
        combined_df = pd.concat([combined_df, characterized_df[multi_ec_mask]], ignore_index=True)

    unmatched_df = combined_df[combined_df['Activity Name'].isnull()]
    if not unmatched_df.empty:
        remaining_cazac_df = pd.concat([
            cazac_df_clean[cazac_df_clean['EC'].str.endswith('.-', na=False)],
            cazac_df_clean[cazac_df_clean['EC'].isin(ambiguous_ecs)],
        ])
        for idx, row in unmatched_df.iterrows():
            protein_name = row['Protein Name']
            for _, cazac_row in remaining_cazac_df.iterrows():
                activity = str(cazac_row['Activity Name'])
                activity = activity.replace('‐', '-').replace('‑', '-').replace('‒', '-').replace('–', '-').replace('—', '-')
                protein_name_norm = str(protein_name).replace('‐', '-').replace('‑', '-').replace('‒', '-').replace('–', '-').replace('—', '-')

                pattern = re.escape(activity)
                pattern = pattern.replace(r'a\-', r'(?:a|α)[-‐-―]?')
                pattern = pattern.replace(r'b\-', r'(?:b|β)[-‐-―]?')
                if re.search(pattern, protein_name_norm, re.IGNORECASE):
                    print(f'Match found for protein {protein_name} with activity {cazac_row["Activity Name"]}')
                    for col in remaining_cazac_df.columns:
                        combined_df.at[idx, col] = cazac_row[col]
                    break

    combined_df.sort_values(
        by=['Domain', 'EC'],
        key=lambda col: col.apply(lambda x: str(x) if isinstance(x, list) else x),
        inplace=True
    )
    print(f"{len(combined_df[combined_df['Activity Name'].isnull()])} entries have missing values after combining characterized and cazac data.")

    return combined_df


CAZAC_TSV = '/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/annotations/CAZy_cazac_ALL_2026-05-21.tsv'
CHAR_TSV  = '/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/annotations/CAZy_char_ALL_2026-05-21.tsv'
OUTPUT    = '/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/annotations/CAZy_cazac_char_ALL_2026-05-21.tsv'


def main():
    combined_df = merge_cazac_char(CAZAC_TSV, CHAR_TSV)
    combined_df.to_csv(OUTPUT, sep='\t', index=False)
    print(f"Merged table written to {OUTPUT} ({len(combined_df)} rows)")


if __name__ == '__main__':
    main()
