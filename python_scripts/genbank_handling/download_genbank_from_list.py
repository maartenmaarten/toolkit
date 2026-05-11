from Bio import Entrez
import argparse
import sys
import os
import tqdm

# Usage: python download_genbank_from_list.py <output_file.gb> [id_list.txt]
#        cat ids.txt | python download_genbank_from_list.py <output_file.gb>

def get_existing_ids(output_file):
    """Parse a GenBank file and return the set of versioned accession IDs already present."""
    import re
    existing = set()
    with open(output_file) as f:
        for line in f:
            # VERSION line looks like: "VERSION     NP_001234.1"
            m = re.match(r'^VERSION\s+(\S+)', line)
            if m:
                existing.add(m.group(1))
    return existing

def download_genbank_batch(genbank_ids, output_file, email, batch_size=200):
    Entrez.email = email

    # Resume: skip IDs already in the output file
    if os.path.exists(output_file):
        existing = get_existing_ids(output_file)
        missing = [id_ for id_ in genbank_ids if id_ not in existing]
        if not missing:
            print(f"All {len(genbank_ids)} IDs already present in {output_file}. Nothing to do.")
            return
        print(f"Resuming: {len(existing)} already downloaded, {len(missing)} remaining.")
        write_mode = "a"
        ids_to_fetch = missing
    else:
        write_mode = "w"
        ids_to_fetch = genbank_ids

    batches = [ids_to_fetch[i:i+batch_size] for i in range(0, len(ids_to_fetch), batch_size)]
    downloaded = 0
    with open(output_file, write_mode) as out_handle:
        with tqdm.tqdm(batches, desc="Downloading GenBank records", unit="batch") as pbar:
            for batch in pbar:
                try:
                    handle = Entrez.efetch(db="protein", id=",".join(batch), rettype="gb", retmode="text")
                    data = handle.read()
                    if not data.strip() or data.startswith("Error") or "<ERROR>" in data:
                        raise ValueError(f"Unexpected response: {data[:200]}")
                    records = data.split("\n//\n")
                    for record in records:
                        if record.strip():
                            out_handle.write(record.strip() + "\n//\n")
                            downloaded += 1
                    pbar.set_postfix({"downloaded": downloaded})
                except Exception as e:
                    print(f"\nFailed to download batch starting at {batch[0]}: {e}")

    actually_downloaded = get_existing_ids(output_file)
    failed_ids = [id_ for id_ in ids_to_fetch if id_ not in actually_downloaded]
    if failed_ids:
        print(f"Done. {downloaded} downloaded, {len(failed_ids)} failed:")
        for id_ in failed_ids:
            print(f"  {id_}")

def main():
    # Usage: python download_genbank_from_list.py <output_file.gb> [id_list.txt]
    #        cat ids.txt | python download_genbank_from_list.py <output_file.gb>
    parser = argparse.ArgumentParser(
        description="Download GenBank records from a list of accession IDs."
    )
    parser.add_argument("output_file", help="Output GenBank file (.gb)")
    parser.add_argument("id_list", nargs="?", help="File with one accession ID per line (or pipe via stdin)")
    parser.add_argument("--batch-size", type=int, default=200, help="Number of IDs to fetch per request (default: 200)")
    args = parser.parse_args()

    output_file = args.output_file
    email = os.environ.get("NCBI_EMAIL", "maarten.boneschansker@wur.nl")

    if args.id_list:
        with open(args.id_list) as f:
            genbank_ids = [line.strip() for line in f if line.strip()]
    elif not sys.stdin.isatty():
        genbank_ids = [line.strip() for line in sys.stdin if line.strip()]
    else:
        parser.error("provide an id list file or pipe IDs via stdin.")

    download_genbank_batch(genbank_ids, output_file, email, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
