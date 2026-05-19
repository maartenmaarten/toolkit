from Bio import Entrez
import argparse
import json
import re
import sys
import os
import time
import tqdm

# Usage: python download_genbank_from_list.py <output_file.gb> [id_list.txt]
#        cat ids.txt | python download_genbank_from_list.py <output_file.gb>

def parse_ids(text):
    """Parse a string of IDs in either JSON-array or one-per-line format.

    Handles:  ["id1","id2"]  |  ['id1','id2']  |  id1\\nid2\\n  |  "id1"\\n"id2"
    """
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
            return [str(i).strip() for i in items if str(i).strip()]
        except json.JSONDecodeError:
            # Single-quoted or otherwise non-standard — extract quoted tokens via regex
            tokens = re.findall(r"['\"]([^'\"]+)['\"]", stripped)
            if tokens:
                return tokens
    # Fall back to line-by-line; strip surrounding quotes from each token
    ids = []
    for line in stripped.splitlines():
        token = line.strip().strip("\",'")
        if token:
            ids.append(token)
    return ids


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

def _fetch_batch(batch):
    """Fetch one batch from NCBI and return raw text, raising on bad response."""
    handle = Entrez.efetch(db="protein", id=",".join(batch), rettype="gb", retmode="text")
    data = handle.read()
    if not data.strip() or data.startswith("Error") or "<ERROR>" in data:
        raise ValueError(f"Unexpected response: {data[:200]}")
    return data


def download_genbank_batch(genbank_ids, output_file, email, batch_size=200,
                           sleep=0.5, retry_sleep=10):
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
    n_ids = len(ids_to_fetch)
    n_batches = len(batches)
    est_seconds = n_batches * (batch_size / 2 + sleep)  # ~2 records/s per batch + inter-batch sleep
    est_minutes = est_seconds / 60
    print(f"\n  Records to fetch : {n_ids}")
    print(f"  Batch size       : {batch_size}  →  {n_batches} batch(es)")
    print(f"  Sleep between    : {sleep}s")
    print(f"  Estimated time   : ~{est_minutes:.1f} min  (assuming 2 records/s throughput)\n")
    downloaded = 0
    with open(output_file, write_mode) as out_handle:
        with tqdm.tqdm(batches, desc="Downloading GenBank records", unit="batch") as pbar:
            for batch in pbar:
                data = None
                try:
                    data = _fetch_batch(batch)
                except Exception as e:
                    print(f"\nBatch starting at {batch[0]} failed ({e}). "
                          f"Retrying after {retry_sleep}s...")
                    time.sleep(retry_sleep)
                    try:
                        data = _fetch_batch(batch)
                    except Exception as e2:
                        print(f"Retry also failed: {e2}. Skipping batch.")

                if data is not None:
                    records = data.split("\n//\n")
                    for record in records:
                        if record.strip():
                            out_handle.write(record.strip() + "\n//\n")
                            downloaded += 1
                    pbar.set_postfix({"downloaded": downloaded})

                time.sleep(sleep)

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
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between batches (default: 0.5 — NCBI allows ~3 req/s without an API key)")
    parser.add_argument("--retry-sleep", type=float, default=10,
                        help="Seconds to wait before retrying a failed batch (default: 10)")
    args = parser.parse_args()

    output_file = args.output_file
    email = os.environ.get("NCBI_EMAIL", "maarten.boneschansker@wur.nl")

    if args.id_list:
        with open(args.id_list) as f:
            genbank_ids = parse_ids(f.read())
    elif not sys.stdin.isatty():
        genbank_ids = parse_ids(sys.stdin.read())
    else:
        parser.error("provide an id list file or pipe IDs via stdin.")

    download_genbank_batch(genbank_ids, output_file, email,
                           batch_size=args.batch_size,
                           sleep=args.sleep,
                           retry_sleep=args.retry_sleep)

if __name__ == "__main__":
    main()
