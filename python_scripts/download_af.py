"""
Download AlphaFold PDB structures and metadata for a list of UniProt IDs.

Usage:
    python download_af.py -o structures/ -m metadata.json P12345 Q8N158
    cat ids.txt | python download_af.py -o structures/ -m metadata.json
"""

import sys
import json
import argparse
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm


AF_API = "https://alphafold.ebi.ac.uk/api/prediction"

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]
)))


def fetch_metadata(uniprot_id: str) -> dict | None:
    response = session.get(f"{AF_API}/{uniprot_id}")
    if response.status_code == 404:
        return None
    response.raise_for_status()
    entries = response.json()
    return entries[0] if entries else None


def download_pdb(url: str, dest: Path) -> bool:
    response = session.get(url, stream=True)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    dest.write_bytes(response.content)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download AlphaFold PDB structures and metadata")
    parser.add_argument("ids", nargs="*", help="UniProt IDs (or pipe via stdin)")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory for PDB files")
    parser.add_argument("--metadata", "-m", required=True, help="Output JSON file for metadata")
    args = parser.parse_args()

    if args.ids:
        ids = args.ids
    elif not sys.stdin.isatty():
        ids = sys.stdin.read().split()
    else:
        parser.error("Provide UniProt IDs as arguments or pipe via stdin")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = {}
    failed = []

    for uniprot_id in tqdm(ids, desc="Downloading"):
        meta = fetch_metadata(uniprot_id)

        if meta is None:
            tqdm.write(f"  No AlphaFold entry: {uniprot_id}")
            failed.append(uniprot_id)
            continue

        all_metadata[uniprot_id] = meta

        pdb_url = meta.get("pdbUrl")
        if pdb_url:
            dest = output_dir / f"{uniprot_id}.pdb"
            ok = download_pdb(pdb_url, dest)
            if not ok:
                tqdm.write(f"  PDB download failed: {uniprot_id}")
        else:
            tqdm.write(f"  No PDB URL in metadata: {uniprot_id}")

    with open(args.metadata, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nDownloaded: {len(all_metadata)} structures -> {output_dir}/")
    print(f"Metadata:   {args.metadata}")
    if failed:
        print(f"No entry:   {len(failed)} IDs ({', '.join(failed[:5])}{'...' if len(failed) > 5 else ''})")


if __name__ == "__main__":
    main()
