import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import typer
from pathlib import Path
from typing import List


def main(
    inputs: List[str] = typer.Argument(..., help="Input .pt files to merge"),
    output: str = typer.Option(..., "--output", "-o", help="Output .pt file"),
    overwrite_duplicates: bool = typer.Option(
        False, "--overwrite-duplicates",
        help="If a key appears in multiple files, keep the last occurrence (default: keep first)"
    ),
):
    merged = {}
    total_dupes = 0

    for path_str in inputs:
        path = Path(path_str)
        if not path.exists():
            typer.echo(f"ERROR: {path} not found", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading {path} …")
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, dict):
            typer.echo(f"ERROR: {path} does not contain a dict", err=True)
            raise typer.Exit(1)

        dupes = [k for k in data if k in merged]
        if dupes:
            total_dupes += len(dupes)
            typer.echo(
                f"  {len(dupes)} duplicate key(s) — "
                f"{'overwriting' if overwrite_duplicates else 'skipping'}"
            )

        for k, v in data.items():
            if k not in merged or overwrite_duplicates:
                merged[k] = v

        typer.echo(f"  {len(data)} embeddings  (running total: {len(merged)})")

    if total_dupes:
        typer.echo(f"\n{total_dupes} duplicate key(s) encountered across all files.")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)
    typer.echo(f"\nSaved {len(merged)} embeddings → {out_path}")


if __name__ == "__main__":
    typer.run(main)
