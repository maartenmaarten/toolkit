import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import typer
from pathlib import Path


def main(
    input_file: str,
    output_file: str,
    n: int = typer.Option(10000, "--n", help="Number of embeddings to sample"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
):
    rng = np.random.default_rng(seed)

    print(f"Loading {input_file}…")
    tensor = torch.load(input_file, map_location="cpu")
    keys = list(tensor.keys())
    print(f"Loaded {len(keys)} embeddings")

    if len(keys) <= n:
        print(f"Fewer than {n} embeddings present, writing all.")
        torch.save(tensor, output_file)
        return

    sel = rng.choice(len(keys), size=n, replace=False)
    subset = {keys[i]: tensor[keys[i]] for i in sel}

    del tensor
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(subset, output_file)
    print(f"Saved {n} embeddings to {output_file}")


if __name__ == "__main__":
    typer.run(main)
