"""One-hot baseline: per-residue (L, 20) matrices, columns in alphabetical
one-letter order (ACDEFGHIKLMNPQRSTVWY).

Not mean-pooled: averaging one-hot over positions only gives the amino-acid
composition. Output is {stem}_full.pt as {sequence_id: (L, 20) Tensor}; L varies
per sequence. No model, so it always runs on CPU and there are no layers.
"""
import torch
import typer

from plm_common import CANONICAL_AAS, DEFAULT_MAX_LEN, EmbeddingStore, embed_all, load_sequences

ALPHABET = sorted(CANONICAL_AAS)
AA_INDEX = {aa: i for i, aa in enumerate(ALPHABET)}


def one_hot(seq):
    idx = torch.tensor([AA_INDEX[aa] for aa in seq])
    return torch.nn.functional.one_hot(idx, num_classes=len(ALPHABET)).float()


def main(
    fasta_file: str,
    output_file: str,
    checkpoint_every: int = 10000,
    max_len: int = DEFAULT_MAX_LEN,
):
    sequences = load_sequences(fasta_file, max_len)
    print(f"Alphabet: {''.join(ALPHABET)} (full per-residue matrices)")

    store = EmbeddingStore(output_file, [None], full=True, mean=False)
    embed_all(
        sequences,
        lambda seqs: [{None: one_hot(s)} for s in seqs],
        store, batch_size=256, checkpoint_every=checkpoint_every,
    )


if __name__ == "__main__":
    typer.run(main)
