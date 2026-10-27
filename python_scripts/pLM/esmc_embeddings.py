"""ESM C embeddings via the EvolutionaryScale `esm` package.

Layer numbering: 0..N-1 = transformer block outputs (N = 30 for 300M, 36 for
600M), no final norm applied.
"""
import typer
from esm.models.esmc import ESMC

from plm_common import (
    DEFAULT_MAX_LEN, OUTPUT_HELP, EmbeddingStore, embed_all, load_sequences, pick_device,
    select_layers,
)


def main(
    fasta_file: str,
    output: str = typer.Argument(..., help=OUTPUT_HELP),
    model_name: str = typer.Option("esmc_300m", "--model", help="esmc_300m or esmc_600m"),
    checkpoint_every: int = 500,
    n_layers: int = 1,
    all_layers: bool = False,
    full: bool = False,
    batch_size: int = 8,
    max_len: int = DEFAULT_MAX_LEN,
    device: str = typer.Option(None, help="Override automatic cuda > mps > cpu choice"),
):
    sequences = load_sequences(fasta_file, max_len)
    dev = pick_device(device)

    client = ESMC.from_pretrained(model_name).to(dev).eval()

    n_total = len(client.transformer.blocks)
    layers = select_layers(n_total, n_layers, all_layers)
    print(f"{model_name}: {n_total} layers; saving layers {layers} ({'full + mean' if full else 'mean only'})")

    def embed_batch(seqs):
        tokens = client._tokenize(seqs)
        hidden_states = client(tokens).hidden_states  # (n_layers, B, L, D)
        # <cls> SEQ <eos> [<pad> ...]
        return [
            {l: hidden_states[l, i, 1 : len(s) + 1] for l in layers}
            for i, s in enumerate(seqs)
        ]

    # "esmc_600m" -> "600m"
    store = EmbeddingStore(output, "esmc", model_name.split("_")[-1], layers, n_total - 1, full)
    embed_all(sequences, embed_batch, store, batch_size, checkpoint_every)


if __name__ == "__main__":
    typer.run(main)
