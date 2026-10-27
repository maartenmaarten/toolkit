"""ESM-2 embeddings via HuggingFace transformers.

Uses the HF port rather than fair-esm: in this conda env fair-esm and the
EvolutionaryScale `esm` package (ESMC/ESM3) overwrite each other's `esm` module.

Layer numbering follows fair-esm's repr_layers: 0 = token embeddings,
1..N = transformer layers (N = 33 for the 650M model), N is post final layer norm.
"""
import typer
from transformers import AutoTokenizer, EsmModel

from plm_common import (
    DEFAULT_MAX_LEN, OUTPUT_HELP, EmbeddingStore, embed_all, esm_size_tag, hf_embed_batch,
    load_sequences, pick_device, select_layers,
)

# facebook/esm2_t6_8M_UR50D, _t12_35M_, _t30_150M_, _t33_650M_, _t36_3B_, _t48_15B_UR50D
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"


def main(
    fasta_file: str,
    output: str = typer.Argument(..., help=OUTPUT_HELP),
    model_name: str = typer.Option(DEFAULT_MODEL, "--model", help="HuggingFace model id"),
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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(dev).eval()

    n_total = model.config.num_hidden_layers + 1  # + embedding layer
    layers = select_layers(n_total, n_layers, all_layers)
    print(f"{model_name}: {n_total - 1} layers; saving layers {layers} ({'full + mean' if full else 'mean only'})")

    store = EmbeddingStore(output, "esm2", esm_size_tag(model_name), layers, n_total - 1, full)
    # <cls> SEQ <eos> [<pad> ...]
    embed_all(
        sequences,
        lambda seqs: hf_embed_batch(model, tokenizer, seqs, layers, dev, offset=1),
        store, batch_size, checkpoint_every,
    )


if __name__ == "__main__":
    typer.run(main)
