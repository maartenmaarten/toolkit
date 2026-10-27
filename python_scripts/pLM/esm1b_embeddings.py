"""ESM-1b embeddings via HuggingFace transformers.

ESM-1b uses learned absolute position embeddings for 1024 tokens, so sequences
longer than 1022 AA (+ <cls>/<eos>) cannot be embedded and are filtered out.

Layer numbering follows fair-esm's repr_layers: 0 = token embeddings,
1..33 = transformer layers, 33 is post final layer norm.
"""
import typer
from transformers import AutoTokenizer, EsmModel

from plm_common import (
    EmbeddingStore, embed_all, hf_embed_batch, load_sequences, pick_device, select_layers,
)

DEFAULT_MODEL = "facebook/esm1b_t33_650M_UR50S"
ESM1B_MAX_LEN = 1022


def main(
    fasta_file: str,
    output_file: str,
    model_name: str = typer.Option(DEFAULT_MODEL, "--model", help="HuggingFace model id"),
    checkpoint_every: int = 500,
    n_layers: int = 1,
    all_layers: bool = False,
    full: bool = False,
    batch_size: int = 8,
    max_len: int = typer.Option(ESM1B_MAX_LEN, help=f"At most {ESM1B_MAX_LEN} (model limit)"),
    device: str = typer.Option(None, help="Override automatic cuda > mps > cpu choice"),
):
    if max_len > ESM1B_MAX_LEN:
        raise typer.BadParameter(f"ESM-1b handles at most {ESM1B_MAX_LEN} residues", param_hint="--max-len")

    sequences = load_sequences(fasta_file, max_len)
    dev = pick_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name, add_pooling_layer=False).to(dev).eval()

    n_total = model.config.num_hidden_layers + 1  # + embedding layer
    layers = select_layers(n_total, n_layers, all_layers)
    print(f"{model_name}: {n_total - 1} layers; saving layers {layers} ({'full + mean' if full else 'mean only'})")

    store = EmbeddingStore(output_file, layers, full)
    # <cls> SEQ <eos> [<pad> ...]
    embed_all(
        sequences,
        lambda seqs: hf_embed_batch(model, tokenizer, seqs, layers, dev, offset=1),
        store, batch_size, checkpoint_every,
    )


if __name__ == "__main__":
    typer.run(main)
