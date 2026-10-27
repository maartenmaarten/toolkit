"""ProtT5-XL-U50 embeddings (encoder only).

The default checkpoint is Rostlab's half-precision encoder export, which is what
they recommend for embedding. It runs in fp16 on CUDA and is upcast to fp32 on
MPS/CPU unless --half is given. Needs `sentencepiece` for the tokenizer.

Layer numbering: 0 = token embeddings, 1..24 = encoder layers, 24 is post final
layer norm.
"""
import torch
import typer
from transformers import T5EncoderModel, T5Tokenizer

from plm_common import (
    DEFAULT_MAX_LEN, EmbeddingStore, embed_all, hf_embed_batch, load_sequences,
    pick_device, select_layers,
)

# Rostlab/prot_t5_xl_uniref50 is the full fp32 encoder-decoder (~11 GB download)
DEFAULT_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"


def main(
    fasta_file: str,
    output_file: str,
    model_name: str = typer.Option(DEFAULT_MODEL, "--model", help="HuggingFace model id"),
    checkpoint_every: int = 500,
    n_layers: int = 1,
    all_layers: bool = False,
    full: bool = False,
    batch_size: int = 8,
    max_len: int = DEFAULT_MAX_LEN,
    half: bool = typer.Option(None, help="fp16 weights; default on for CUDA, off otherwise"),
    device: str = typer.Option(None, help="Override automatic cuda > mps > cpu choice"),
):
    sequences = load_sequences(fasta_file, max_len)
    dev = pick_device(device)
    if half is None:
        half = dev.type == "cuda"

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(model_name)
    model = (model.half() if half else model.float()).to(dev).eval()
    print(f"Weights in {'fp16' if half else 'fp32'}")

    n_total = model.config.num_layers + 1  # + embedding layer
    layers = select_layers(n_total, n_layers, all_layers)
    print(f"{model_name}: {n_total - 1} layers; saving layers {layers} ({'full + mean' if full else 'mean only'})")

    store = EmbeddingStore(output_file, layers, full)
    # ProtT5 expects space-separated residues and only appends </s>: M K T ... </s> [<pad> ...]
    embed_all(
        sequences,
        lambda seqs: hf_embed_batch(model, tokenizer, seqs, layers, dev, offset=0, prepare=" ".join),
        store, batch_size, checkpoint_every,
    )


if __name__ == "__main__":
    typer.run(main)
