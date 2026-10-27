"""ProtBERT (Rostlab/prot_bert) embeddings.

Layer numbering: 0 = token embeddings, 1..30 = transformer layers.
"""
import typer
from transformers import BertModel, BertTokenizer

from plm_common import (
    DEFAULT_MAX_LEN, EmbeddingStore, embed_all, hf_embed_batch, load_sequences,
    pick_device, select_layers,
)

DEFAULT_MODEL = "Rostlab/prot_bert"


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
    device: str = typer.Option(None, help="Override automatic cuda > mps > cpu choice"),
):
    sequences = load_sequences(fasta_file, max_len)
    dev = pick_device(device)

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name, add_pooling_layer=False).to(dev).eval()

    n_total = model.config.num_hidden_layers + 1  # + embedding layer
    layers = select_layers(n_total, n_layers, all_layers)
    print(f"{model_name}: {n_total - 1} layers; saving layers {layers} ({'full + mean' if full else 'mean only'})")

    store = EmbeddingStore(output_file, layers, full)
    # ProtBERT expects space-separated residues: [CLS] M K T ... [SEP] [PAD ...]
    embed_all(
        sequences,
        lambda seqs: hf_embed_batch(model, tokenizer, seqs, layers, dev, offset=1, prepare=" ".join),
        store, batch_size, checkpoint_every,
    )


if __name__ == "__main__":
    typer.run(main)
