"""SeqVec (ELMo trained on UniRef50) embeddings.

SeqVec needs allennlp, which does not install alongside a current torch /
Python 3.12, so run this from its own environment, e.g.
    conda create -n seqvec python=3.8
    conda activate seqvec
    pip install "allennlp==2.10.1" biopython typer tqdm
(allennlp 0.9 also works; only allennlp.modules.elmo is used.)

Weights: download seqvec.zip from https://github.com/rostlab/SeqVec (README),
unzip, and pass the folder holding options.json and weights.hdf5 as --model-dir.

Layers: 0 = character CNN, 1 = first LSTM, 2 = second LSTM (all 1024-dim).
"sum" = 0 + 1 + 2, which mean-pooled is the standard SeqVec per-protein embedding
and is the default output.
"""
from pathlib import Path

import typer
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids

from plm_common import (
    DEFAULT_MAX_LEN, OUTPUT_HELP, EmbeddingStore, embed_all, load_sequences, pick_device,
)


def main(
    fasta_file: str,
    output: str = typer.Argument(..., help=OUTPUT_HELP),
    model_dir: str = typer.Option(..., help="Folder with options.json and weights.hdf5"),
    checkpoint_every: int = 500,
    all_layers: bool = typer.Option(False, help="Also save layers 0, 1 and 2 separately"),
    full: bool = False,
    batch_size: int = 8,
    max_len: int = DEFAULT_MAX_LEN,
    device: str = typer.Option(None, help="Override automatic cuda > mps > cpu choice"),
):
    options_file = Path(model_dir) / "options.json"
    weight_file = Path(model_dir) / "weights.hdf5"
    for f in (options_file, weight_file):
        if not f.exists():
            raise typer.BadParameter(f"{f} not found", param_hint="--model-dir")

    sequences = load_sequences(fasta_file, max_len)
    dev = pick_device(device)

    model = _ElmoBiLm(str(options_file), str(weight_file), requires_grad=False).to(dev).eval()

    layers = [0, 1, 2, "sum"] if all_layers else ["sum"]
    print(f"SeqVec: 3 layers; saving layers {layers} ({'full + mean' if full else 'mean only'})")

    def embed_batch(seqs):
        char_ids = batch_to_ids([list(s) for s in seqs]).to(dev)
        activations = model(char_ids)["activations"]  # 3 x (B, T + 2, 1024)
        out = []
        for i, s in enumerate(seqs):
            # <S> SEQ </S> [pad ...]
            per_layer = {l: activations[l][i, 1 : len(s) + 1] for l in (0, 1, 2)}
            per_layer["sum"] = per_layer[0] + per_layer[1] + per_layer[2]
            out.append({l: per_layer[l] for l in layers})
        return out

    # SeqVec has a single size (93M parameters); "sum" is the untagged default layer
    store = EmbeddingStore(output, "seqvec", "93m", layers, "sum", full)
    embed_all(sequences, embed_batch, store, batch_size, checkpoint_every)


if __name__ == "__main__":
    typer.run(main)
