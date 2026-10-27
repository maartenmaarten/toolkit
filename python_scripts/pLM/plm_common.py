"""Shared plumbing for the *_embeddings.py scripts.

Each script only has to load its model and define an `embed_batch` function that
maps a list of sequences to per-residue tensors for the requested layers; this
module handles FASTA loading/cleaning, device selection, resumable checkpointing
and writing one {sequence_id: Tensor} .pt file per layer and kind (mean / full),
named OUTPUT.<model>.<mean|full>.<size>[.layerN].pt, so
ML_utilities.load_embeddings and dimensio_reductio.py can read them as-is.

Kept compatible with Python 3.7 / old torch so seqvec_embeddings.py can import it
from a legacy allennlp environment.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import torch
from Bio import SeqIO
from tqdm import tqdm

CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_MAX_LEN = 2000


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr)


def load_sequences(fasta_file, max_len=DEFAULT_MAX_LEN):
    """Read a FASTA and return [(id, SEQUENCE)], uppercased with a trailing stop
    '*' removed, dropping sequences longer than max_len or containing anything
    outside the 20 canonical amino acids."""
    raw = [(r.id, str(r.seq).upper().rstrip("*")) for r in SeqIO.parse(fasta_file, "fasta")]
    print(f"Loaded {len(raw)} sequences")

    dupes = [k for k, n in Counter(i for i, _ in raw).items() if n > 1]
    if dupes:
        warn(f"{len(dupes)} duplicate sequence IDs (e.g. {dupes[0]!r}); only the last one of each is kept")

    n_too_long = sum(1 for _, s in raw if len(s) > max_len)
    n_noncanonical = sum(1 for _, s in raw if len(s) <= max_len and not set(s) <= CANONICAL_AAS)
    n_empty = sum(1 for _, s in raw if not s)
    sequences = [(i, s) for i, s in raw if s and len(s) <= max_len and set(s) <= CANONICAL_AAS]
    print(f"Filtered {n_too_long} sequences > {max_len} AA")
    print(f"Filtered {n_noncanonical} sequences with non-canonical amino acids")
    if n_empty:
        print(f"Filtered {n_empty} empty sequences")
    print(f"Remaining: {len(sequences)} sequences")
    return sequences


def pick_device(requested=None):
    """CUDA if available, else MPS, else CPU; warns on either fallback.
    `requested` (e.g. "cpu", "cuda:1") overrides the automatic choice."""
    if requested:
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        mps = getattr(torch.backends, "mps", None)  # absent in torch < 1.12
        if mps is not None and mps.is_available():
            warn("CUDA not available, falling back to MPS")
            device = torch.device("mps")
        else:
            warn("Neither CUDA nor MPS available, falling back to CPU; this will be slow")
            device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def select_layers(n_total, n_layers=1, all_layers=False):
    """Indices of the last `n_layers` of `n_total` layers, or all of them."""
    if all_layers:
        return list(range(n_total))
    if not 1 <= n_layers <= n_total:
        raise ValueError(f"n_layers must be between 1 and {n_total}, got {n_layers}")
    return list(range(n_total - n_layers, n_total))


def size_tag(text):
    """Make a model/size label safe for the dot-separated file name."""
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")


def esm_size_tag(model_name):
    """'facebook/esm2_t33_650M_UR50D' -> '650m'."""
    m = re.search(r"_t\d+_(\d+[MB])_", model_name)
    return m.group(1).lower() if m else size_tag(model_name.split("/")[-1])


OUTPUT_HELP = (
    "Output prefix (may include a directory); files are written as "
    "OUTPUT.<model>.<mean|full>.<size>[.layerN].pt"
)


class EmbeddingStore:
    """Resumable per-layer output files.

    Files are named OUTPUT.<model>.<kind>.<size>[.layerN].pt, e.g. with
    output "out/gh43", model "esm2" and size "650m":
        out/gh43.esm2.mean.650m.pt          default layer (the last one)
        out/gh43.esm2.full.650m.layer20.pt  any other layer, with full=True
    `default_layer` is the layer written without a ".layerN" tag; size=None
    (one-hot) leaves the size part out.
    """

    def __init__(self, output, model, size, layers, default_layer, full=False, mean=True):
        if not (full or mean):
            raise ValueError("EmbeddingStore needs full and/or mean output")
        prefix = Path(output)
        if prefix.suffix == ".pt":
            prefix = prefix.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.model = model
        self.size = size
        self.default_layer = default_layer
        self.layers = list(layers)
        self.kinds = (["mean"] if mean else []) + (["full"] if full else [])
        # Resume from existing files so interrupted runs don't restart from scratch
        self.data = {
            kind: {l: self._load(self.path(kind, l)) for l in self.layers}
            for kind in self.kinds
        }

    def path(self, kind, layer):
        parts = [self.prefix.name, self.model, kind]
        if self.size:
            parts.append(self.size)
        if layer != self.default_layer:
            parts.append(f"layer{layer}")
        return self.prefix.with_name(".".join(parts) + ".pt")

    @staticmethod
    def _load(path):
        return torch.load(path, map_location="cpu") if path.exists() else {}

    def done_ids(self):
        """IDs present in every output file, so adding a layer or --full on a rerun
        recomputes what is missing."""
        return set.intersection(*(set(d) for by_layer in self.data.values() for d in by_layer.values()))

    def add(self, seq_id, per_layer):
        """per_layer: {layer: (L, D) per-residue tensor}, special tokens already removed."""
        for layer in self.layers:
            # float32 on CPU, cloned so a slice doesn't drag the whole batch's
            # storage into the saved file.
            per_residue = per_layer[layer].detach().to("cpu", torch.float32).clone()
            if "full" in self.data:
                self.data["full"][layer][seq_id] = per_residue
            if "mean" in self.data:
                self.data["mean"][layer][seq_id] = per_residue.mean(dim=0)

    def __len__(self):
        return len(self.data[self.kinds[0]][self.layers[-1]])

    def save(self):
        for kind in self.kinds:
            for layer in self.layers:
                torch.save(self.data[kind][layer], self.path(kind, layer))

    def report(self):
        print(f"\nSaved {len(self)} embeddings")
        for layer in self.layers:
            name = "embedding" if layer is None else f"layer {layer}"
            print(f"  {name}: " + "   ".join(f"{self.path(k, layer)}  ({k})" for k in self.kinds))


def embed_all(sequences, embed_batch, store, batch_size=8, checkpoint_every=500):
    """Run `embed_batch` over all sequences not yet in `store`, checkpointing as it goes.

    embed_batch(list_of_sequences) must return one {layer: (L, D) tensor} per
    sequence, where L is the sequence length (special tokens and padding removed).
    """
    done = store.done_ids()
    if done:
        print(f"Resuming: {len(done)} embeddings already saved")
    todo = [(i, s) for i, s in sequences if i not in done]
    print(f"Sequences to embed: {len(todo)}")

    # Sort by length so sequences of similar size land in the same batch,
    # minimizing padding waste.
    todo.sort(key=lambda r: len(r[1]))

    n_since_checkpoint = 0
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(todo), batch_size), desc="Embedding", unit="batch"):
            batch = todo[batch_start : batch_start + batch_size]
            outputs = embed_batch([s for _, s in batch])
            for (seq_id, seq), per_layer in zip(batch, outputs):
                for layer, t in per_layer.items():
                    if t.shape[0] != len(seq):
                        raise RuntimeError(
                            f"{seq_id}: layer {layer} has {t.shape[0]} positions for a "
                            f"{len(seq)} AA sequence; special-token trimming is off"
                        )
                store.add(seq_id, per_layer)

            n_since_checkpoint += len(batch)
            if n_since_checkpoint >= checkpoint_every:
                store.save()
                tqdm.write(f"Checkpoint: {len(store)} embeddings saved")
                n_since_checkpoint = 0

    store.save()
    store.report()


def hf_embed_batch(model, tokenizer, seqs, layers, device, offset, prepare=None):
    """embed_batch for HuggingFace encoders called with output_hidden_states=True.

    hidden_states[0] is the embedding-layer output and hidden_states[k] the output
    of transformer layer k, so `layers` index 0..n_layers. `offset` is the number
    of special tokens before the first residue (1 for <cls>/[CLS], 0 for ProtT5);
    `prepare` reformats a sequence for the tokenizer (e.g. space-separating).
    """
    texts = [prepare(s) for s in seqs] if prepare else list(seqs)
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    hidden_states = model(**enc, output_hidden_states=True).hidden_states
    return [
        {l: hidden_states[l][i, offset : offset + len(s)] for l in layers}
        for i, s in enumerate(seqs)
    ]
