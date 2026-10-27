"""Shared plumbing for the *_embeddings.py scripts.

Each script only has to load its model and define an `embed_batch` function that
maps a list of sequences to per-residue tensors for the requested layers; this
module handles FASTA loading/cleaning, device selection, resumable checkpointing
and writing one {sequence_id: Tensor} .pt file per layer and kind (mean / full),
so ML_utilities.load_embeddings and dimensio_reductio.py can read them as-is.

Kept compatible with Python 3.7 / old torch so seqvec_embeddings.py can import it
from a legacy allennlp environment.
"""
from __future__ import annotations

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


class EmbeddingStore:
    """Resumable per-layer output files.

    For output_file "out/emb.pt" and layer 33 this writes out/emb_mean_layer33.pt
    (and out/emb_full_layer33.pt with full=True). A layer of None drops the
    "_layer..." part, for embeddings that have no layers (one-hot).
    """

    def __init__(self, output_file, layers, full=False):
        self.out = Path(output_file)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.layers = list(layers)
        self.full = full
        # Resume from existing files so interrupted runs don't restart from scratch
        self.mean_by_layer = {l: self._load(self.path("mean", l)) for l in self.layers}
        self.full_by_layer = {l: self._load(self.path("full", l)) for l in self.layers} if full else {}

    def path(self, kind, layer):
        tag = f"_{kind}" if layer is None else f"_{kind}_layer{layer}"
        return self.out.with_name(f"{self.out.stem}{tag}{self.out.suffix}")

    @staticmethod
    def _load(path):
        return torch.load(path, map_location="cpu") if path.exists() else {}

    def done_ids(self):
        """IDs present in every output file, so adding a layer or --full on a rerun
        recomputes what is missing."""
        dicts = list(self.mean_by_layer.values()) + list(self.full_by_layer.values())
        return set.intersection(*(set(d) for d in dicts))

    def add(self, seq_id, per_layer):
        """per_layer: {layer: (L, D) per-residue tensor}, special tokens already removed."""
        for layer in self.layers:
            # float32 on CPU, cloned so a slice doesn't drag the whole batch's
            # storage into the saved file.
            per_residue = per_layer[layer].detach().to("cpu", torch.float32).clone()
            if self.full:
                self.full_by_layer[layer][seq_id] = per_residue
            self.mean_by_layer[layer][seq_id] = per_residue.mean(dim=0)

    def __len__(self):
        return len(self.mean_by_layer[self.layers[-1]])

    def save(self):
        for layer in self.layers:
            if self.full:
                torch.save(self.full_by_layer[layer], self.path("full", layer))
            torch.save(self.mean_by_layer[layer], self.path("mean", layer))

    def report(self):
        print(f"\nSaved {len(self)} embeddings")
        for layer in self.layers:
            name = "embedding" if layer is None else f"layer {layer}"
            line = f"  {name}: {self.path('mean', layer)}  (mean)"
            if self.full:
                line += f"   {self.path('full', layer)}  (full)"
            print(line)


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
