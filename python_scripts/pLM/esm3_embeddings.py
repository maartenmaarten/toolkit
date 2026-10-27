"""ESM3 (open small, 1.4B) sequence-only embeddings via the EvolutionaryScale `esm` package.

Differs from ESMC in two ways:
  * ESM3.forward() drops the per-layer hidden states and only returns the final
    (pre-norm) layer as `embeddings`, so per-layer outputs are captured with
    forward hooks on the transformer blocks.
  * ESM3 is multimodal. Only sequence tokens are given here; the structure,
    secondary-structure, SASA and function tracks are left at their mask/pad
    defaults, which is how ESM3 is run on sequence alone.

The weights are gated: accept the licence at
https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1 and run
`huggingface-cli login` before the first run.

Layer numbering matches esmc_embeddings.py: 0..N-1 = transformer block outputs
(N = 48), no final norm applied.
"""
import torch
import typer
from esm.models.esm3 import ESM3
from esm.utils.constants.models import ESM3_OPEN_SMALL, normalize_model_name
from esm.utils.encoding import tokenize_sequence
from esm.utils.misc import stack_variable_length_tensors

from plm_common import (
    DEFAULT_MAX_LEN, OUTPUT_HELP, EmbeddingStore, embed_all, load_sequences, pick_device,
    select_layers, size_tag,
)


def main(
    fasta_file: str,
    output: str = typer.Argument(..., help=OUTPUT_HELP),
    model_name: str = typer.Option("esm3-open", "--model", help="ESM3 model name"),
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

    # from_pretrained casts to bfloat16 on any non-CPU device
    model = ESM3.from_pretrained(model_name, device=dev).eval()
    tokenizer = model.tokenizers.sequence
    pad_token_id = tokenizer.pad_token_id

    blocks = model.transformer.blocks
    layers = select_layers(len(blocks), n_layers, all_layers)
    print(f"{model_name}: {len(blocks)} layers; saving layers {layers} ({'full + mean' if full else 'mean only'})")

    captured = {}
    for layer_idx in layers:
        blocks[layer_idx].register_forward_hook(
            lambda module, inputs, output, layer_idx=layer_idx: captured.__setitem__(layer_idx, output)
        )

    def embed_batch(seqs):
        tokens = stack_variable_length_tensors(
            [tokenize_sequence(s, tokenizer, add_special_tokens=True) for s in seqs],
            constant_value=pad_token_id,
        ).to(dev)
        B, L = tokens.shape
        captured.clear()
        model(
            sequence_tokens=tokens,
            # Boolean sequence_id keeps padding from attending to real residues
            # (ESM3 applies no padding mask otherwise).
            sequence_id=tokens != pad_token_id,
            # forward() builds these with batch size 1 when omitted; give them
            # the real batch shape so geometric attention doesn't rely on broadcasting.
            structure_coords=torch.full((B, L, 3, 3), float("nan"), device=dev),
            chain_id=torch.zeros((B, L), dtype=torch.long, device=dev),
        )
        # <cls> SEQ <eos> [<pad> ...]
        return [
            {l: captured[l][i, 1 : len(s) + 1] for l in layers}
            for i, s in enumerate(seqs)
        ]

    # esm3_sm_open_v1 (1.4B) is the only ESM3 that runs locally
    size = "sm-open" if normalize_model_name(model_name) == ESM3_OPEN_SMALL else size_tag(model_name)
    store = EmbeddingStore(output, "esm3", size, layers, len(blocks) - 1, full)
    embed_all(sequences, embed_batch, store, batch_size, checkpoint_every)


if __name__ == "__main__":
    typer.run(main)
