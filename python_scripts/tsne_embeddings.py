import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import typer
import pandas as pd
from typing import Optional

from dimensio_reductio import PROJECTORS, plot_embedding

METHOD_NAMES = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP", "tmap": "TMAP"}


def main(
    embeddings_file: str,
    output_file: str,
    annotation_file: Optional[str] = typer.Option(None, "--annotation-file", help="Optional TSV file with annotations"),
    annotation_column: str = typer.Option("Domain", "--annotation-column", help="Column name in TSV to use for coloring"),
):
    """
    Load embeddings from a .pt file, project them to 2D with PCA, t-SNE and
    UMAP, and plot each projection as its own figure.
    Optionally color by annotations from a TSV file.

    t-SNE and UMAP, and the seaborn scatter plot, come from
    dimensio_reductio.py (same folder), so both scripts give the same maps:
    embeddings standardized per dimension, UMAP on cosine distance,
    unannotated points in light grey.

    Args:
        embeddings_file: Path to .pt file with embeddings
        output_file: Output plot file; one figure per method is written next
            to it as <name>_<method><ext>, e.g. plot.png -> plot_pca.png,
            plot_tsne.png, plot_umap.png
        annotation_file: Optional TSV file with annotations (must have ID column matching embedding IDs)
        annotation_column: Column name in TSV to use for coloring
    """
    # Load embeddings
    embeddings_dict = torch.load(embeddings_file)
    print(f"Loaded {len(embeddings_dict)} embeddings")

    # Convert to numpy array
    ids = list(embeddings_dict.keys())
    embeddings = torch.stack([embeddings_dict[id] for id in ids]).numpy()

    print(f"Embedding shape: {embeddings.shape}")

    # Apply PCA
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    projections = {"pca": (pca.fit_transform(embeddings), None)}
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # t-SNE and UMAP: (coordinates, edges = None) per method
    for method in ("tsne", "umap"):
        print(f"Running {METHOD_NAMES[method]}...")
        start = time.perf_counter()
        projections[method] = PROJECTORS[method](embeddings)
        print(f"  {METHOD_NAMES[method]} done in {time.perf_counter() - start:.1f}s")

    # Load annotations if provided
    color_by = None
    if annotation_file:
        print(f"Loading annotations from {annotation_file}...")
        ann_df = pd.read_csv(annotation_file, sep="\t")
        # Use GenBank column as ID column; the first row counts for repeated IDs
        id_col = "GenBank"
        annotations = ann_df.drop_duplicates(id_col).set_index(id_col)[annotation_column]
        color_by = pd.Series(ids).map(annotations)  # NaN where an ID has no annotation

        unique_colors = sorted(color_by.dropna().astype(str).unique())
        print(f"Found {len(unique_colors)} unique annotations: {unique_colors}")
        print(f"{color_by.isna().sum()} embedding(s) without annotation (plotted in grey)")

    # Plot: one figure per method
    sns.set_theme(style="whitegrid")
    output = Path(output_file)
    for method, (coords, edges) in projections.items():
        name = METHOD_NAMES[method]
        if method == "pca":
            xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
            ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
        else:
            xlabel, ylabel = f"{name} 1", f"{name} 2"
        title = f"{name} visualization ({len(embeddings)} embeddings)"
        if color_by is not None:
            title += f", colored by {annotation_column}"
        fig = plot_embedding(coords, xlabel, ylabel, color_by=color_by, title=title, edges=edges)

        plot_file = output.with_name(f"{output.stem}_{method}{output.suffix or '.png'}")
        fig.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {plot_file}")


if __name__ == "__main__":
    typer.run(main)
