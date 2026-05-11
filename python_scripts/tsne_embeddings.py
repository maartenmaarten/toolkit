import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import typer
import pandas as pd
from typing import Optional


def main(
    embeddings_file: str,
    output_file: str,
    annotation_file: Optional[str] = typer.Option(None, "--annotation-file", help="Optional TSV file with annotations"),
    annotation_column: str = typer.Option("Domain", "--annotation-column", help="Column name in TSV to use for coloring"),
):
    """
    Load embeddings from a .pt file, apply PCA, and plot.
    Optionally color by annotations from a TSV file.
    
    Args:
        embeddings_file: Path to .pt file with embeddings
        output_file: Output plot file
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
    
    # Subsample if needed
    if len(embeddings) > 1000:
        indices = np.random.choice(len(embeddings), 1000, replace=False)
        embeddings = embeddings[indices]
        ids = [ids[i] for i in indices]
        print(f"Subsampled to 1000 embeddings")
    
    numeric_colors = None  # Initialize for later use
    unique_colors = None
    
    # Apply PCA
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Load annotations if provided
    colors = None
    if annotation_file:
        print(f"Loading annotations from {annotation_file}...")
        ann_df = pd.read_csv(annotation_file, sep="\t")
        # Use GenBank column as ID column
        id_col = "GenBank"
        
        # Create color mapping
        colors = []
        for id in ids:
            match = ann_df[ann_df[id_col] == id]
            if len(match) > 0:
                colors.append(match[annotation_column].iloc[0])
            else:
                colors.append("unknown")
        
        # Create numeric color map
        unique_colors = sorted(set(colors))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        numeric_colors = [color_map[c] for c in colors]
        
        print(f"Found {len(unique_colors)} unique annotations: {unique_colors}")
    
    # Plot
    plt.figure(figsize=(14, 10))
    if colors:
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=numeric_colors,
            alpha=0.6,
            s=50,
            cmap="tab20",
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label(annotation_column)
        # Set colorbar ticks and labels
        cbar.set_ticks(range(len(unique_colors)))
        cbar.set_ticklabels(unique_colors, fontsize=8)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
    
    plt.title(f"PCA visualization ({len(embeddings)} embeddings)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    typer.run(main)
