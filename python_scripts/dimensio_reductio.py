"""Project CAZyDB embeddings (PCA / t-SNE / UMAP / TMAP) and plot the results.

For each requested method: compute the 2D projection, cache the coordinates
as .npz, join the ids against the CAZy_characterized annotation table, then
render and save one scatter plot per coloring (none, Domain, EC, Subf) --
popping each one up on screen as it's made.
"""
import argparse
import ast
import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tmap import TMAP

DEFAULT_EMBEDDINGS = (
    "/Users/maartenboneschansker/Documents/PhD/projects/cazyme_deep_dive/"
    "GH43_deep_dive/results/ESMC/full_embeddings/GH43_char_300m_mean_layer21.pt"
)
DEFAULT_ANNOT_TSV = (
    "/Users/maartenboneschansker/Documents/PhD/projects/cazyme_data_collection/"
    "2026-06-29_download/CAZy_characterized/GH43_characterized.tsv"
)
DEFAULT_OUTPUT_DIR = "projections"

# Seed for t-SNE and UMAP, so the same embeddings give the same map every run
# (UMAP runs single-threaded when seeded)
RANDOM_STATE = 42

# Only this many of the most common colour categories get a colour (tab20's);
# points in any other category are grey, like unannotated ones
MAX_COLOURS = 20

# Marker shapes for plot_embedding's shape_by (all filled: seaborn can't mix
# filled and line markers); points without a shape value get an X.
SHAPE_MARKERS = ["o", "s", "D", "^", "v", "P", "<", ">", "p", "h"]
NO_SHAPE_MARKER = "X"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--umap", action="store_true")
    parser.add_argument("--tmap", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all methods")
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS,
                         help="Torch dict of embeddings, keyed by id (default: %(default)s)")
    parser.add_argument("--annot-tsv", default=DEFAULT_ANNOT_TSV,
                         help="CAZy_characterized TSV with GenBank/Domain/EC/Subf columns, used "
                              "to color the plots (default: %(default)s)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                         help="Where to write {method}.npz coordinate caches and "
                              "{method}{coloring}.png plots (default: %(default)s)")
    args = parser.parse_args()

    if args.all:
        args.pca = args.tsne = args.umap = args.tmap = True
    if not any([args.pca, args.tsne, args.umap, args.tmap]):
        parser.error("Specify at least one of --pca, --tsne, --umap, --tmap, or --all")
    return args


# ----------------------------------------------------------------------
# Load embeddings
# ----------------------------------------------------------------------
def load_embeddings(path: str) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.load(path, map_location="cpu")
    keys = list(tensor.keys())

    print("Raw sample headers:")
    for k in keys[:5]:
        print("  ", repr(k))
    print(f"Total sequences: {len(keys)}")

    n_subsample = 10**7
    if len(keys) > n_subsample:
        sel = np.random.choice(len(keys), n_subsample, replace=False)
        keys = [keys[i] for i in sel]

    ids = np.array([k.split("|")[0] for k in keys])
    embeddings = np.stack([
        tensor[k].float().numpy() if isinstance(tensor[k], torch.Tensor)
        else tensor[k]["per_sequence"].float().numpy()
        for k in keys
    ]).astype(np.float32)
    print(f"Embeddings: {embeddings.shape} ({embeddings.dtype})")
    return ids, embeddings


# ----------------------------------------------------------------------
# Projections
# ----------------------------------------------------------------------
def standardize(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def pca_project(embeddings):
    return PCA(n_components=2).fit_transform(standardize(embeddings))


def tsne_project(embeddings, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE)
    return tsne.fit_transform(standardize(embeddings))


def umap_project(embeddings, n_neighbors=20, min_dist=0.1, metric="cosine"):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                        random_state=RANDOM_STATE)
    return reducer.fit_transform(standardize(embeddings))


def tmap_project(embeddings, metric="cosine", n_neighbors=100):
    model = TMAP(metric=metric, n_neighbors=n_neighbors).fit(standardize(embeddings))
    return model, model.embedding_


PROJECTORS = {
    "pca": lambda emb: (pca_project(emb), None),
    "tsne": lambda emb: (tsne_project(emb), None),
    "umap": lambda emb: (umap_project(emb), None),
    "tmap": lambda emb: (lambda m: (m[1], m[0].tree_.edges))(tmap_project(emb)),
}


# ----------------------------------------------------------------------
# Annotations
# ----------------------------------------------------------------------
def _clean_ec(ec):
    """CAZy's EC column sometimes holds a Python-list repr for multifunctional
    enzymes (e.g. "['3.2.1.37', '3.2.1.55']"); flatten that to "3.2.1.37+3.2.1.55"
    so it reads as one legend label instead of a stray bracketed string."""
    if pd.isna(ec):
        return ec
    ec = str(ec).strip()
    if ec.startswith("["):
        try:
            return "+".join(str(item) for item in ast.literal_eval(ec))
        except (ValueError, SyntaxError):
            return ec
    return ec


def load_annotations(annot_tsv: str) -> pd.DataFrame:
    annot_df = pd.read_csv(annot_tsv, sep="\t", usecols=["GenBank", "Domain", "EC", "Subf"])
    annot_df["EC"] = annot_df["EC"].map(_clean_ec)
    # Subf is numeric (e.g. 1.0, 4.0); render as clean ints, keeping NaN as NaN.
    annot_df["Subf"] = annot_df["Subf"].apply(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    return annot_df.drop_duplicates("GenBank")


def build_meta(ids: np.ndarray, annot_df: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame({"id": ids}).merge(annot_df, left_on="id", right_on="GenBank", how="left")
    print(f"  Domain match rate: {m['Domain'].notna().mean():.1%}")
    print(f"  EC match rate:     {m['EC'].notna().mean():.1%}")
    print(f"  Subf match rate:   {m['Subf'].notna().mean():.1%}")
    return m


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def category_palette(cats):
    """{category: RGB} for the sorted, non-missing colour categories, from
    tab20 (so at most MAX_COLOURS of them)."""
    if len(cats) > MAX_COLOURS:
        raise ValueError(f"{len(cats)} colour categories; tab20 has only {MAX_COLOURS} colours")
    return dict(zip(cats, sns.color_palette("tab20", n_colors=len(cats))))


def shape_markers(shape_by, shape_label="shape"):
    """(shapes, {shape: matplotlib marker}) for the non-missing values of
    shape_by, most common first so it gets the first marker (a circle);
    missing values (key "nan") get NO_SHAPE_MARKER."""
    values = pd.Series(np.asarray(shape_by, dtype=object)).dropna().astype(str)
    shapes = list(values.value_counts().index)
    if len(shapes) > len(SHAPE_MARKERS):
        raise ValueError(f"{shape_label} has {len(shapes)} distinct values; "
                         f"at most {len(SHAPE_MARKERS)} can get their own marker shape")
    markers = dict(zip(shapes, SHAPE_MARKERS))
    markers["nan"] = NO_SHAPE_MARKER
    return shapes, markers


def plot_embedding(coords, xlabel, ylabel, color_by=None, title="ESM embeddings", edges=None,
                   shape_by=None, shape_label="shape", color_label="colour", point_labels=None,
                   continuous=False):
    """Scatter of a 2D projection, optionally coloured by color_by and with
    the marker shape set by shape_by (one value per point; NaN = no
    annotation, drawn as an X). With shape_by, one legend outside the axes
    on the right lists the colours under color_label and the shapes under
    shape_label. point_labels (one string per point) are written next to
    the points in tiny type; save at a high dpi to read them.

    With continuous, color_by holds numbers, coloured on a viridis scale with
    a colour bar (labelled color_label) on the right; values that are not
    numbers are grey, like NaN. A shape legend then goes inside the axes."""
    fig, ax = plt.subplots(figsize=(8, 8))

    if edges is not None:
        segments = coords[edges]  # (E, 2, 2)
        ax.add_collection(LineCollection(segments, linewidths=0.3, colors="grey",
                                          alpha=0.3, zorder=0))

    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    markers = None
    if shape_by is not None:
        df["shape"] = pd.Series(np.asarray(shape_by, dtype=object)).fillna("nan").astype(str).values
        shapes, markers = shape_markers(shape_by, shape_label)
    style = dict(style="shape", markers=markers) if markers else {}

    cats, palette = [], {}
    if color_by is None:
        sns.scatterplot(data=df, x="x", y="y", s=4 if not markers else 20, alpha=0.45, linewidth=0,
                         color="steelblue", ax=ax, legend=False, **style)
    else:
        if continuous:
            df["value"] = pd.to_numeric(pd.Series(np.asarray(color_by, dtype=object)), errors="coerce").values
            missing = df["value"].isna()
        else:
            df["group"] = pd.Series(np.asarray(color_by)).fillna("nan").astype(str).values
            group_counts = df.loc[df["group"] != "nan", "group"].value_counts()
            if len(group_counts) > MAX_COLOURS:  # only the most common get a colour; the rest are grey
                df.loc[~df["group"].isin(group_counts.index[:MAX_COLOURS]), "group"] = "nan"
            missing = df["group"] == "nan"

        nan_rows = df[missing]
        if not nan_rows.empty:
            # Same size as the coloured points but more transparent, drawn underneath them
            sns.scatterplot(data=nan_rows, x="x", y="y", s=20, alpha=0.45,
                             color="lightgrey", linewidth=0, ax=ax, legend=False, **style)

        known = df[~missing]
        if continuous:
            if not known.empty:
                norm = Normalize(known["value"].min(), known["value"].max())
                sns.scatterplot(data=known, x="x", y="y", hue="value", hue_norm=norm, palette="viridis",
                                s=20, alpha=0.7, linewidth=0, ax=ax, legend=False, **style)
                colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax,
                                        fraction=0.04, pad=0.02)
                colorbar.set_label(color_label, fontsize=8)
                colorbar.ax.tick_params(labelsize=6)
        else:
            cats = sorted(known["group"].unique())
            cat_counts = known["group"].value_counts()
        if cats:
            palette = category_palette(cats)
            sns.scatterplot(data=known, x="x", y="y", hue="group", hue_order=cats,
                             palette=palette, s=20, alpha=0.7, linewidth=0, ax=ax,
                             legend=not markers, **style)
            if not markers:
                handles, labels = ax.get_legend_handles_labels()
                legend = ax.legend(handles, [f"{c} (n = {cat_counts[c]})" for c in labels], markerscale=1.2,
                                   fontsize=6, labelspacing=0.3, handletextpad=0.3, borderpad=0.4,
                                   frameon=True, framealpha=0.7, loc="best", title=None)
                for handle in legend.legend_handles:  # opaque swatches, easier to match to the points
                    handle.set_alpha(1)

    if markers:
        # One legend (a second one added with add_artist is left out of
        # savefig's bbox_inches="tight" and gets clipped); an invisible
        # handle with the column name heads each section.
        def header(text):
            return Line2D([], [], linestyle="none", marker="none", label=text)

        def point(marker, colour, text):
            return Line2D([], [], linestyle="", marker=marker, markersize=4, color=colour, label=text)

        handles = []
        if cats:
            handles += [header(color_label)] + [point("o", palette[c], f"{c} (n = {cat_counts[c]})")
                                                for c in cats] + [header("")]
        counts = df["shape"].value_counts()
        handles += [header(shape_label)] + [point(markers[s], "dimgrey", f"{s} (n = {counts[s]})")
                                            for s in shapes]
        if "nan" in counts:
            handles.append(point(NO_SHAPE_MARKER, "dimgrey", f"no {shape_label} (n = {counts['nan']})"))
        # Outside on the right, unless a colour bar is there already
        place = (dict(loc="best", frameon=True, framealpha=0.7) if continuous and color_by is not None
                 else dict(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False))
        ax.legend(handles=handles, fontsize=6, labelspacing=0.3, handletextpad=0.3, **place)

    if point_labels is not None:
        for (px, py), label in zip(coords, point_labels):
            ax.annotate(str(label), (px, py), xytext=(1.5, 0), textcoords="offset points", fontsize=2,
                        color="dimgrey", va="center")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")

    ids, embeddings = load_embeddings(args.embeddings)
    annot_df = load_annotations(args.annot_tsv)
    os.makedirs(args.output_dir, exist_ok=True)

    methods = [name for name in ("pca", "tsne", "umap", "tmap") if getattr(args, name)]

    timings = {}
    for method in methods:
        print(f"\nRunning {method.upper()}…")
        t0 = time.perf_counter()
        coords, edges = PROJECTORS[method](embeddings)
        elapsed = time.perf_counter() - t0
        timings[method] = elapsed
        print(f"  {method} done in {elapsed:.1f}s")

        npz_path = os.path.join(args.output_dir, f"{method}.npz")
        np.savez(npz_path, coords=coords, ids=ids)
        print(f"  Saved coords → {npz_path}")

        print(f"  Joining annotations for {method}…")
        meta = build_meta(ids, annot_df)

        colorings = [
            ("", None),
            ("_domain", meta["Domain"]),
            ("_ec", meta["EC"]),
            ("_subf", meta["Subf"]),
        ]
        for suffix, color_by in colorings:
            label = suffix.lstrip("_").replace("_", " ") if suffix else None
            title = f"CAZyDB {method.upper()}" + (f" — {label}" if label else "")
            fig = plot_embedding(coords, "Dim 1", "Dim 2", color_by=color_by,
                                  title=title, edges=edges)

            png_path = os.path.join(args.output_dir, f"{method}{suffix}.png")
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"  Saved plot → {png_path}")

            plt.show()
            plt.close(fig)

    print("\n--- Timings ---")
    for method, t in timings.items():
        print(f"  {method:<8} {t:.1f}s")


if __name__ == "__main__":
    main()
