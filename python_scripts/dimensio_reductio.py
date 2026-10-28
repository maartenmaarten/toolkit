"""Project CAZyDB embeddings (PCA / t-SNE / UMAP / TMAP) and plot the results.

For each requested method: compute the 2D projection (or load it from an
earlier run's saved files), join the ids against the CAZy_characterized
annotation table, then render and save one scatter plot per coloring (none,
plus one per --color-by column; default Domain, EC, Subf), optionally with
marker shapes from a --shape-by column -- popping each one up on screen as
it's made.

Output files share the --output prefix, by default projections/<embeddings
file name without .pt>:
    PREFIX.<method>.npz                           ids + coordinates (+ TMAP tree edges)
    PREFIX.tmap.model                             full fitted TMAP (TMAP.save / TMAP.load)
    PREFIX.<method>[.<color column>][.shape_<shape column>].png
"""
import argparse
import ast
import os
import time
from pathlib import Path

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
DEFAULT_ID_COLUMN = "GenBank"
DEFAULT_COLOR_BY = ["Domain", "EC", "Subf"]

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
                         help="CAZy_characterized TSV with the --id-column plus the "
                              "--color-by / --shape-by columns (default: %(default)s)")
    parser.add_argument("--id-column", default=DEFAULT_ID_COLUMN,
                         help="Annotation column matched against the embedding ids "
                              "(default: %(default)s)")
    parser.add_argument("--color-by", nargs="+", default=DEFAULT_COLOR_BY, metavar="COLUMN",
                         help="Annotation column(s) to color by, one plot each, next to the "
                              "uncolored one (default: %(default)s)")
    parser.add_argument("--shape-by", default=None, metavar="COLUMN",
                         help=f"Annotation column setting the marker shape in every plot; at "
                              f"most {len(SHAPE_MARKERS)} distinct values (default: none)")
    parser.add_argument("--output", default=None, metavar="PREFIX",
                         help="Prefix (may include a directory) for the PREFIX.<method>.npz "
                              "caches and PREFIX.<method>[.<coloring>].png plots (default: "
                              f"{DEFAULT_OUTPUT_DIR}/<embeddings file name without .pt>)")
    parser.add_argument("--recompute", action="store_true",
                         help="Recompute projections even if a matching PREFIX.<method>.npz "
                              "cache exists")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, Path(args.embeddings).stem)

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
    return TMAP(metric=metric, n_neighbors=n_neighbors).fit(standardize(embeddings))


# Each returns (coords, tree edges or None, fitted model to save or None)
PROJECTORS = {
    "pca": lambda emb: (pca_project(emb), None, None),
    "tsne": lambda emb: (tsne_project(emb), None, None),
    "umap": lambda emb: (umap_project(emb), None, None),
    "tmap": lambda emb: (lambda m: (m.embedding_, m.tree_.edges, m))(tmap_project(emb)),
}


def cache_paths(prefix, method):
    """(.npz with ids + coords [+ edges], full model file or None). TMAP's model
    is written with its own TMAP.save() (map, tree with weights, kNN graph,
    parameters) and read back with TMAP.load()."""
    return f"{prefix}.{method}.npz", (f"{prefix}.{method}.model" if method == "tmap" else None)


def save_projection(prefix, method, ids, coords, edges, model):
    npz_path, model_path = cache_paths(prefix, method)
    extra = {} if edges is None else {"edges": np.asarray(edges)}
    np.savez(npz_path, coords=coords, ids=ids, **extra)
    print(f"  Saved coords → {npz_path}")
    if model_path:
        print(f"  Saved {method.upper()} model → {model.save(model_path)}")


def load_projection(prefix, method, ids):
    """(coords, edges) from an earlier run's cache, or None when there is none
    or it no longer fits: different ids (e.g. new embeddings under the same
    name) or, for TMAP, no saved model. TMAP coords and edges come from the
    saved model."""
    npz_path, model_path = cache_paths(prefix, method)
    if not os.path.exists(npz_path):
        return None
    cache = np.load(npz_path, allow_pickle=False)
    if not np.array_equal(cache["ids"], ids):
        print(f"  Ignoring {npz_path}: its ids differ from the current embeddings")
        return None
    if model_path:
        if not os.path.exists(model_path):
            print(f"  Ignoring {npz_path}: no saved {method.upper()} model ({model_path})")
            return None
        model = TMAP.load(model_path)
        print(f"  Loaded {method.upper()} model ← {model_path}")
        return model.embedding_, model.tree_.edges
    return cache["coords"], None


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


def load_annotations(annot_tsv: str, id_column: str, columns: list[str]) -> pd.DataFrame:
    available = pd.read_csv(annot_tsv, sep="\t", nrows=0).columns
    missing = [c for c in [id_column, *columns] if c not in available]
    if missing:
        raise SystemExit(f"Column(s) {missing} not in {annot_tsv}; available: {list(available)}")

    # Ids as strings, so numeric ids still match the embedding keys
    annot_df = pd.read_csv(annot_tsv, sep="\t", usecols=list(dict.fromkeys([id_column, *columns])),
                           dtype={id_column: str})
    if "EC" in columns:
        annot_df["EC"] = annot_df["EC"].map(_clean_ec)
    # Whole-number float columns (e.g. Subf: 1.0, 4.0) are labels; render them
    # as clean ints, keeping NaN as NaN.
    for col in columns:
        values = annot_df[col].dropna()
        if annot_df[col].dtype.kind == "f" and (values == values.round()).all():
            annot_df[col] = annot_df[col].apply(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    return annot_df.drop_duplicates(id_column)


def build_meta(ids: np.ndarray, annot_df: pd.DataFrame, id_column: str,
               columns: list[str]) -> pd.DataFrame:
    m = pd.DataFrame({"id": ids}).merge(annot_df, left_on="id", right_on=id_column, how="left")
    width = max(len(col) for col in columns)
    for col in columns:
        print(f"  {col + ' match rate:':<{width + 12}} {m[col].notna().mean():.1%}")
    return m


def file_tag(column: str) -> str:
    """'Protein Name' -> 'protein_name', for plot file names."""
    return "_".join(column.lower().split())


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
    columns = list(dict.fromkeys(args.color_by + ([args.shape_by] if args.shape_by else [])))
    annot_df = load_annotations(args.annot_tsv, args.id_column, columns)
    print(f"Joining annotations on {args.id_column}…")
    meta = build_meta(ids, annot_df, args.id_column, columns)
    shape_by = meta[args.shape_by] if args.shape_by else None
    if args.shape_by:
        # Fail on too many shape values now rather than after the projections
        try:
            shape_markers(shape_by, args.shape_by)
        except ValueError as e:
            raise SystemExit(f"--shape-by: {e}")
    # Shaped plots get their own file names, so they don't overwrite unshaped ones
    shape_tag = f".shape_{file_tag(args.shape_by)}" if args.shape_by else ""
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    methods = [name for name in ("pca", "tsne", "umap", "tmap") if getattr(args, name)]

    timings = {}
    for method in methods:
        print(f"\n{method.upper()}")
        cached = None if args.recompute else load_projection(args.output, method, ids)
        if cached is not None:
            coords, edges = cached
            print("  Reusing the saved projection (--recompute to rerun)")
        else:
            print(f"  Running {method.upper()}…")
            t0 = time.perf_counter()
            coords, edges, model = PROJECTORS[method](embeddings)
            elapsed = time.perf_counter() - t0
            timings[method] = elapsed
            print(f"  {method} done in {elapsed:.1f}s")
            save_projection(args.output, method, ids, coords, edges, model)

        for col in [None, *args.color_by]:
            label = file_tag(col).replace("_", " ") if col else None
            title = f"CAZyDB {method.upper()}" + (f" — {label}" if label else "")
            fig = plot_embedding(coords, "Dim 1", "Dim 2",
                                  color_by=meta[col] if col else None, color_label=col or "colour",
                                  shape_by=shape_by, shape_label=args.shape_by or "shape",
                                  title=title, edges=edges)

            color_tag = f".{file_tag(col)}" if col else ""
            png_path = f"{args.output}.{method}{color_tag}{shape_tag}.png"
            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            print(f"  Saved plot → {png_path}")

            plt.show()
            plt.close(fig)

    if timings:
        print("\n--- Timings ---")
        for method, t in timings.items():
            print(f"  {method:<8} {t:.1f}s")


if __name__ == "__main__":
    main()
