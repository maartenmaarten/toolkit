import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import typer
import pandas as pd
import plotly.express as px
from matplotlib.colors import to_hex
from typing import Optional

from dimensio_reductio import MAX_COLOURS, PROJECTORS, category_palette, plot_embedding, shape_markers

METHOD_NAMES = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP", "tmap": "TMAP"}
# Below this many embeddings, an interactive plotly HTML is written next to each PNG
MAX_HTML_POINTS = 1000
# Below this many embeddings, each point is labelled with its id
MAX_LABEL_POINTS = 1000
# plotly symbol for each matplotlib marker in dimensio_reductio.SHAPE_MARKERS / NO_SHAPE_MARKER
PLOTLY_SYMBOLS = {"o": "circle", "s": "square", "D": "diamond", "^": "triangle-up", "v": "triangle-down",
                  "P": "cross", "<": "triangle-left", ">": "triangle-right", "p": "pentagon", "h": "hexagon",
                  "X": "x"}


def write_html(path, coords, ids, xlabel, ylabel, title, color_by=None, color_label="colour",
               shape_by=None, shape_label="shape", missing_label="no annotation", continuous=False):
    """Interactive plotly version of plot_embedding's figure: same colours
    and shapes, the id and annotations on hover. Grey points (NaN in
    color_by) are listed as missing_label. Each legend entry (a colour, or
    a colour-shape pair) ends in its point count. With continuous, color_by
    holds numbers, coloured on a viridis scale."""
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "id": ids})
    styling = {}
    if color_by is not None and continuous:
        df[color_label] = pd.to_numeric(color_by, errors="coerce").values
        styling.update(color=color_label, color_continuous_scale="viridis")
    elif color_by is not None:
        cats = sorted(color_by.dropna().astype(str).unique())
        missing = missing_label
        df[color_label] = color_by.astype(object).where(color_by.notna(), missing).astype(str).values
        colours = {c: to_hex(rgb) for c, rgb in category_palette(cats).items()}
        styling.update(color=color_label, color_discrete_map={**colours, missing: "#d3d3d3"},
                       category_orders={color_label: cats + [missing]})
    if shape_by is not None:
        shapes, markers = shape_markers(shape_by, shape_label)
        missing = f"no {shape_label}"
        df[shape_label] = shape_by.astype(object).where(shape_by.notna(), missing).astype(str).values
        symbols = {s: PLOTLY_SYMBOLS[markers[s]] for s in shapes}
        symbols[missing] = PLOTLY_SYMBOLS[markers["nan"]]
        styling.update(symbol=shape_label, symbol_map=symbols)
        styling.setdefault("category_orders", {})[shape_label] = shapes + [missing]
    common = dict(x="x", y="y", hover_name="id", hover_data={"x": False, "y": False},
                  labels={"x": xlabel, "y": ylabel}, title=title, template="plotly_white")
    if color_by is not None and continuous:
        # A colour scale has no colour for NaN: points without a value are grey traces of their
        # own, drawn first so they lie underneath
        missing = df[color_label].isna()
        fig = px.scatter(df[~missing], **common, **styling)
        no_colour = {key: value for key, value in styling.items() if not key.startswith("color")}
        grey = px.scatter(df[missing], **common, **no_colour, color_discrete_sequence=["#d3d3d3"])
        grey.for_each_trace(lambda trace: trace.update(
            name=f"{missing_label}, {trace.name}" if trace.name else missing_label, showlegend=True))
        fig.add_traces(grey.data)
        fig.data = fig.data[-len(grey.data):] + fig.data[:-len(grey.data)]
    else:
        fig = px.scatter(df, **common, **styling)
    fig.update_traces(marker=dict(size=8, opacity=0.6, line=dict(width=0.5, color="white")))
    if color_by is not None:  # annotated points more opaque than the grey ones
        fig.update_traces(marker_opacity=0.85, selector=lambda trace: not (
            isinstance(trace.marker.color, str) and trace.marker.color == "#d3d3d3"))
    fig.for_each_trace(lambda trace: trace.update(name=f"{trace.name} (n = {len(trace.x)})"))
    fig.write_html(path)


def pca_project(embeddings):
    pca = PCA(n_components=2, random_state=42)
    return {"coords": pca.fit_transform(embeddings), "explained_variance_ratio": pca.explained_variance_ratio_}


def load_or_project(name, cache_file, ids, embeddings_sha1, project, recompute=False):
    """Arrays of one projection (a dict with at least "coords"), loaded from
    cache_file (.npz) when it was made from the same embeddings: the same
    ids in the same order and the same sha1 of the embedding matrix.
    Otherwise project() is run and its arrays are saved to cache_file."""
    if cache_file.exists() and not recompute:
        with np.load(cache_file) as cache:
            if ("embeddings_sha1" in cache.files and str(cache["embeddings_sha1"]) == embeddings_sha1
                    and cache["ids"].tolist() == ids):
                print(f"Loaded {name} from {cache_file}")
                return {key: cache[key] for key in cache.files if key not in ("ids", "embeddings_sha1")}
        print(f"{cache_file} was made from other embeddings")
    print(f"Running {name}...")
    start = time.perf_counter()
    arrays = project()
    print(f"  {name} done in {time.perf_counter() - start:.1f}s")
    # Write to a temporary file first, so an interrupted run leaves no half-written cache
    tmp_file = cache_file.with_suffix(".tmp.npz")
    np.savez(tmp_file, ids=np.array(ids, dtype=str), embeddings_sha1=embeddings_sha1, **arrays)
    tmp_file.replace(cache_file)
    print(f"  Saved {name} coordinates to {cache_file}")
    return arrays


def main(
    embeddings_file: str,
    output_file: str,
    annotation_file: Optional[str] = typer.Option(None, "--annotation-file", help="Optional TSV file with annotations"),
    annotation_column: str = typer.Option("Domain", "--annotation-column", help="Column name in TSV to use for coloring"),
    top_annotations: int = typer.Option(MAX_COLOURS, "--top-annotations", min=1, max=MAX_COLOURS,
                                        help="Colour only the N most common annotations (tab20, so at "
                                             f"most {MAX_COLOURS}); the rest are grey"),
    continuous: bool = typer.Option(False, "--continuous",
                                    help="The annotation column holds numbers: colour on a viridis "
                                         "scale with a colour bar instead of by category"),
    shape_column: Optional[str] = typer.Option(None, "--shape-column",
                                               help="Optional column in the annotation TSV to use for marker shape"),
    point_labels: bool = typer.Option(True, "--point-labels/--no-point-labels",
                                      help="Write every embedding's id next to its point, in tiny type "
                                           f"(skipped with {MAX_LABEL_POINTS} or more embeddings)"),
    recompute: bool = typer.Option(False, "--recompute",
                                   help="Recompute PCA, t-SNE and UMAP even if cached coordinates exist"),
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
            plot_tsne.png, plot_umap.png. With fewer than 1000 embeddings, an
            interactive plotly <name>_<method>.html is written as well, with
            the same colours and shapes and the id on hover. The 2D
            coordinates are cached next to it as <embeddings name>_<method>.npz
            (e.g. GH43_pca.npz) and reused on later runs on the same
            embeddings, so changing only the annotations just replots
        annotation_file: Optional TSV file with annotations (must have ID column matching embedding IDs)
        annotation_column: Column name in TSV to use for coloring
        top_annotations: Colour only this many (at most 20, the colours of
            tab20) of the most common values of annotation_column;
            embeddings with any other value are plotted in grey, like
            unannotated ones. Not used with --continuous
        continuous: Treat annotation_column as numbers (e.g. the
            nn_cosine_similarity of plm_knn_top-k.py's TSV) and colour them
            on a viridis scale, with a colour bar; values that are not
            numbers are grey, like unannotated points
        shape_column: Optional column name in the same TSV to use for marker
            shape (at most 10 distinct values; unannotated points get an X)
        point_labels: Write each embedding's id next to its point in tiny
            type (PNGs are saved at 400 dpi so they can be read zoomed in);
            skipped with 1000 or more embeddings
        recompute: Ignore the cached .npz coordinates and recompute them,
            e.g. after changing the t-SNE or UMAP settings in
            dimensio_reductio.py (the cache only checks the embeddings)
    """
    if shape_column and not annotation_file:
        raise typer.BadParameter("--shape-column needs --annotation-file")

    # Load embeddings
    embeddings_dict = torch.load(embeddings_file)
    print(f"Loaded {len(embeddings_dict)} embeddings")

    # Convert to numpy array
    ids = list(embeddings_dict.keys())
    embeddings = torch.stack([embeddings_dict[id] for id in ids]).numpy()
    if point_labels and len(ids) >= MAX_LABEL_POINTS:
        print(f"{len(ids)} embeddings: not labelling points (only below {MAX_LABEL_POINTS})")
        point_labels = False

    print(f"Embedding shape: {embeddings.shape}")

    # PCA, t-SNE and UMAP: loaded from <embeddings name>_<method>.npz next to the
    # output file if those hold these embeddings' projections, else computed and saved there
    output = Path(output_file)
    embeddings_sha1 = hashlib.sha1(embeddings.tobytes()).hexdigest()
    projectors = {"pca": lambda: pca_project(embeddings),
                  "tsne": lambda: {"coords": PROJECTORS["tsne"](embeddings)[0]},
                  "umap": lambda: {"coords": PROJECTORS["umap"](embeddings)[0]}}
    projections = {}
    for method, project in projectors.items():
        cache_file = output.with_name(f"{Path(embeddings_file).stem}_{method}.npz")
        projections[method] = load_or_project(METHOD_NAMES[method], cache_file, ids, embeddings_sha1,
                                              project, recompute)
    pca_ratio = projections["pca"]["explained_variance_ratio"]
    print(f"Explained variance ratio: {pca_ratio}")

    # Load annotations if provided
    color_by = None
    shape_by = None
    missing_label = "no annotation"
    if annotation_file:
        print(f"Loading annotations from {annotation_file}...")
        ann_df = pd.read_csv(annotation_file, sep="\t")
        # Use GenBank column as ID column; the first row counts for repeated IDs
        id_col = "GenBank"
        missing = [c for c in (annotation_column, shape_column) if c and c not in ann_df.columns]
        if missing:
            raise typer.BadParameter(f"column(s) {missing} not in {annotation_file}; "
                                     f"available: {list(ann_df.columns)}")
        annotations = ann_df.drop_duplicates(id_col).set_index(id_col)
        color_by = pd.Series(ids).map(annotations[annotation_column])  # NaN where an ID has no annotation

        if continuous:
            numbers = pd.to_numeric(color_by, errors="coerce")
            not_numbers = color_by.notna() & numbers.isna()
            color_by = numbers
            if color_by.notna().sum() == 0:
                raise typer.BadParameter(f"--continuous: {annotation_column} holds no numbers for these "
                                         f"embeddings")
            print(f"{annotation_column}: {color_by.notna().sum()} values from {color_by.min():.4g} "
                  f"to {color_by.max():.4g}"
                  + (f"; {not_numbers.sum()} that are not numbers count as missing" if not_numbers.any()
                     else ""))
            print(f"{color_by.isna().sum()} embedding(s) without a value (plotted in grey)")
        else:
            unique_colors = sorted(color_by.dropna().astype(str).unique())
            print(f"Found {len(unique_colors)} unique annotations: {unique_colors}")
            print(f"{color_by.isna().sum()} embedding(s) without annotation (plotted in grey)")
        if not continuous and len(unique_colors) > top_annotations:
            top = color_by.value_counts().index[:top_annotations]
            other = color_by.notna() & ~color_by.isin(top)
            color_by = color_by.where(~other)
            missing_label = "other / no annotation"
            print(f"Colouring only the {top_annotations} most common annotations; "
                  f"{other.sum()} embedding(s) with another annotation also plotted in grey")

        if shape_column:
            shape_by = pd.Series(ids).map(annotations[shape_column])
            unique_shapes = sorted(shape_by.dropna().astype(str).unique())
            print(f"Found {len(unique_shapes)} unique {shape_column} values for shape: {unique_shapes}")
            print(f"{shape_by.isna().sum()} embedding(s) without {shape_column} (plotted as X)")

    # Plot: one figure per method
    sns.set_theme(style="whitegrid")
    for method, projection in projections.items():
        name = METHOD_NAMES[method]
        coords = projection["coords"]
        if method == "pca":
            xlabel = f"PC1 ({pca_ratio[0]:.1%})"
            ylabel = f"PC2 ({pca_ratio[1]:.1%})"
        else:
            xlabel, ylabel = f"{name} 1", f"{name} 2"
        title = f"{name} visualization ({len(embeddings)} embeddings)"
        if color_by is not None:
            title += f", colored by {annotation_column}"
            if missing_label != "no annotation":
                title += f" (top {top_annotations})"
        if shape_by is not None:
            title += f", shaped by {shape_column}"
        fig = plot_embedding(coords, xlabel, ylabel, color_by=color_by, title=title,
                             shape_by=shape_by, shape_label=shape_column or "shape",
                             color_label=annotation_column, point_labels=ids if point_labels else None,
                             continuous=continuous)

        plot_file = output.with_name(f"{output.stem}_{method}{output.suffix or '.png'}")
        fig.savefig(plot_file, dpi=400 if point_labels else 150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {plot_file}")

        if len(ids) < MAX_HTML_POINTS:
            html_file = output.with_name(f"{output.stem}_{method}.html")
            write_html(html_file, coords, ids, xlabel, ylabel, title, color_by=color_by,
                       color_label=annotation_column, shape_by=shape_by, shape_label=shape_column or "shape",
                       missing_label=missing_label, continuous=continuous)
            print(f"Saved interactive plot to {html_file}")


if __name__ == "__main__":
    typer.run(main)
