#!/usr/bin/env python3
"""
Convert a DeepTMHMM-style .3line topology file to a greyscale iTOL
DATASET_COLORSTRIP ring, with the topology class (SP, GLOB, TM, ...) printed
inside each strip.

Input (.3line): three lines per protein
    >SEQ_ID | CLASS
    SEQUENCE
    TOPOLOGY_STRING

Classes are drawn in white-to-black shades with a black border, so the ring
stays readable next to coloured functional annotation rings. iTOL colour
strips don't support hatch/pattern fills, so shade + text label is what
distinguishes the classes.

SP/TM boxes on the proteins themselves are drawn by itol_from_domain.py
(--topology), which reuses parse_3line from this script.

Usage:
    python itol_from_topology.py --topology predicted_topologies.3line \
        --out itol_topology.txt
"""

import argparse
import sys
from pathlib import Path

# Known classes get a fixed shade; unknown classes take the next free one.
CLASS_SHADES = {
    "GLOB": "#FFFFFF",
    "SP": "#D9D9D9",
    "TM": "#969696",
    "SP+TM": "#525252",
}
SPARE_SHADES = ["#BDBDBD", "#737373", "#252525", "#000000"]


def parse_3line(path: str) -> dict:
    """Return {seq_id: (class, topology_string)}."""
    records = {}
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    for i, line in enumerate(lines):
        if not line.startswith(">"):
            continue
        header = line[1:].strip()
        if "|" not in header:
            sys.exit(f"ERROR: header without '| CLASS' field: {line}")
        seq_id, topo_class = (s.strip() for s in header.split("|", 1))
        if i + 2 >= len(lines) or lines[i + 2].startswith(">"):
            sys.exit(f"ERROR: missing sequence/topology lines for {seq_id}")
        records[seq_id] = (topo_class, lines[i + 2])
    return records


def assign_shades(classes: set) -> dict:
    shades = {c: CLASS_SHADES[c] for c in classes if c in CLASS_SHADES}
    spare = iter(SPARE_SHADES)
    for c in sorted(classes - shades.keys()):
        try:
            shades[c] = next(spare)
        except StopIteration:
            sys.exit(f"ERROR: more topology classes than available shades: {sorted(classes)}")
    return shades


def write_colorstrip(out_path: str, dataset_label: str, id_to_class: dict,
                     shades: dict, show_labels: bool):
    order = [c for c in CLASS_SHADES if c in shades] + sorted(set(shades) - CLASS_SHADES.keys())

    lines = [
        "DATASET_COLORSTRIP",
        "SEPARATOR TAB",
        f"DATASET_LABEL\t{dataset_label}",
        "COLOR\t#000000",
        "",
        # black border so the white class stays visible against the background
        "BORDER_WIDTH\t1",
        "BORDER_COLOR\t#000000",
        "COMPLETE_BORDER\t1",
        "STRIP_WIDTH\t40",
        "",
        f"LEGEND_TITLE\t{dataset_label}",
        "LEGEND_SHAPES\t" + "\t".join("1" for _ in order),
        "LEGEND_COLORS\t" + "\t".join(shades[c] for c in order),
        "LEGEND_LABELS\t" + "\t".join(order),
        "",
    ]
    if show_labels:
        lines += [
            "SHOW_STRIP_LABELS\t1",
            "STRIP_LABEL_POSITION\tcenter",
            "STRIP_LABEL_SIZE_FACTOR\t0.8",
            "STRIP_LABEL_COLOR\t#000000",
            # white halo keeps black text readable on the dark shades
            "STRIP_LABEL_OUTLINE\t1",
            "",
        ]
    lines.append("DATA")
    for seq_id, topo_class in sorted(id_to_class.items()):
        lines.append(f"{seq_id}\t{shades[topo_class]}\t{topo_class}")

    Path(out_path).write_text("\n".join(lines) + "\n")
    counts = {c: sum(1 for v in id_to_class.values() if v == c) for c in order}
    print(f"Wrote {out_path}: {len(id_to_class)} sequences")
    for c in order:
        print(f"  {c:<6} {shades[c]}  {counts[c]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".3line topology → greyscale iTOL DATASET_COLORSTRIP")
    parser.add_argument("--topology", required=True, help="Input .3line topology file")
    parser.add_argument("--out", required=True, help="Output iTOL colour strip file")
    parser.add_argument("--label", default="Topology", help="Dataset/legend label (default: Topology)")
    parser.add_argument("--no-text", action="store_true", help="Don't print the class text in the strip")
    args = parser.parse_args()

    records = parse_3line(args.topology)
    if not records:
        sys.exit(f"ERROR: no '>' headers found in {args.topology}")
    id_to_class = {seq_id: topo_class for seq_id, (topo_class, _) in records.items()}
    shades = assign_shades(set(id_to_class.values()))
    write_colorstrip(args.out, args.label, id_to_class, shades, show_labels=not args.no_text)

