#!/usr/bin/env python3
"""
Merge domain annotations from several tools into one iTOL DATASET_DOMAINS file.

Sources (all optional, at least one required), in priority order:
  1. dbCAN        — HMMER domtblout (hmmscan or hmmsearch vs dbCAN-HMMdb), or
                    run_dbcan's own tabular output with a header row
                    (hmmer.out / dbCAN_hmm_results.tsv)
  2. Pfam         — HMMER domtblout (hmmscan or hmmsearch vs Pfam-A)
  3. InterProScan — TSV output (-f tsv, no header row)
  4. SignalP      — SignalP 5/6 prediction summary (prediction_results.txt,
                    *_summary.signalp5) or its GFF3 output

Protein backbones come from the full-length FASTA: every protein in it gets a
row with its true length, also when no tool annotated it. Tool coordinates are
used as-is, so the FASTA must be the one the tools were run on (not extracted
domain sequences). Lengths reported by the tools (domtblout qlen/tlen,
InterProScan column 3) are checked against the FASTA and mismatches reported.

Overlaps: within a source, boxes are ranked by i-Evalue (InterProScan: entries
with an InterPro accession first, then longest); across sources, by the order
above. A box is dropped when it overlaps an already placed box by more than
--max-overlap of the shorter of the two. --keep-overlaps keeps everything.

Each source gets its own box shape; colors are per domain label.

Usage:
    python itol_from_domain_sources.py --fasta proteins.faa --out itol_domains.txt \\
        --dbcan dbcan.domtblout --pfam pfam.domtblout \\
        --interproscan proteins.faa.tsv --signalp prediction_results.txt
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
import seaborn as sns

from itol_from_domain import (
    ID_ALIASES, DOMAIN_ALIASES, START_ALIASES, END_ALIASES,
    _match_col, _is_domtblout, read_fasta_lengths,
)


# source name -> box shape, in priority order
SOURCES = {
    "dbCAN":        "RE",
    "Pfam":         "HH",
    "InterProScan": "EL",
    "SignalP":      "RE",
}

PALETTE = (
    sns.color_palette("tab20").as_hex()
    + sns.color_palette("tab20b").as_hex()
    + sns.color_palette("tab20c").as_hex()
)

# SignalP prediction type -> fixed box color
SP_COLORS = {
    "SP":      "#FF00E1",
    "LIPO":    "#9400D3",
    "TAT":     "#FF8C00",
    "TATLIPO": "#8B4513",
    "PILIN":   "#000000",
}

# extra aliases for run_dbcan's tabular outputs
DBCAN_ID_ALIASES     = ID_ALIASES | {"gene id", "target name"}
DBCAN_DOMAIN_ALIASES = DOMAIN_ALIASES | {"hmm profile", "hmm name"}
DBCAN_START_ALIASES  = START_ALIASES | {"gene start", "target from"}
DBCAN_END_ALIASES    = END_ALIASES | {"gene end", "target to"}
DBCAN_EVALUE_ALIASES = {"e value", "evalue", "i-evalue", "i_evalue"}
DBCAN_COV_ALIASES    = {"coverage"}

_CS_POS = re.compile(r"CS pos:\s*(\d+)-(\d+)")


# ── helpers ────────────────────────────────────────────────────────────────────

def _first_word(s: str) -> str:
    return s.strip().split()[0] if s.strip() else ""


def _clean_label(s: str) -> str:
    # '|' separates the fields of a domain entry in iTOL
    return s.strip().replace("|", "/").replace("\t", " ")


def _box(seq_id, label, start, end, source, rank, seq_len=None) -> dict:
    return {
        "seq_id": seq_id, "label": _clean_label(label),
        "start": int(start), "end": int(end),
        "source": source, "rank": rank, "seq_len": seq_len,
    }


def assign_colors(labels: list) -> dict:
    unique = sorted(set(labels))
    palette = PALETTE if len(unique) <= len(PALETTE) else sns.color_palette("husl", len(unique)).as_hex()
    return {d: palette[i % len(palette)] for i, d in enumerate(unique)}


# ── parsers: HMMER (dbCAN, Pfam) ───────────────────────────────────────────────

def parse_domtblout(path: str, source: str, fasta_ids: set,
                    evalue_threshold: float, min_coverage: float = 0.0) -> list:
    """Domain rows from HMMER domtblout, filtered on i-Evalue and HMM coverage."""
    records = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(None, 22)
            if len(parts) >= 22:
                records.append(parts)

    # hmmscan: query = protein, target = HMM; hmmsearch: the other way round
    query_hits  = sum(p[3] in fasta_ids for p in records)
    target_hits = sum(p[0] in fasta_ids for p in records)
    seq_i, hmm_i = (0, 3) if target_hits > query_hits else (3, 0)

    rows = []
    for p in records:
        try:
            i_evalue = float(p[12])
            hmm_from, hmm_to = int(p[15]), int(p[16])
            env_from, env_to = int(p[19]), int(p[20])
            hmm_len, seq_len = int(p[hmm_i + 2]), int(p[seq_i + 2])
        except ValueError:
            continue
        if i_evalue > evalue_threshold:
            continue
        # dbCAN's coverage definition
        if (hmm_to - hmm_from) / hmm_len < min_coverage:
            continue
        rows.append(_box(
            p[seq_i], p[hmm_i].removesuffix(".hmm"), env_from, env_to,
            source, (i_evalue, env_from - env_to), seq_len,
        ))
    return rows


def parse_dbcan_table(path: str, evalue_threshold: float, min_coverage: float) -> list:
    """run_dbcan's own tabular HMMER output (header row, tab-separated)."""
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        headers = reader.fieldnames or []
        cols = {
            "sequence-ID": _match_col(headers, DBCAN_ID_ALIASES),
            "domain-name": _match_col(headers, DBCAN_DOMAIN_ALIASES),
            "start":       _match_col(headers, DBCAN_START_ALIASES),
            "end":         _match_col(headers, DBCAN_END_ALIASES),
        }
        missing = [name for name, col in cols.items() if col is None]
        if missing:
            sys.exit(
                f"ERROR: dbCAN file {path} is not domtblout and could not auto-detect "
                f"column(s) for: {', '.join(missing)}\nAvailable columns: {headers}"
            )
        col_evalue = _match_col(headers, DBCAN_EVALUE_ALIASES)
        col_cov    = _match_col(headers, DBCAN_COV_ALIASES)

        for row in reader:
            evalue = float(row[col_evalue]) if col_evalue else 0.0
            if evalue > evalue_threshold:
                continue
            if col_cov and float(row[col_cov]) < min_coverage:
                continue
            start, end = int(row[cols["start"]]), int(row[cols["end"]])
            rows.append(_box(
                _first_word(row[cols["sequence-ID"]]),
                row[cols["domain-name"]].strip().removesuffix(".hmm"),
                start, end, "dbCAN", (evalue, start - end),
            ))
    return rows


# ── parser: InterProScan ───────────────────────────────────────────────────────

def parse_interproscan(path: str, label_mode: str,
                       include: set, exclude: set) -> list:
    """
    InterProScan TSV columns (0-based): 0 protein, 1 md5, 2 length, 3 analysis,
    4 signature acc, 5 signature desc, 6 start, 7 stop, 8 score, 9 status,
    10 date, 11 InterPro acc, 12 InterPro desc, 13+ GO / pathways.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            analysis = f[3].strip()
            if include and analysis.lower() not in include:
                continue
            if analysis.lower() in exclude:
                continue

            def field(i):
                v = f[i].strip() if len(f) > i else ""
                return "" if v == "-" else v

            sig_acc, sig_desc = field(4), field(5)
            ipr_acc, ipr_desc = field(11), field(12)
            if label_mode == "interpro":
                label = ipr_desc or sig_desc or sig_acc
            elif label_mode == "description":
                label = sig_desc or sig_acc
            else:
                label = sig_acc

            start, end = int(f[6]), int(f[7])
            rows.append(_box(
                _first_word(f[0]), label, start, end, "InterProScan",
                (0 if ipr_acc else 1, start - end), int(f[2]),
            ))
    return rows


# ── parser: SignalP ────────────────────────────────────────────────────────────

def _sp_type_from_gff(feature: str) -> str:
    feature = feature.lower()
    if "tat" in feature and "lipo" in feature:
        return "TATLIPO"
    for key in ("lipo", "tat", "pilin"):
        if key in feature:
            return key.upper()
    return "SP"


def _is_gff(path: str) -> bool:
    with open(path) as fh:
        for line in fh:
            if line.startswith("##gff"):
                return True
            if line.startswith("#") or not line.strip():
                continue
            # the SignalP 6 summary also has 9 columns, so check the content too
            f = line.rstrip("\n").split("\t")
            return len(f) == 9 and f[3].isdigit() and f[4].isdigit()
    return False


def parse_signalp(path: str) -> list:
    """Signal peptides (1-based inclusive) from SignalP 5/6 summary or GFF3."""
    rows = []
    if _is_gff(path):
        # region-mode GFF3 (--mode region) splits each SP into n/h/c-regions
        # plus markers; merge them into one box per protein
        regions = defaultdict(list)
        with open(path) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 9:
                    continue
                seq_id, feature = _first_word(f[0]), f[2].lower()
                if "signal" in feature:
                    rows.append(_box(
                        seq_id, _sp_type_from_gff(feature), f[3], f[4], "SignalP", (0,),
                    ))
                else:
                    regions[seq_id].append((feature, int(f[3]), int(f[4])))

        for seq_id, feats in regions.items():
            names = {name for name, _, _ in feats}
            # the lipid-modified cysteine is the first residue after cleavage
            span = [(s, e) for name, s, e in feats if name.endswith("-region")]
            if not span:
                continue
            lipo = any("lipid" in name for name in names)
            tat = any("twin-arginine" in name for name in names)
            sp_type = "TATLIPO" if lipo and tat else "LIPO" if lipo else "TAT" if tat else "SP"
            rows.append(_box(
                seq_id, sp_type, min(s for s, _ in span), max(e for _, e in span),
                "SignalP", (0,),
            ))
        return rows

    # summary: "# ID  Prediction  <probabilities ...>  CS Position"
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            # SignalP 5 writes e.g. "SP(Sec/SPI)", SignalP 6 just "SP"
            sp_type = f[1].split("(")[0].strip().upper()
            if sp_type in ("OTHER", "NO_SP", ""):
                continue
            cs = _CS_POS.search(line)
            if not cs:
                continue
            rows.append(_box(
                _first_word(f[0]), sp_type, 1, cs.group(1), "SignalP", (0,),
            ))
    return rows


# ── merging ────────────────────────────────────────────────────────────────────

def _overlap_frac(a: dict, b: dict) -> float:
    overlap = min(a["end"], b["end"]) - max(a["start"], b["start"]) + 1
    if overlap <= 0:
        return 0.0
    shorter = min(a["end"] - a["start"], b["end"] - b["start"]) + 1
    return overlap / shorter


def resolve_overlaps(boxes: list, max_overlap: float) -> list:
    """Place boxes by source priority, then rank; drop those overlapping placed ones."""
    priority = {name: i for i, name in enumerate(SOURCES)}
    kept = []
    for b in sorted(boxes, key=lambda b: (priority[b["source"]], b["rank"])):
        if all(_overlap_frac(b, k) <= max_overlap for k in kept):
            kept.append(b)
    return kept


def _warn_ids(msg: str, ids: set):
    if ids:
        print(
            f"  WARNING: {len(ids)} {msg}: {', '.join(sorted(ids)[:5])}"
            + (" ..." if len(ids) > 5 else "")
        )


# ── main conversion ────────────────────────────────────────────────────────────

def convert_sources(
    fasta_path: str,
    out_path: str,
    dbcan_path: Optional[str] = None,
    pfam_path: Optional[str] = None,
    ips_path: Optional[str] = None,
    signalp_path: Optional[str] = None,
    dbcan_evalue: float = 1e-15,
    dbcan_coverage: float = 0.35,
    pfam_evalue: float = 1e-5,
    ips_label: str = "interpro",
    ips_include: Optional[set] = None,
    ips_exclude: Optional[set] = None,
    max_overlap: float = 0.5,
    keep_overlaps: bool = False,
    show_labels: bool = False,
    dataset_label: str = "Domains",
):
    if not any([dbcan_path, pfam_path, ips_path, signalp_path]):
        sys.exit("ERROR: give at least one of --dbcan, --pfam, --interproscan, --signalp")

    lengths = read_fasta_lengths(fasta_path)
    if not lengths:
        sys.exit(f"ERROR: no sequences found in {fasta_path}")
    print(f"  Read {len(lengths)} proteins from {fasta_path}")
    fasta_ids = set(lengths)

    boxes = []
    if dbcan_path:
        if _is_domtblout(dbcan_path):
            rows = parse_domtblout(dbcan_path, "dbCAN", fasta_ids, dbcan_evalue, dbcan_coverage)
        else:
            rows = parse_dbcan_table(dbcan_path, dbcan_evalue, dbcan_coverage)
        print(f"  dbCAN:        {len(rows)} domains (i-Evalue <= {dbcan_evalue}, coverage >= {dbcan_coverage})")
        boxes += rows
    if pfam_path:
        rows = parse_domtblout(pfam_path, "Pfam", fasta_ids, pfam_evalue)
        print(f"  Pfam:         {len(rows)} domains (i-Evalue <= {pfam_evalue})")
        boxes += rows
    if ips_path:
        rows = parse_interproscan(ips_path, ips_label, ips_include or set(), ips_exclude or set())
        print(f"  InterProScan: {len(rows)} matches")
        boxes += rows
    if signalp_path:
        rows = parse_signalp(signalp_path)
        print(f"  SignalP:      {len(rows)} signal peptides")
        boxes += rows

    if not boxes:
        sys.exit("ERROR: no domain entries found after parsing/filtering")

    _warn_ids("sequence ID(s) not found in FASTA (skipped)",
              {b["seq_id"] for b in boxes} - fasta_ids)
    boxes = [b for b in boxes if b["seq_id"] in fasta_ids]

    # catch a FASTA that is not the one the tools were run on
    _warn_ids("protein(s) with a length in the annotations that differs from the FASTA",
              {b["seq_id"] for b in boxes if b["seq_len"] and b["seq_len"] != lengths[b["seq_id"]]})
    too_long = {b["seq_id"] for b in boxes if b["end"] > lengths[b["seq_id"]]}
    _warn_ids("protein(s) with boxes past the FASTA sequence end (clipped or dropped)", too_long)
    boxes = [b for b in boxes if b["start"] <= lengths[b["seq_id"]]]
    for b in boxes:
        b["end"] = min(b["end"], lengths[b["seq_id"]])

    by_seq = defaultdict(list)
    for b in boxes:
        by_seq[b["seq_id"]].append(b)
    if not keep_overlaps:
        n_before = len(boxes)
        by_seq = {s: resolve_overlaps(bs, max_overlap) for s, bs in by_seq.items()}
        n_after = sum(len(bs) for bs in by_seq.values())
        print(f"  Dropped {n_before - n_after} overlapping boxes (max overlap {max_overlap})")
    kept = [b for bs in by_seq.values() for b in bs]

    colors = assign_colors([b["label"] for b in kept if b["source"] != "SignalP"])

    def box_color(b):
        return SP_COLORS.get(b["label"], "#808080") if b["source"] == "SignalP" else colors[b["label"]]

    lines: list[str] = []
    lines.append("DATASET_DOMAINS")
    lines.append("SEPARATOR TAB")
    lines.append(f"DATASET_LABEL\t{dataset_label}")
    lines.append("COLOR\t#000000")
    lines.append("")
    lines.append(f"LEGEND_TITLE\t{dataset_label}")
    legend = []
    for source, shape in SOURCES.items():
        for b in sorted({b["label"]: b for b in kept if b["source"] == source}.values(),
                        key=lambda b: b["label"]):
            legend.append((shape, box_color(b), f"{b['label']} ({source})"))
    lines.append("LEGEND_SHAPES\t" + "\t".join(s for s, _, _ in legend))
    lines.append("LEGEND_COLORS\t" + "\t".join(c for _, c, _ in legend))
    lines.append("LEGEND_LABELS\t" + "\t".join(l for _, _, l in legend))
    lines.append("")
    if show_labels:
        lines.append("SHOW_DOMAIN_LABELS\t1")
        lines.append("LABELS_ON_TOP\t0")
        lines.append("")
    lines.append("DATA")

    # every FASTA protein gets its full-length backbone, annotated or not
    for seq_id, seq_len in sorted(lengths.items()):
        fields = [seq_id, str(seq_len)]
        fields += [
            f"{SOURCES[b['source']]}|{b['start']}|{b['end']}|{box_color(b)}|{b['label']}"
            for b in sorted(by_seq.get(seq_id, []), key=lambda x: x["start"])
        ]
        lines.append("\t".join(fields))

    Path(out_path).write_text("\n".join(lines) + "\n")
    per_source = defaultdict(int)
    for b in kept:
        per_source[b["source"]] += 1
    print(
        f"  Wrote {out_path}: {len(lengths)} proteins ({len(by_seq)} annotated), "
        f"{len(kept)} boxes ("
        + ", ".join(f"{s} {per_source[s]}" for s in SOURCES if per_source[s])
        + f"), {len(legend)} legend entries"
    )


# ── CLI shim ───────────────────────────────────────────────────────────────────

def _csv_set(s: Optional[str]) -> set:
    return {x.strip().lower() for x in s.split(",") if x.strip()} if s else set()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="dbCAN + Pfam + InterProScan + SignalP → one iTOL DATASET_DOMAINS file"
    )
    parser.add_argument("--fasta", required=True,
                        help="Full-length protein FASTA the tools were run on")
    parser.add_argument("--out", required=True, help="Output iTOL annotation file")

    src = parser.add_argument_group("sources (priority order)")
    src.add_argument("--dbcan", help="dbCAN domtblout, or run_dbcan hmmer.out / dbCAN_hmm_results.tsv")
    src.add_argument("--pfam", help="Pfam domtblout")
    src.add_argument("--interproscan", help="InterProScan TSV (-f tsv)")
    src.add_argument("--signalp", help="SignalP 5/6 prediction summary or GFF3")

    flt = parser.add_argument_group("filtering")
    flt.add_argument("--dbcan-evalue", type=float, default=1e-15,
                     help="dbCAN i-Evalue threshold (default: 1e-15, as run_dbcan)")
    flt.add_argument("--dbcan-coverage", type=float, default=0.35,
                     help="dbCAN minimum HMM coverage (default: 0.35, as run_dbcan)")
    flt.add_argument("--pfam-evalue", type=float, default=1e-5,
                     help="Pfam i-Evalue threshold (default: 1e-5)")
    flt.add_argument("--ips-analyses",
                     help="Only these InterProScan analyses, comma-separated (e.g. CDD,SMART,Gene3D)")
    flt.add_argument("--ips-exclude",
                     help="Skip these InterProScan analyses, comma-separated (e.g. MobiDBLite,Coils)")
    flt.add_argument("--max-overlap", type=float, default=0.5,
                     help="Drop a lower-priority box overlapping a placed one by more than this "
                          "fraction of the shorter box (default: 0.5)")
    flt.add_argument("--keep-overlaps", action="store_true",
                     help="Keep all boxes, also overlapping ones")

    out = parser.add_argument_group("output")
    out.add_argument("--ips-label", choices=["interpro", "signature", "description"],
                     default="interpro",
                     help="InterProScan box label: InterPro entry name (falls back to signature "
                          "description), signature accession, or signature description "
                          "(default: interpro)")
    out.add_argument("--show-labels", action="store_true", help="Print labels inside the boxes")
    out.add_argument("--dataset-label", default="Domains", help="iTOL dataset label")
    args = parser.parse_args()

    convert_sources(
        args.fasta, args.out,
        dbcan_path=args.dbcan, pfam_path=args.pfam,
        ips_path=args.interproscan, signalp_path=args.signalp,
        dbcan_evalue=args.dbcan_evalue, dbcan_coverage=args.dbcan_coverage,
        pfam_evalue=args.pfam_evalue,
        ips_label=args.ips_label,
        ips_include=_csv_set(args.ips_analyses), ips_exclude=_csv_set(args.ips_exclude),
        max_overlap=args.max_overlap, keep_overlaps=args.keep_overlaps,
        show_labels=args.show_labels, dataset_label=args.dataset_label,
    )
