#!/usr/bin/env python3
"""
Rebuild species-correct sequence priors for GRNPred.

Pipeline (all writable paths under GRNPred/):
  1. Map expression gene symbols → genome gene names (ortholog map for mouse
     humanized BEELINE symbols; identity / case-insensitive for human).
  2. Parse GTF → TSS; build promoter BED (TSS ± window).
  3. bedtools getfasta → promoter FASTA.
  4. Filter JASPAR MEME to expressed TFs; run FIMO.
  5. Aggregate best FIMO score per (motif TF, target gene).
  6. Write priors/<dataset>/{tf_binding_logodds.csv, motif_score.tsv, MANIFEST.json}.

Outputs match data.py loaders:
  tf_binding_logodds.csv  columns: motif, gene, best_score
  motif_score.tsv         FIMO-like or motif/gene/best_score (also accepted)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOURCES = ROOT / "resources"
DEFAULT_PRIORS = ROOT / "priors"


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _normalize_chrom(chrom: str, fasta_chroms: set[str] | None = None) -> str:
    c = str(chrom).strip()
    if not c:
        return c
    candidates = [c]
    if c.startswith("chr"):
        candidates.append(c[3:])
    else:
        candidates.append("chr" + c)
    if fasta_chroms is None:
        return c
    for cand in candidates:
        if cand in fasta_chroms:
            return cand
    return c


def load_fasta_chroms(fai_path: Path) -> set[str]:
    chroms: set[str] = set()
    with fai_path.open() as fh:
        for line in fh:
            if line.strip():
                chroms.add(line.split("\t", 1)[0])
    return chroms


def parse_gtf_tss(gtf_path: Path, gene_symbols: set[str] | None = None) -> pd.DataFrame:
    """One TSS per gene_name (Ensembl gene_name attribute)."""
    want = {g.upper() for g in gene_symbols} if gene_symbols else None
    # gene_name -> list of (chrom, tss, strand)
    buckets: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    attr_re = re.compile(r'(\w+)\s+"([^"]*)"')

    with _open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = dict(attr_re.findall(parts[8]))
            name = attrs.get("gene_name") or attrs.get("gene")
            if not name:
                continue
            if want is not None and name.upper() not in want:
                continue
            chrom, start, end, strand = parts[0], int(parts[3]), int(parts[4]), parts[6]
            tss = start if strand == "+" else end
            buckets[name].append((chrom, tss, strand))

    rows = []
    for name, hits in buckets.items():
        # Prefer canonical: smallest start for +, largest end for -
        plus = [h for h in hits if h[2] == "+"]
        minus = [h for h in hits if h[2] == "-"]
        if plus:
            chrom, tss, strand = min(plus, key=lambda x: x[1])
        elif minus:
            chrom, tss, strand = max(minus, key=lambda x: x[1])
        else:
            chrom, tss, strand = hits[0]
        rows.append({"gene_name": name, "chr": chrom, "tss": int(tss), "strand": strand})
    return pd.DataFrame(rows)


def load_ortholog_map(ortholog_tsv: Path | None) -> dict[str, str]:
    """Map humanized / human symbol (UPPER) -> mouse gene_name."""
    mapping: dict[str, str] = {}
    if ortholog_tsv is None or not ortholog_tsv.is_file():
        return mapping
    df = pd.read_csv(ortholog_tsv, sep="\t")
    # Normalize headers: lower + strip + spaces→underscores
    cols = {re.sub(r"\s+", "_", c.strip().lower()): c for c in df.columns}
    human_col = None
    mouse_col = None
    for key in (
        "external_gene_name",
        "gene_name",
        "human_gene",
        "human_symbol",
        "human_gene_name",
    ):
        if key in cols:
            human_col = cols[key]
            break
    for key in (
        "mmusculus_homolog_associated_gene_name",
        "mouse_gene_name",
        "mouse_gene",
        "mouse_symbol",
        "ortholog_gene",
    ):
        if key in cols:
            mouse_col = cols[key]
            break
    if human_col is None or mouse_col is None:
        # BioMart / mygene export: human symbol, ensembl id, mouse symbol, …
        if df.shape[1] >= 3:
            human_col, mouse_col = df.columns[0], df.columns[2]
        elif df.shape[1] >= 2:
            human_col, mouse_col = df.columns[0], df.columns[1]
        else:
            return mapping
    otype_col = cols.get("mmusculus_homolog_orthology_type") or cols.get("mouse_homology_type")
    for _, r in df.iterrows():
        h = r.get(human_col)
        m = r.get(mouse_col)
        if pd.isna(h) or pd.isna(m) or not str(m).strip():
            continue
        if otype_col is not None and not pd.isna(r.get(otype_col)):
            ot = str(r[otype_col]).lower()
            if "one2one" not in ot and "1_to_1" not in ot and "ortholog_one2one" not in ot:
                # Prefer one2one; keep first seen otherwise
                if str(h).upper() in mapping:
                    continue
        mapping[str(h).upper()] = str(m).strip()
    return mapping


def map_symbols_to_genome(
    symbols: list[str],
    tss_df: pd.DataFrame,
    species: str,
    ortholog_map: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """
    Return (expr_symbol -> genome_gene_name, unmatched_symbols).
    Mouse: ortholog map then case-insensitive GTF match.
    Human: identity / case-insensitive GTF match.
    """
    gtf_by_upper = {str(n).upper(): str(n) for n in tss_df["gene_name"]}
    mapped: dict[str, str] = {}
    unmatched: list[str] = []
    for sym in symbols:
        s = str(sym).strip()
        if not s:
            continue
        genome_name = None
        if species == "mouse":
            ortho = ortholog_map.get(s.upper())
            if ortho and ortho.upper() in gtf_by_upper:
                genome_name = gtf_by_upper[ortho.upper()]
            elif s.upper() in gtf_by_upper:
                genome_name = gtf_by_upper[s.upper()]
        else:
            if s.upper() in gtf_by_upper:
                genome_name = gtf_by_upper[s.upper()]
            elif s in set(tss_df["gene_name"]):
                genome_name = s
        if genome_name is None:
            unmatched.append(s)
        else:
            mapped[s] = genome_name
    return mapped, unmatched


def build_promoter_bed(
    symbols: list[str],
    symbol_to_genome: dict[str, str],
    tss_df: pd.DataFrame,
    fasta_chroms: set[str],
    window: int,
    bed_path: Path,
) -> int:
    """BED name = original expression symbol (so FIMO sequence_name maps back)."""
    tss_lookup = {r.gene_name: r for r in tss_df.itertuples(index=False)}
    n = 0
    with bed_path.open("w") as out:
        for sym in symbols:
            gname = symbol_to_genome.get(sym)
            if gname is None or gname not in tss_lookup:
                continue
            r = tss_lookup[gname]
            chrom = _normalize_chrom(r.chr, fasta_chroms)
            if chrom not in fasta_chroms:
                continue
            start = max(0, int(r.tss) - window - 1)  # BED 0-based
            end = int(r.tss) + window
            strand = r.strand if r.strand in ("+", "-") else "+"
            # nameOnly: use expression symbol so aggregation uses dataset IDs
            out.write(f"{chrom}\t{start}\t{end}\t{sym}\t0\t{strand}\n")
            n += 1
    return n


def filter_meme_for_tfs(meme_path: Path, tf_names: list[str], out_path: Path) -> tuple[Path, dict[str, str]]:
    """Keep motifs whose MOTIF alt name / ID matches a TF (case-insensitive)."""
    want = {t.upper() for t in tf_names}
    motif_to_tf: dict[str, str] = {}
    keep_blocks: list[str] = []
    header: list[str] = []
    with meme_path.open() as fh:
        text = fh.read()
    # Split on MOTIF lines
    parts = re.split(r"(?=^MOTIF\s)", text, flags=re.M)
    if parts:
        header = [parts[0]]
    for block in parts[1:]:
        first = block.splitlines()[0]
        # MOTIF MA0001.1 Arnt   OR  MOTIF Arnt
        m = re.match(r"^MOTIF\s+(\S+)(?:\s+(\S+))?", first)
        if not m:
            continue
        motif_id, alt = m.group(1), m.group(2)
        candidates = []
        if alt:
            candidates.append(alt)
        candidates.append(motif_id)
        # strip version suffix
        candidates.append(motif_id.split(".")[0])
        matched_tf = None
        for c in candidates:
            cu = c.upper()
            if cu in want:
                matched_tf = next(t for t in tf_names if t.upper() == cu)
                break
            # JASPAR sometimes uses TF::TF dimers
            for piece in re.split(r"[::/]", c):
                if piece.upper() in want:
                    matched_tf = next(t for t in tf_names if t.upper() == piece.upper())
                    break
            if matched_tf:
                break
        if matched_tf:
            keep_blocks.append(block)
            key = alt or motif_id
            motif_to_tf[key] = matched_tf
            motif_to_tf[motif_id] = matched_tf
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        out.write("".join(header))
        if keep_blocks and not any("MOTIF" in h for h in header):
            pass
        for b in keep_blocks:
            out.write(b if b.endswith("\n") else b + "\n")
    return out_path, motif_to_tf


def run_bedtools_getfasta(bed: Path, fasta: Path, out_fa: Path) -> None:
    cmd = [
        "bedtools", "getfasta",
        "-fi", str(fasta),
        "-bed", str(bed),
        "-fo", str(out_fa),
        "-nameOnly",
        "-s",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bedtools getfasta failed:\n{r.stderr}")
    if not out_fa.is_file() or out_fa.stat().st_size == 0:
        raise RuntimeError("bedtools getfasta produced empty FASTA")


def run_fimo(meme: Path, fasta: Path, out_dir: Path, pvalue: float, max_stored_scores: int = 1000000) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "fimo",
        f"--thresh", str(pvalue),
        "--max-stored-scores", str(max_stored_scores),
        "--oc", str(out_dir),
        str(meme),
        str(fasta),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tsv = out_dir / "fimo.tsv"
    # fimo often returns non-zero even on success; check output
    if not tsv.is_file():
        raise RuntimeError(f"FIMO failed (rc={r.returncode}):\n{r.stderr[-2000:]}\n{r.stdout[-1000:]}")
    return tsv


def aggregate_fimo(
    fimo_tsv: Path,
    motif_to_tf: dict[str, str],
    expr_symbols: set[str],
) -> pd.DataFrame:
    rows = []
    with fimo_tsv.open() as fh:
        header = None
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            if header is None:
                header = line.rstrip("\n").split("\t")
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rec = dict(zip(header, parts))
            motif_id = rec.get("motif_id", "")
            motif_alt = rec.get("motif_alt_id", "")
            seq = rec.get("sequence_name", "")
            # strip possible bedtools suffix
            gene = seq.split("(")[0].strip()
            if gene not in expr_symbols:
                # sometimes name has ::
                gene = gene.split("::")[0]
            if gene not in expr_symbols:
                continue
            tf = motif_to_tf.get(motif_alt) or motif_to_tf.get(motif_id)
            if tf is None:
                # try alt upper match against map values already done
                continue
            try:
                score = float(rec.get("score", "nan"))
            except ValueError:
                continue
            rows.append((tf, gene, score))
    if not rows:
        return pd.DataFrame(columns=["motif", "gene", "best_score"])
    df = pd.DataFrame(rows, columns=["motif", "gene", "best_score"])
    return df.groupby(["motif", "gene"], as_index=False)["best_score"].max()


def resolve_paths(args: argparse.Namespace) -> dict:
    res = Path(args.resources_dir)
    if args.species == "mouse":
        genome_dir = res / "genomes" / "GRCm39"
        assembly = "GRCm39"
    else:
        genome_dir = res / "genomes" / "GRCh38"
        assembly = "GRCh38"
    fasta = Path(args.genome_fasta) if args.genome_fasta else genome_dir / "genome.fa"
    gtf = Path(args.gtf) if args.gtf else genome_dir / "annotation.gtf"
    meme = Path(args.meme) if args.meme else res / "jaspar" / "JASPAR2026_CORE_vertebrates.meme"
    orth = Path(args.ortholog_tsv) if args.ortholog_tsv else res / "orthologs" / "human_mouse_orthologs.tsv"
    return {
        "assembly": assembly,
        "fasta": fasta,
        "gtf": gtf,
        "meme": meme,
        "ortholog": orth if orth.is_file() else None,
        "fai": Path(str(fasta) + ".fai"),
    }


def dataset_name_from_expr(expr_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    # inputs/examples/<dataset>/ExpressionData.csv
    if expr_path.parent.name and expr_path.parent.name not in (".", ""):
        return expr_path.parent.name
    return expr_path.stem.replace("_ExpressionData", "")


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild species-correct GRNPred sequence priors")
    p.add_argument("--species", choices=["mouse", "human"], required=True)
    p.add_argument("--expr_path", required=True, help="ExpressionData.csv")
    p.add_argument("--tf_path", required=True, help="TF list CSV (first column)")
    p.add_argument("--out_dir", default=None, help="priors/<dataset> (default derived)")
    p.add_argument("--dataset_name", default=None)
    p.add_argument("--resources_dir", default=str(DEFAULT_RESOURCES))
    p.add_argument("--genome_fasta", default=None)
    p.add_argument("--gtf", default=None)
    p.add_argument("--meme", default=None)
    p.add_argument("--ortholog_tsv", default=None)
    p.add_argument("--window", type=int, default=2000, help="TSS ± window bp")
    p.add_argument("--fimo_pvalue", type=float, default=1e-4)
    p.add_argument("--keep_tmp", action="store_true")
    args = p.parse_args()

    for tool in ("bedtools", "fimo"):
        if shutil.which(tool) is None:
            print(f"ERROR: {tool} not found on PATH", file=sys.stderr)
            return 1

    paths = resolve_paths(args)
    for key in ("fasta", "gtf", "meme"):
        if not paths[key].is_file():
            print(f"ERROR: missing {key}: {paths[key]}", file=sys.stderr)
            return 1
    if not paths["fai"].is_file():
        print(f"ERROR: missing FASTA index: {paths['fai']} (run samtools faidx)", file=sys.stderr)
        return 1

    expr_path = Path(args.expr_path)
    tf_path = Path(args.tf_path)
    genes = pd.read_csv(expr_path, index_col=0).index.astype(str).tolist()
    tfs = pd.read_csv(tf_path).iloc[:, 0].astype(str).tolist()
    # Only scan TFs present in expression matrix (and with motifs)
    tfs = [t for t in tfs if t in set(genes)]
    ds = dataset_name_from_expr(expr_path, args.dataset_name)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_PRIORS / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset={ds} species={args.species} assembly={paths['assembly']}")
    print(f"[INFO] genes={len(genes)} tfs_in_expr={len(tfs)}")

    # Restrict GTF parse to ortholog targets + identity matches for speed
    orth_map = load_ortholog_map(paths["ortholog"]) if args.species == "mouse" else {}
    print(f"[INFO] ortholog map entries: {len(orth_map)}")

    # Parse full gene features once (needed for accurate TSS); filter later
    print(f"[INFO] parsing GTF: {paths['gtf']}")
    tss_df = parse_gtf_tss(paths["gtf"], gene_symbols=None)
    print(f"[INFO] GTF genes with TSS: {len(tss_df)}")

    symbol_to_genome, unmatched = map_symbols_to_genome(genes, tss_df, args.species, orth_map)
    print(f"[INFO] mapped symbols: {len(symbol_to_genome)} / {len(genes)}; unmatched={len(unmatched)}")
    unmatched_path = out_dir / "unmatched_symbols.txt"
    unmatched_path.write_text("\n".join(unmatched) + ("\n" if unmatched else ""))

    fasta_chroms = load_fasta_chroms(paths["fai"])

    if args.keep_tmp:
        tmp = Path(tempfile.mkdtemp(prefix="grnpred_priors_"))
        print(f"[INFO] keeping tmp: {tmp}")
        tmp_cm = None
    else:
        tmp_cm = tempfile.TemporaryDirectory(prefix="grnpred_priors_")
        tmp = Path(tmp_cm.name)

    try:
        bed = tmp / "promoters.bed"
        n_bed = build_promoter_bed(
            genes, symbol_to_genome, tss_df, fasta_chroms, args.window, bed
        )
        print(f"[INFO] promoter BED intervals: {n_bed}")
        if n_bed == 0:
            print("ERROR: no promoters written — check ortholog/GTF/chrom matching", file=sys.stderr)
            return 1

        fa = tmp / "promoters.fa"
        print("[INFO] bedtools getfasta …")
        run_bedtools_getfasta(bed, paths["fasta"], fa)

        meme_filt = tmp / "tfs.meme"
        print("[INFO] filtering JASPAR MEME …")
        meme_filt, motif_to_tf = filter_meme_for_tfs(paths["meme"], tfs, meme_filt)
        print(
            f"[INFO] motifs matched to TFs: {len(set(motif_to_tf.values()))} TFs, "
            f"{len(motif_to_tf)} motif keys"
        )
        if not motif_to_tf:
            print("ERROR: no JASPAR motifs matched TF list", file=sys.stderr)
            return 1

        fimo_dir = tmp / "fimo_out"
        print(f"[INFO] running FIMO (p≤{args.fimo_pvalue}) …")
        fimo_tsv = run_fimo(meme_filt, fa, fimo_dir, args.fimo_pvalue)

        # Copy full FIMO as motif_score.tsv (data.py accepts FIMO columns)
        motif_score_path = out_dir / "motif_score.tsv"
        shutil.copyfile(fimo_tsv, motif_score_path)

        agg = aggregate_fimo(fimo_tsv, motif_to_tf, set(genes))
        bind_path = out_dir / "tf_binding_logodds.csv"
        agg.to_csv(bind_path, index=False)
        print(f"[INFO] wrote {bind_path} rows={len(agg)}")
        print(f"[INFO] wrote {motif_score_path}")

        manifest = {
            "dataset": ds,
            "species": args.species,
            "assembly": paths["assembly"],
            "expr_path": str(expr_path.resolve()),
            "tf_path": str(tf_path.resolve()),
            "genome_fasta": str(paths["fasta"].resolve()),
            "gtf": str(paths["gtf"].resolve()),
            "meme": str(paths["meme"].resolve()),
            "ortholog_tsv": str(paths["ortholog"].resolve()) if paths["ortholog"] else None,
            "window_bp": args.window,
            "fimo_pvalue": args.fimo_pvalue,
            "n_genes": len(genes),
            "n_tfs_scanned": len(tfs),
            "n_promoters": n_bed,
            "n_mapped_symbols": len(symbol_to_genome),
            "n_unmatched_symbols": len(unmatched),
            "n_binding_pairs": int(len(agg)),
            "n_motif_tfs": len(set(motif_to_tf.values())),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                "tf_binding_logodds.csv": str(bind_path.resolve()),
                "motif_score.tsv": str(motif_score_path.resolve()),
                "unmatched_symbols.txt": str(unmatched_path.resolve()),
            },
        }
        man_path = out_dir / "MANIFEST.json"
        man_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[INFO] wrote {man_path}")
    finally:
        if tmp_cm is not None:
            tmp_cm.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
