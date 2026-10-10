#!/usr/bin/env python3
"""Build yeast TF motif priors (FIMO) for GRNPred yeast_grnformer.

Uses:
  - SGD GFF for TSS + systematic→standard TF name mapping
  - Ensembl R64 genome FASTA (resources/genomes/R64/genome.fa)
  - JASPAR CORE fungi MEME motifs

Writes priors/yeast_grnformer/{tf_binding_logodds.csv,motif_score.tsv,MANIFEST.json}
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import tempfile
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rebuild_sequence_priors import (  # noqa: E402
    aggregate_fimo,
    load_fasta_chroms,
    run_bedtools_getfasta,
    run_fimo,
)

DEFAULT_RES = ROOT / "resources/genomes/R64"
DEFAULT_JASPAR = ROOT / "resources/jaspar/JASPAR2026_CORE_fungi_non-redundant_pfms_meme.txt"
JASPAR_URL = (
    "https://jaspar.elixir.no/download/data/2026/CORE/"
    "JASPAR2026_CORE_fungi_non-redundant_pfms_meme.txt"
)
DEFAULT_EXPR = ROOT / "inputs/examples/yeast_grnformer/ExpressionData.csv"
DEFAULT_TF = ROOT / "TFYeast.csv"
DEFAULT_GFF = DEFAULT_RES / "sgd_saccharomyces_cerevisiae.gff.gz"
DEFAULT_OUT = ROOT / "priors/yeast_grnformer"


def normalize_yeast_chrom(chrom: str, fasta_chroms: set[str]) -> str | None:
    c = chrom.strip()
    mapping = {
        "chrmt": "Mito",
        "MT": "Mito",
        "Mito": "Mito",
    }
    if c in mapping:
        c = mapping[c]
    if c.startswith("chr"):
        c = c[3:]
    if c in fasta_chroms:
        return c
    return None


def parse_sgd_gff_tss(gff_gz: Path, gene_symbols: set[str] | None = None) -> pd.DataFrame:
    want = gene_symbols
    rows = []
    with gzip.open(gff_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            parts = line.rstrip("\n").split("\t")
            chrom, start, end, strand = parts[0], int(parts[3]), int(parts[4]), parts[6]
            attrs = {
                k: unquote(v)
                for k, v in (x.split("=", 1) for x in parts[8].split(";") if "=" in x)
            }
            name = attrs.get("Name") or attrs.get("ID")
            if not name:
                continue
            if want is not None and name not in want:
                continue
            tss = start if strand == "+" else end
            rows.append(
                {
                    "gene_name": name,
                    "chr": chrom,
                    "tss": tss,
                    "strand": strand,
                }
            )
    return pd.DataFrame(rows)


def build_tf_alias_map(gff_gz: Path, tf_systematic: set[str]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    with gzip.open(gff_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            attrs = {
                k: unquote(v)
                for k, v in (x.split("=", 1) for x in line.rstrip().split("\t")[8].split(";") if "=" in x)
            }
            sysname = attrs.get("Name") or attrs.get("ID")
            if sysname not in tf_systematic:
                continue
            if attrs.get("gene"):
                aliases[sysname].add(attrs["gene"].upper())
            if attrs.get("Alias"):
                for a in attrs["Alias"].split(","):
                    a = a.strip().upper()
                    if re.fullmatch(r"[A-Z0-9-]+", a) and len(a) <= 12:
                        aliases[sysname].add(a)
    return dict(aliases)


def filter_meme_for_yeast_tfs(
    meme_path: Path,
    tf_systematic: list[str],
    alias_map: dict[str, set[str]],
    out_path: Path,
) -> tuple[Path, dict[str, str]]:
    """Map JASPAR motif alt IDs → systematic TF names used in expression data."""
    symbol_to_sys: dict[str, str] = {}
    for sysname in tf_systematic:
        for sym in alias_map.get(sysname, ()):
            symbol_to_sys.setdefault(sym.upper(), sysname)

    motif_to_tf: dict[str, str] = {}
    keep_blocks: list[str] = []
    header: list[str] = []
    with meme_path.open() as fh:
        text = fh.read()
    parts = re.split(r"(?=^MOTIF\s)", text, flags=re.M)
    if parts:
        header = [parts[0]]

    for block in parts[1:]:
        first = block.splitlines()[0]
        m = re.match(r"^MOTIF\s+(\S+)(?:\s+(\S+))?", first)
        if not m:
            continue
        motif_id, alt = m.group(1), m.group(2)
        candidates: list[str] = []
        if alt:
            candidates.append(alt)
            candidates.extend(re.split(r"[::/]", alt))
        candidates.append(motif_id.split(".")[0])

        matched_sys = None
        for c in candidates:
            cu = c.upper()
            if cu in symbol_to_sys:
                matched_sys = symbol_to_sys[cu]
                break
        if matched_sys is None:
            continue
        keep_blocks.append(block)
        key = alt or motif_id
        motif_to_tf[key] = matched_sys
        motif_to_tf[motif_id] = matched_sys

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        out.write("".join(header))
        for b in keep_blocks:
            out.write(b if b.endswith("\n") else b + "\n")
    return out_path, motif_to_tf


def build_promoter_bed_yeast(
    genes: list[str],
    tss_df: pd.DataFrame,
    fasta_chroms: set[str],
    window: int,
    bed_path: Path,
) -> int:
    tss_lookup = {r.gene_name: r for r in tss_df.itertuples(index=False)}
    n = 0
    with bed_path.open("w") as out:
        for sym in genes:
            if sym not in tss_lookup:
                continue
            r = tss_lookup[sym]
            chrom = normalize_yeast_chrom(r.chr, fasta_chroms)
            if chrom is None:
                continue
            start = max(0, int(r.tss) - window - 1)
            end = int(r.tss) + window
            strand = r.strand if r.strand in ("+", "-") else "+"
            out.write(f"{chrom}\t{start}\t{end}\t{sym}\t0\t{strand}\n")
            n += 1
    return n


def ensure_jaspar(meme_path: Path) -> None:
    if meme_path.is_file() and meme_path.stat().st_size > 0:
        return
    meme_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DL] {JASPAR_URL}\n     -> {meme_path}")
    urllib.request.urlretrieve(JASPAR_URL, meme_path)


def main() -> int:
    p = argparse.ArgumentParser(description="Build yeast FIMO motif priors for GRNPred")
    p.add_argument("--expr_path", default=str(DEFAULT_EXPR))
    p.add_argument("--tf_path", default=str(DEFAULT_TF))
    p.add_argument("--gff", default=str(DEFAULT_GFF))
    p.add_argument("--genome_fasta", default=str(DEFAULT_RES / "genome.fa"))
    p.add_argument("--meme", default=str(DEFAULT_JASPAR))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--window", type=int, default=2000)
    p.add_argument("--fimo_pvalue", type=float, default=1e-4)
    p.add_argument("--keep_tmp", action="store_true")
    args = p.parse_args()

    for tool in ("bedtools", "fimo"):
        if shutil.which(tool) is None:
            print(f"ERROR: {tool} not found on PATH", file=sys.stderr)
            return 1

    expr_path = Path(args.expr_path)
    tf_path = Path(args.tf_path)
    gff = Path(args.gff)
    fasta = Path(args.genome_fasta)
    fai = Path(str(fasta) + ".fai")
    meme_path = Path(args.meme)
    out_dir = Path(args.out_dir)

    for path in (expr_path, tf_path, gff, fasta, fai):
        if not path.is_file():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 1

    ensure_jaspar(meme_path)

    genes = pd.read_csv(expr_path, index_col=0).index.astype(str).tolist()
    tfs = pd.read_csv(tf_path).iloc[:, 0].astype(str).tolist()
    tfs = [t for t in tfs if t in set(genes)]
    tf_set = set(tfs)

    print(f"[INFO] genes={len(genes)} tfs_in_expr={len(tfs)}")

    tss_df = parse_sgd_gff_tss(gff, gene_symbols=set(genes))
    print(f"[INFO] SGD TSS for expression genes: {len(tss_df)}")

    alias_map = build_tf_alias_map(gff, tf_set)
    print(f"[INFO] TF alias map: {len(alias_map)} TFs")

    fasta_chroms = load_fasta_chroms(fai)

    if args.keep_tmp:
        tmp = Path(tempfile.mkdtemp(prefix="yeast_priors_"))
        tmp_cm = None
        print(f"[INFO] keeping tmp: {tmp}")
    else:
        tmp_cm = tempfile.TemporaryDirectory(prefix="yeast_priors_")
        tmp = Path(tmp_cm.name)

    try:
        bed = tmp / "promoters.bed"
        n_bed = build_promoter_bed_yeast(genes, tss_df, fasta_chroms, args.window, bed)
        print(f"[INFO] promoter BED intervals: {n_bed}")
        if n_bed == 0:
            print("ERROR: no promoters written", file=sys.stderr)
            return 1

        fa = tmp / "promoters.fa"
        print("[INFO] bedtools getfasta …")
        run_bedtools_getfasta(bed, fasta, fa)

        meme_filt_path = tmp / "tfs.meme"
        print("[INFO] filtering JASPAR fungi MEME …")
        meme_filt, motif_to_tf = filter_meme_for_yeast_tfs(
            meme_path, tfs, alias_map, meme_filt_path
        )
        n_motif_tfs = len(set(motif_to_tf.values()))
        print(f"[INFO] motifs matched: {n_motif_tfs} TFs, {len(motif_to_tf)} motif keys")
        if not motif_to_tf:
            print("ERROR: no JASPAR fungi motifs matched yeast TF list", file=sys.stderr)
            return 1

        fimo_dir = tmp / "fimo_out"
        print(f"[INFO] running FIMO (p≤{args.fimo_pvalue}) …")
        fimo_tsv = run_fimo(meme_filt, fa, fimo_dir, args.fimo_pvalue)

        out_dir.mkdir(parents=True, exist_ok=True)
        motif_score_path = out_dir / "motif_score.tsv"
        shutil.copyfile(fimo_tsv, motif_score_path)

        agg = aggregate_fimo(fimo_tsv, motif_to_tf, set(genes))
        bind_path = out_dir / "tf_binding_logodds.csv"
        agg.to_csv(bind_path, index=False)
        print(f"[INFO] wrote {bind_path} rows={len(agg)}")
        print(f"[INFO] wrote {motif_score_path}")

        manifest = {
            "dataset": "yeast_grnformer",
            "species": "saccharomyces_cerevisiae",
            "assembly": "R64-1-1",
            "expr_path": str(expr_path.resolve()),
            "tf_path": str(tf_path.resolve()),
            "gff": str(gff.resolve()),
            "genome_fasta": str(fasta.resolve()),
            "meme": str(meme_path.resolve()),
            "window_bp": args.window,
            "fimo_pvalue": args.fimo_pvalue,
            "n_genes": len(genes),
            "n_tfs_scanned": len(tfs),
            "n_promoters": n_bed,
            "n_motif_tfs": n_motif_tfs,
            "n_binding_pairs": int(len(agg)),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                "tf_binding_logodds.csv": str(bind_path.resolve()),
                "motif_score.tsv": str(motif_score_path.resolve()),
            },
        }
        man_path = out_dir / "MANIFEST_motif.json"
        man_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[INFO] wrote {man_path}")
    finally:
        if tmp_cm is not None:
            tmp_cm.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
