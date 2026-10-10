#!/usr/bin/env python3
"""Download S. cerevisiae reference genome + annotations and build yeast priors.

Sources (public, no credentials):
  - Ensembl Fungi R64-1-1: genome FASTA + GTF (sequence / promoter priors)
  - SGD latest GFF: gene descriptions + GO terms (annotation priors)

Outputs under GRNPred/resources/genomes/R64/:
  genome.fa.gz, annotation.gtf.gz, sgd_saccharomyces_cerevisiae.gff.gz
  MANIFEST.json

Outputs under GRNPred/priors/yeast_grnformer/ (for expression gene set):
  gene_go_pairs.csv
  text_description.csv
  MANIFEST.json
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resources/genomes/R64"
PRIORS = ROOT / "priors/yeast_grnformer"
EXPR = ROOT / "inputs/examples/yeast_grnformer/ExpressionData.csv"
TF = ROOT / "TFYeast.csv"

URLS = {
    "genome.fa.gz": (
        "https://ftp.ensembl.org/pub/release-112/fasta/saccharomyces_cerevisiae/"
        "dna/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa.gz"
    ),
    "annotation.gtf.gz": (
        "https://ftp.ensembl.org/pub/release-112/gtf/saccharomyces_cerevisiae/"
        "Saccharomyces_cerevisiae.R64-1-1.112.gtf.gz"
    ),
    "sgd_saccharomyces_cerevisiae.gff.gz": (
        "https://downloads.yeastgenome.org/curation/chromosomal_feature/"
        "saccharomyces_cerevisiae.gff.gz"
    ),
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[DL] {url}\n     -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def gunzip_to(src_gz: Path, dest: Path) -> None:
    if dest.exists() and dest.stat().st_mtime >= src_gz.stat().st_mtime:
        print(f"[SKIP] {dest} up to date")
        return
    print(f"[UNZIP] {src_gz} -> {dest}")
    with gzip.open(src_gz, "rb") as fin, dest.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


def index_fasta(fa: Path) -> None:
    fai = Path(str(fa) + ".fai")
    if fai.exists():
        return
    samtools = shutil.which("samtools")
    if samtools:
        print(f"[FAI] samtools faidx {fa}")
        subprocess.run([samtools, "faidx", str(fa)], check=True)
    else:
        print("[WARN] samtools not found; skipping .fai index")


def parse_sgd_gff(gff_gz: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (gene_go_pairs, text_description) from SGD gene features."""
    go_rows: list[dict[str, str]] = []
    text_rows: list[dict[str, str]] = []

    with gzip.open(gff_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            if cols[2] != "gene":
                continue
            attrs = cols[8]
            fields: dict[str, str] = {}
            for part in attrs.split(";"):
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                fields[k] = unquote(v)

            gene = fields.get("Name") or fields.get("ID")
            if not gene:
                continue

            note = fields.get("Note", "").replace("%20", " ")
            onto = fields.get("Ontology_term", "")
            go_terms = [
                t.strip()
                for t in onto.split(",")
                if t.strip().startswith("GO:")
            ]
            for go in go_terms:
                go_rows.append({"gene": gene, "go_term": go})

            desc = note or fields.get("display", "")
            text_rows.append(
                {
                    "gene": gene,
                    "name": gene,
                    "description": desc,
                    "function": desc,
                }
            )

    go_df = pd.DataFrame(go_rows).drop_duplicates()
    text_df = pd.DataFrame(text_rows).drop_duplicates(subset=["gene"], keep="first")
    return go_df, text_df


def filter_to_expression(
    go_df: pd.DataFrame, text_df: pd.DataFrame, genes: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    go_f = go_df[go_df["gene"].isin(genes)].copy()
    text_f = text_df[text_df["gene"].isin(genes)].copy()
    missing = sorted(genes - set(text_f["gene"]))
    for g in missing:
        text_f = pd.concat(
            [
                text_f,
                pd.DataFrame(
                    [{"gene": g, "name": g, "description": "", "function": ""}]
                ),
            ],
            ignore_index=True,
        )
    return go_f, text_f


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    PRIORS.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        dest = RES / name
        if not dest.exists():
            download(url, dest)

    fa_gz = RES / "genome.fa.gz"
    gtf_gz = RES / "annotation.gtf.gz"
    gff_gz = RES / "sgd_saccharomyces_cerevisiae.gff.gz"

    fa = RES / "genome.fa"
    gtf = RES / "annotation.gtf"
    gunzip_to(fa_gz, fa)
    gunzip_to(gtf_gz, gtf)
    index_fasta(fa)

    if not EXPR.is_file():
        # fall back to raw yeast_grnformer expression
        alt = ROOT / "Data/yeast_grnformer/yeast_expression.csv"
        if alt.is_file():
            EXPR_USE = alt
        else:
            print(f"[WARN] expression not found at {EXPR}; skipping priors filter")
            EXPR_USE = None
    else:
        EXPR_USE = EXPR

    go_df, text_df = parse_sgd_gff(gff_gz)
    print(f"[SGD] parsed genes={len(text_df)} go_pairs={len(go_df)}")

    if EXPR_USE is not None:
        expr = pd.read_csv(EXPR_USE, index_col=0)
        genes = set(expr.index.astype(str))
        go_df, text_df = filter_to_expression(go_df, text_df, genes)
        print(
            f"[FILTER] expression genes={len(genes)} "
            f"go_pairs={len(go_df)} text_rows={len(text_df)} "
            f"genes_with_go={go_df['gene'].nunique()}"
        )

    go_path = PRIORS / "gene_go_pairs.csv"
    text_path = PRIORS / "text_description.csv"
    go_df.to_csv(go_path, index=False)
    text_df.to_csv(text_path, index=False)

    res_manifest = {
        "species": "saccharomyces_cerevisiae",
        "assembly": "R64-1-1 (Ensembl 112) + SGD GFF annotations",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": {k: str(RES / k) for k in URLS},
        "genome_fasta": str(fa),
        "annotation_gtf": str(gtf),
    }
    (RES / "MANIFEST.json").write_text(json.dumps(res_manifest, indent=2) + "\n")

    pri_manifest = {
        "dataset": "yeast_grnformer",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "go_and_text": str(gff_gz),
            "expression": str(EXPR_USE) if EXPR_USE else None,
        },
        "n_genes_text": int(len(text_df)),
        "n_go_pairs": int(len(go_df)),
        "n_genes_with_go": int(go_df["gene"].nunique()) if len(go_df) else 0,
        "outputs": {
            "gene_go_pairs.csv": str(go_path),
            "text_description.csv": str(text_path),
        },
    }
    (PRIORS / "MANIFEST.json").write_text(json.dumps(pri_manifest, indent=2) + "\n")

    print(f"[DONE] genome reference -> {RES}")
    print(f"[DONE] annotation priors -> {PRIORS}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
