#!/usr/bin/env python3
"""
Build high-coverage text + GO annotation tables for a GRNPred expression matrix.

Why: the shipped text_description.csv / gene_go_pairs.csv miss many BEELINE symbols
(~47% text, ~50–80% GO). This script queries MyGene (NCBI/Ensembl-backed) and
writes dataset-local CSVs that match data.py loaders:

  text CSV:  must have column 'gene'; other columns are concatenated into text
  GO CSV:    columns gene, go_term  (GO:#######)

Usage (mouse BEELINE = humanized symbols → query as human):
  python scripts/build_gene_annotations.py \\
    --expr_path inputs/examples/mHSC-GM_nonspecific_chipseq_tf1000/ExpressionData.csv \\
    --species human \\
    --out_dir priors/mHSC-GM_nonspecific_chipseq_tf1000

  python main.py ... \\
    --priors_path priors/.../gene_go_pairs.csv \\
    --anno_csv_path priors/.../text_description.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


def _chunked(xs: list[str], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def _extract_go_ids(go_block) -> set[str]:
    ids: set[str] = set()
    if not go_block:
        return ids
    if not isinstance(go_block, dict):
        return ids
    for branch in ("BP", "MF", "CC"):
        items = go_block.get(branch)
        if not items:
            continue
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if isinstance(item, dict):
                gid = item.get("id")
                if gid and str(gid).startswith("GO:"):
                    ids.add(str(gid))
            elif isinstance(item, str) and item.startswith("GO:"):
                ids.add(item)
    return ids


def fetch_annotations(
    genes: list[str],
    species: str,
    batch_size: int = 500,
    sleep_s: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    try:
        import mygene
    except ImportError as e:
        raise SystemExit(
            "mygene is required: pip/conda install mygene"
        ) from e

    mg = mygene.MyGeneInfo()
    text_rows: list[dict] = []
    go_rows: list[tuple[str, str]] = []
    stats = {
        "n_genes": len(genes),
        "n_hit": 0,
        "n_with_summary": 0,
        "n_with_name": 0,
        "n_with_go": 0,
        "n_go_pairs": 0,
        "n_notfound": 0,
    }

    # Prefer first hit per query; keep original expression symbol as gene key
    seen_query: set[str] = set()

    for batch in _chunked(genes, batch_size):
        res = mg.querymany(
            batch,
            scopes="symbol,alias",
            species=species,
            fields="symbol,name,summary,go",
            as_dataframe=False,
            returnall=False,
        )
        for r in res:
            q = str(r.get("query", "")).strip()
            if not q or q in seen_query:
                continue
            seen_query.add(q)
            if r.get("notfound"):
                stats["n_notfound"] += 1
                # still emit empty text row so coverage accounting is explicit
                text_rows.append(
                    {"gene": q, "name": "", "description": "", "function": ""}
                )
                continue
            stats["n_hit"] += 1
            name = (r.get("name") or "").strip()
            summary = (r.get("summary") or "").strip()
            if name:
                stats["n_with_name"] += 1
            if summary:
                stats["n_with_summary"] += 1
            text_rows.append(
                {
                    "gene": q,
                    "name": name,
                    "description": summary,
                    "function": summary,  # data.py concatenates all non-gene cols
                }
            )
            go_ids = _extract_go_ids(r.get("go"))
            if go_ids:
                stats["n_with_go"] += 1
                for gid in sorted(go_ids):
                    go_rows.append((q, gid))
        time.sleep(sleep_s)

    # genes mygene silently dropped
    for g in genes:
        if g not in seen_query:
            stats["n_notfound"] += 1
            text_rows.append({"gene": g, "name": "", "description": "", "function": ""})

    text_df = pd.DataFrame(text_rows).drop_duplicates(subset=["gene"], keep="first")
    go_df = pd.DataFrame(go_rows, columns=["gene", "go_term"]).drop_duplicates()
    stats["n_go_pairs"] = int(len(go_df))
    stats["text_nonzero"] = int((text_df["description"].astype(str).str.len() > 0).sum())
    stats["go_gene_coverage"] = int(go_df["gene"].nunique()) if len(go_df) else 0
    return text_df, go_df, stats


def maybe_merge_existing_go(go_df: pd.DataFrame, existing_path: Path | None) -> pd.DataFrame:
    if existing_path is None or not existing_path.is_file():
        return go_df
    old = pd.read_csv(existing_path)
    if not {"gene", "go_term"}.issubset(old.columns):
        return go_df
    merged = pd.concat([go_df, old[["gene", "go_term"]]], ignore_index=True)
    return merged.drop_duplicates()


def maybe_merge_existing_text(
    text_df: pd.DataFrame, existing_path: Path | None, genes: list[str]
) -> pd.DataFrame:
    """Fill empty descriptions from an existing UniProt-style table when gene matches."""
    if existing_path is None or not existing_path.is_file():
        return text_df
    old = pd.read_csv(existing_path, low_memory=False)
    if "gene" not in old.columns:
        return text_df
    non_gene = [c for c in old.columns if c != "gene"]
    old["gene_u"] = old["gene"].astype(str).str.upper()
    # also map entry_name SYMBOL_HUMAN → symbol
    if "entry_name" in old.columns:
        stem = old["entry_name"].astype(str).str.replace(
            r"_(HUMAN|MOUSE)$", "", regex=True
        )
        old = pd.concat(
            [
                old,
                pd.DataFrame(
                    {
                        "gene": stem,
                        "gene_u": stem.str.upper(),
                        **{c: old[c] for c in non_gene},
                    }
                ),
            ],
            ignore_index=True,
        )
    by_u = {}
    for _, r in old.iterrows():
        gu = str(r["gene_u"])
        if gu in by_u:
            continue
        blob = ". ".join(
            str(r[c]).strip()
            for c in non_gene
            if pd.notna(r.get(c)) and str(r[c]).strip()
        )
        if blob:
            by_u[gu] = blob

    filled = 0
    descs = []
    for _, r in text_df.iterrows():
        d = str(r.get("description") or "")
        if not d.strip():
            alt = by_u.get(str(r["gene"]).upper(), "")
            if alt:
                d = alt
                filled += 1
        descs.append(d)
    text_df = text_df.copy()
    text_df["description"] = descs
    text_df["function"] = descs
    print(f"[INFO] filled {filled} empty texts from {existing_path}")
    return text_df


def main() -> int:
    p = argparse.ArgumentParser(description="Build text + GO annotations via MyGene")
    p.add_argument("--expr_path", required=True)
    p.add_argument(
        "--species",
        required=True,
        choices=["human", "mouse"],
        help="MyGene species. Use 'human' for BEELINE mouse (humanized symbols).",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--merge_text_from",
        default=None,
        help="Optional existing text_description.csv to fill gaps",
    )
    p.add_argument(
        "--merge_go_from",
        default=None,
        help="Optional existing gene_go_pairs.csv to union",
    )
    p.add_argument("--batch_size", type=int, default=500)
    args = p.parse_args()

    expr = Path(args.expr_path)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    genes = pd.read_csv(expr, index_col=0).index.astype(str).tolist()
    print(f"[INFO] genes={len(genes)} species={args.species}")

    text_df, go_df, stats = fetch_annotations(
        genes, species=args.species, batch_size=args.batch_size
    )
    text_df = maybe_merge_existing_text(
        text_df,
        Path(args.merge_text_from) if args.merge_text_from else None,
        genes,
    )
    go_df = maybe_merge_existing_go(
        go_df, Path(args.merge_go_from) if args.merge_go_from else None
    )
    # restrict GO to this dataset's genes (keep original symbol keys)
    gene_set = set(genes)
    go_df = go_df[go_df["gene"].isin(gene_set)].copy()

    text_path = out / "text_description.csv"
    go_path = out / "gene_go_pairs.csv"
    text_df.to_csv(text_path, index=False)
    go_df.to_csv(go_path, index=False)

    stats["text_nonzero_final"] = int(
        (text_df["description"].astype(str).str.len() > 0).sum()
    )
    stats["go_gene_coverage_final"] = int(go_df["gene"].nunique()) if len(go_df) else 0
    stats["out_text"] = str(text_path.resolve())
    stats["out_go"] = str(go_path.resolve())
    man = out / "ANNOTATION_MANIFEST.json"
    man.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"[INFO] wrote {text_path}")
    print(f"[INFO] wrote {go_path}")
    print(f"[INFO] wrote {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
