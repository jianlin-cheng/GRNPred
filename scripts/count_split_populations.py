#!/usr/bin/env python3
"""Train/val/test population counts for the 42-setting grid.

A candidate is an undirected TF-involving coexpression pair on that split,
after the same dedup as eval._dedup_tf_role (the population written to
full_prediction.csv). Positives are pairs whose either direction is a
reference edge. Targets are the non-TF endpoints of those pairs.

These counts do not depend on the training seed: TF assignment is greedy,
and every TF-incident edge sits in the center TF's chips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_data
from eval import _dedup_tf_role

CELLS = ["hESC", "hHep", "mESC", "mDC", "mHSC-GM", "mHSC-E", "mHSC-L"]
NETWORKS = ["nonspecific_chipseq", "specific_chipseq", "string"]
TF_SIZES = ["tf500", "tf1000"]
HUMAN = {"hESC", "hHep"}


def settings() -> list[str]:
    return [f"{c}_{n}_{t}" for c in CELLS for n in NETWORKS for t in TF_SIZES]


def cell_of(setting: str) -> str:
    for c in CELLS:
        if setting.startswith(c + "_"):
            return c
    raise ValueError(setting)


def scored_frame(g, split: str) -> pd.DataFrame:
    mask = getattr(g, f"{split}_mask").detach().cpu().numpy().reshape(-1).astype(bool)
    src = g.edge_index[0].detach().cpu().numpy().reshape(-1)[mask]
    dst = g.edge_index[1].detach().cpu().numpy().reshape(-1)[mask]
    y = g.y.detach().cpu().numpy().reshape(-1)[mask]
    genes = list(g.gene_names)
    tf_set = {int(i) for i in g.tf_indices}
    if src.size == 0:
        return pd.DataFrame(columns=["Gene1", "Gene2", "TrueLabel"])
    probs = np.zeros(src.shape[0], dtype=np.float64)
    return _dedup_tf_role(src, dst, probs, y, tf_set, genes, True)


def split_counts(df: pd.DataFrame, tf_names: set[str]) -> dict:
    if df.empty:
        return {"candidates": 0, "positives": 0, "negatives": 0, "tfs": 0, "targets": 0}
    genes = set(df["Gene1"].astype(str)) | set(df["Gene2"].astype(str))
    tfs = genes & tf_names
    return {
        "candidates": int(len(df)),
        "positives": int((df["TrueLabel"] == 1).sum()),
        "negatives": int((df["TrueLabel"] == 0).sum()),
        "tfs": int(len(tfs)),
        "targets": int(len(genes - tf_names)),
    }


def count_setting(setting: str) -> dict:
    cell = cell_of(setting)
    ds = ROOT / "inputs/examples" / setting
    pr = ROOT / "priors" / setting
    tf_path = ROOT / ("TFHumans.csv" if cell in HUMAN else "TFMouse.csv")
    print(f"[LOAD] {setting}", flush=True)
    g = load_data(
        expr_path=str(ds / "ExpressionData.csv"),
        tf_path=str(tf_path),
        priors_path=str(pr / "gene_go_pairs.csv"),
        tf_bind_path=str(pr / "tf_binding_logodds.csv"),
        label_path=str(ds / "refNetwork.csv"),
        corr_threshold=0.1,
        seed=42,
    )
    genes = list(g.gene_names)
    tf_names = {genes[int(i)] for i in g.tf_indices}
    row = {
        "setting": setting,
        "cell": cell,
        "species": "human" if cell in HUMAN else "mouse",
        "network": setting[len(cell) + 1: setting.rfind("_tf")],
        "tf_size": setting.rsplit("_", 1)[-1],
    }
    for split in ("train", "val", "test"):
        df = scored_frame(g, split)
        for k, v in split_counts(df, tf_names).items():
            row[f"{split}_{k}"] = v
    for seed in (123, 456):
        pred = ROOT / f"Results/main42_fullcoexp_seed{seed}" / setting / "full_prediction.csv"
        if not pred.is_file():
            continue
        got = pd.read_csv(pred, usecols=["TrueLabel"])
        row[f"test_csv_seed{seed}_candidates"] = int(len(got))
        row[f"test_csv_seed{seed}_positives"] = int((got["TrueLabel"] == 1).sum())
        row[f"test_csv_seed{seed}_match"] = int(
            row["test_candidates"] == row[f"test_csv_seed{seed}_candidates"]
            and row["test_positives"] == row[f"test_csv_seed{seed}_positives"]
        )
    print(
        f"[COUNT] {setting} "
        f"train {row['train_candidates']}/{row['train_positives']} "
        f"val {row['val_candidates']}/{row['val_positives']} "
        f"test {row['test_candidates']}/{row['test_positives']}",
        flush=True,
    )
    return row


def main() -> None:
    wanted = sys.argv[1:]
    jobs = [s for s in settings() if not wanted or s in wanted]
    if len(jobs) == 1:
        rows = [count_setting(jobs[0])]
    else:
        from multiprocessing import get_context
        ctx = get_context("spawn")
        with ctx.Pool(4) as pool:
            rows = pool.map(count_setting, jobs)
    out = ROOT / "Results/main42_split_population_counts.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    match_cols = [c for c in df.columns if c.endswith("_match")]
    n_bad = 0
    for c in match_cols:
        n_bad += int((df[c] == 0).sum())
    print(f"[WROTE] {out} rows={len(df)} test_mismatches={n_bad}", flush=True)


if __name__ == "__main__":
    main()
