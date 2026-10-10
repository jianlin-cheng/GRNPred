#!/usr/bin/env python3
"""Main 42-setting table, current protocol, any seed.

7 cells × 3 networks × 2 TF sizes. Same flags as the mHSC-GM full-coexp
ablations (BCE + pairAUC, hop PE, hops=2, max_nodes=100, no input noise).

  python scripts/run_main42_fullcoexp.py --seed 123
  python scripts/run_main42_fullcoexp.py --seed 456
  python scripts/run_main42_fullcoexp.py --seed 42 123 456
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MIN_FREE_MIB = int(os.environ.get("MIN_FREE_MIB", "18000"))
MAX_OURS_PER_GPU = int(os.environ.get("MAX_OURS_PER_GPU", "4"))

CELLS = ["hESC", "hHep", "mESC", "mDC", "mHSC-GM", "mHSC-E", "mHSC-L"]
NETWORKS = ["nonspecific_chipseq", "specific_chipseq", "string"]
TF_SIZES = ["tf500", "tf1000"]
HUMAN = {"hESC", "hHep"}

PROTOCOL_FLAGS = [
    "--input_noise_std", "0.0",
    "--w_bce", "1.0",
    "--w_ap", "0.0",
    "--w_pairauc", "1.0",
    "--edge_select", "auroc",
    "--hops_tf", "2",
    "--max_nodes_tf", "100",
]


def paper_settings() -> list[str]:
    return [
        f"{cell}_{net}_{tf}"
        for cell in CELLS
        for net in NETWORKS
        for tf in TF_SIZES
    ]


def cell_of(setting: str) -> str:
    for c in CELLS:
        if setting.startswith(c + "_"):
            return c
    raise ValueError(f"cannot parse cell from {setting}")


def tf_csv(cell: str) -> Path:
    return ROOT / ("TFHumans.csv" if cell in HUMAN else "TFMouse.csv")


def allowed_gpus(n_gpu: int) -> set[int]:
    raw = os.environ.get("ALLOWED_GPUS")
    if raw is None or not raw.strip():
        return set(range(n_gpu))
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def gpu_free() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x.strip().split()[0]) for x in out.strip().splitlines()]


def ours_by_gpu(n_gpu: int) -> list[int]:
    counts = [0] * n_gpu
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return counts
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    uuid_to_idx = {u.strip(): i for i, u in enumerate(uuids)}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        uuid, pid = parts[0], parts[1]
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        if "GRNPred/main.py" not in cmd and f"{ROOT}/main.py" not in cmd:
            continue
        gi = uuid_to_idx.get(uuid)
        if gi is not None:
            counts[gi] += 1
    return counts


def running_cmdlines() -> list[str]:
    cmds = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return cmds
    for pid in out.strip().splitlines():
        pid = pid.strip()
        if not pid:
            continue
        try:
            cmds.append(Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode())
        except OSError:
            continue
    return cmds


def is_running_base(base: Path) -> bool:
    needle = str(base)
    return any(needle in cmd for cmd in running_cmdlines())


def metrics_done(base: Path) -> bool:
    return Path(f"{base}_prediction_metrics.csv").is_file()


def out_dir(seed: int) -> Path:
    return ROOT / f"Results/main42_fullcoexp_seed{seed}"


def cmd_for(base: Path, setting: str, seed: int) -> list[str]:
    cell = cell_of(setting)
    ds = ROOT / "inputs/examples" / setting
    pr = ROOT / "priors" / setting
    return [
        PY, str(ROOT / "main.py"),
        "--expr_path", str(ds / "ExpressionData.csv"),
        "--label_path", str(ds / "refNetwork.csv"),
        "--tf_path", str(tf_csv(cell)),
        "--priors_path", str(pr / "gene_go_pairs.csv"),
        "--anno_csv_path", str(pr / "text_description.csv"),
        "--tf_bind_path", str(pr / "tf_binding_logodds.csv"),
        "--tf_bind2_path", str(pr / "motif_score.tsv"),
        "--seed", str(seed),
        "--device", "cuda",
        "--patience", "5",
        "--hidden_dim", "64",
        "--base_name", str(base),
        *PROTOCOL_FLAGS,
    ]


def launch(cmd: list[str], log_path: Path, gpu: int, label: str) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"[LAUNCH] {label} on GPU {gpu}", flush=True)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)


def build_pending(seeds: list[int]) -> list[tuple[str, list[str], Path, Path]]:
    pending = []
    settings = paper_settings()
    if len(settings) != 42:
        raise RuntimeError(f"expected 42 settings, got {len(settings)}")
    for seed in seeds:
        dest = out_dir(seed)
        for setting in settings:
            base = dest / setting / "full"
            if metrics_done(base) or is_running_base(base):
                continue
            label = f"seed{seed}/{setting}"
            pending.append((label, cmd_for(base, setting, seed), Path(f"{base}.log"), base))
    return pending


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the 42-setting full-coexp grid for one or more seeds.")
    p.add_argument(
        "--seed",
        type=int,
        nargs="+",
        required=True,
        help="One or more seeds, e.g. --seed 123 or --seed 42 123 456",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seed
    pending = build_pending(seeds)
    children: list[tuple[str, int, subprocess.Popen]] = []
    print(f"[INFO] pending={len(pending)} jobs  seeds={seeds}", flush=True)
    print(f"[INFO] max_ours_per_gpu={MAX_OURS_PER_GPU}", flush=True)
    print(f"[INFO] queue={[lab for lab, *_ in pending]}", flush=True)
    while pending or children:
        alive = []
        for label, gpu, proc in children:
            rc = proc.poll()
            if rc is None:
                alive.append((label, gpu, proc))
            else:
                print(f"[DONE] {label} gpu={gpu} rc={rc}", flush=True)
        children = alive

        free = gpu_free()
        ours = ours_by_gpu(len(free))
        allowed = allowed_gpus(len(free))
        if pending:
            for gpu, (fr, n_ours) in enumerate(zip(free, ours)):
                if gpu not in allowed:
                    continue
                if not pending:
                    break
                if fr >= MIN_FREE_MIB and n_ours < MAX_OURS_PER_GPU:
                    label, cmd, log_path, _base = pending.pop(0)
                    children.append((label, gpu, launch(cmd, log_path, gpu, label)))
                    ours[gpu] += 1
        time.sleep(20)
    print("[ALL DONE]", flush=True)


if __name__ == "__main__":
    main()
