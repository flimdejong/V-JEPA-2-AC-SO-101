"""Average distance-to-goal and latent distance over runs, one plot per run group.

Expects this layout (script lives in a sibling folder of `runs`):

    <parent>/
        <script_folder>/plot_distances.py   <- this file
        runs/
            reach_forward_5cm/data.csv
            reach_forward_5cm_2/data.csv
            reach_forward_5cm_3/data.csv
            reach_up_5cm/data.csv
            ...

Runs are grouped by name with a trailing "_<number>" stripped, so
reach_forward_5cm, reach_forward_5cm_2, reach_forward_5cm_3 form one group.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR.parent / "runs"
OUT_DIR = SCRIPT_DIR  # where the PNGs are saved

DIST_COL = "dist_xz_cm"   # distance to goal (cm)
LATENT_COL = "latent_l2"  # distance in latent space


def group_name(run_name: str) -> str:
    return re.sub(r"_\d+$", "", run_name)


def load_groups():
    groups = {}
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        csv = run_dir / "data.csv"
        if not csv.exists():
            print(f"skipping {run_dir.name}: no data.csv")
            continue
        groups.setdefault(group_name(run_dir.name), []).append(
            (run_dir.name, pd.read_csv(csv))
        )
    return groups


def average(runs, col):
    """Return step, mean, std across runs (aligned on step; uses steps present in all runs)."""
    wide = pd.concat(
        [df.set_index("step")[col].rename(name) for name, df in runs], axis=1
    ).dropna()
    return wide.index.values, wide.mean(axis=1).values, wide.std(axis=1, ddof=0).values


def plot_group(name, runs):
    c1, c2 = "tab:blue", "tab:orange"
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    steps, m, s = average(runs, DIST_COL)
    l1 = ax1.plot(steps, m, marker="o", color=c1, label="Distance to goal (cm)")
    ax1.fill_between(steps, m - s, m + s, color=c1, alpha=0.2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Distance to goal (cm)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    steps, m, s = average(runs, LATENT_COL)
    l2 = ax2.plot(steps, m, marker="s", color=c2, label="Distance to latent (L2)")
    ax2.fill_between(steps, m - s, m + s, color=c2, alpha=0.2)
    ax2.set_ylabel("Latent L2 distance", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax1.set_title(f"{name}  (mean ± std over {len(runs)} runs)")

    fig.tight_layout()
    out = OUT_DIR / f"{name}_distances.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")

def print_summary(name, runs):
    rows = []
    for _, df in runs:
        df = df.sort_values("step")
        d0, d1 = df[DIST_COL].iloc[0], df[DIST_COL].iloc[-1]
        l0, l1 = df[LATENT_COL].iloc[0], df[LATENT_COL].iloc[-1]
        rows.append([d0, d1, d0 - d1, 100 * (l0 - l1) / l0])
    r = pd.DataFrame(rows, columns=["dist_start", "dist_end", "dist_decrease_cm", "latent_decrease_pct"])
    print(f"  {'':22s} {'mean':>8s} {'std':>8s}")
    for col in r.columns:
        print(f"  {col:22s} {r[col].mean():8.2f} {r[col].std(ddof=1) if len(r) > 1 else 0:8.2f}")

def main():
    groups = load_groups()
    if not groups:
        raise SystemExit(f"No runs found in {RUNS_DIR}")
    for name, runs in groups.items():
        print(f"{name}: {[n for n, _ in runs]}")
        plot_group(name, runs)
        print_summary(name, runs)


if __name__ == "__main__":
    main()