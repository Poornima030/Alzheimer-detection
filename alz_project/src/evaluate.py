"""
evaluate.py — builds the ablation comparison table from saved per-experiment metrics.

Usage:
    python -m src.evaluate --ablation
"""
import argparse
import os
import pandas as pd

from . import config as cfg
from .utils import load_metrics


def build_ablation_table(experiments=None) -> pd.DataFrame:
    experiments = experiments or cfg.EXPERIMENTS
    rows = []
    for exp in experiments:
        try:
            m = load_metrics(exp)
            rows.append(m)
        except FileNotFoundError:
            print(f"[skip] no saved metrics for '{exp}' yet — run: python -m src.train --experiment {exp}")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("experiment")
        df = df.rename(index={
            "baseline": "Tripathy baseline",
            "cbam": "Baseline + CBAM",
            "cbam_transformer": "Baseline + CBAM + Transformer",
            "cbam_transformer_ssl": "Baseline + CBAM + Transformer + SSL",
        })
    out_path = os.path.join(cfg.RESULTS_DIR, "ablation_table.csv")
    df.to_csv(out_path)
    print(f"Saved ablation table -> {out_path}")
    print(df)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    if args.ablation:
        build_ablation_table()
