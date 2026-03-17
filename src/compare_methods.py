"""
Visual comparison of all available prediction methods.

Loads OOF predictions and test submissions, computes validation metrics,
and generates a 4-panel figure saved to outputs/figures/method_comparison.png.

Panels
------
1. OOF / LOO WMAE per method (bar chart, green = beats baseline)
2. Pairwise Pearson correlation of test submissions (heatmap)
3. Per-perturbation WMAE for methods with OOF predictions (line plot)
4. Mean prediction magnitude per method (sanity check bar chart)

Usage
-----
    python src/compare_methods.py          # save figure only
    python src/compare_methods.py --show   # save + open the figure
    python src/compare_methods.py --output my_name.png
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd

# Set backend before importing pyplot so the script works headless by default.
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import load_ground_truth, compute_wmae, compute_wmae_per_pert

# ── Constants ──────────────────────────────────────────────────────────────

BASELINE_WMAE = 0.1268   # always-predict-zero Kaggle baseline

# Test submission files to load (in display order).
# Add new entries here when you run more methods.
SUBMISSION_FILES = [
    ("submission_knn.csv",             "KNN"),
    ("submission_gnn.csv",             "GNN v1"),
    ("submission_ensemble.csv",        "Ensemble v1"),
    ("submission_ridge.csv",           "Ridge"),
    ("submission_ridge_pairwise.csv",  "Ridge-PW"),
    ("submission_lgbm.csv",            "LGBM"),
    ("submission_mlp.csv",             "MLP"),
    ("submission_gnn_v2_local.csv",    "GNN v2 (local)"),
    ("submission_gnn_v3_local.csv",    "GNN v3 (local)"),
    ("submission_cpa_local.csv",       "CPA (local)"),
    ("submission_gnn_v2.csv",          "GNN v2"),
    ("submission_cpa.csv",             "CPA"),
    ("submission_ensemble_v2.csv",     "Ensemble v2"),
]

# OOF prediction files (must have 'pert_id' column + gene columns for training perts).
OOF_FILES = [
    ("oof_gnn.csv",          "GNN v1"),
    ("oof_gnn_v2.csv",       "GNN v2"),
    ("oof_gnn_v2_local.csv", "GNN v2 (local)"),
    ("oof_gnn_v3_local.csv", "GNN v3 (local)"),
    ("oof_lgbm.csv",         "LGBM"),
    ("oof_mlp.csv",          "MLP"),
]

# Pre-computed LOO / CV WMAE for methods that don't produce an OOF file.
PRECOMPUTED_WMAE = {
    "KNN":          0.1232,   # LOO WMAE (leave-one-out, knn_baseline.py)
    "CPA (local)":  0.1414,   # train WMAE from checkpoint (no held-out OOF)
}

# Colour palette for line / bar charts.
_PALETTE = [
    "#2ecc71", "#3498db", "#9b59b6", "#e74c3c",
    "#f39c12", "#1abc9c", "#e67e22", "#2c3e50",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def _load(fname: str):
    """Load a CSV from the submissions directory. Returns None if missing."""
    path = os.path.join(config.SUBMISSIONS_DIR, fname)
    return pd.read_csv(path) if os.path.exists(path) else None


def _oof_to_array(df: pd.DataFrame,
                  train_genes: list,
                  gene_order: list) -> np.ndarray:
    """
    Convert an OOF dataframe (columns: pert_id, gene1, gene2, …) to a
    (n_train, n_genes) float32 array aligned to train_genes / gene_order.
    Missing rows are filled with NaN.
    """
    gene_cols = [g for g in gene_order if g in df.columns]
    row_map   = {g: i for i, g in enumerate(df["pert_id"].tolist())}
    out       = np.full((len(train_genes), len(gene_order)), np.nan, dtype=np.float32)
    for i, g in enumerate(train_genes):
        if g in row_map:
            out[i] = df.iloc[row_map[g]][gene_cols].values.astype(np.float32)
    return out


def _sub_to_array(df: pd.DataFrame, gene_order: list) -> np.ndarray:
    """Convert submission dataframe to (n_test, n_genes) float32 array."""
    gene_cols = [g for g in gene_order if g in df.columns]
    return df[gene_cols].values.astype(np.float32)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare prediction methods")
    parser.add_argument("--show",   action="store_true",
                        help="Display the figure after saving")
    parser.add_argument("--output", default="method_comparison.png",
                        help="Output filename (saved in outputs/figures/)")
    args = parser.parse_args()

    config.make_dirs()
    de_matrix, weight_matrix, gene_order, train_genes = load_ground_truth()

    print("=" * 60)
    print(" Loading OOF predictions …")
    print("=" * 60)

    # ── 1. Compute OOF / LOO WMAE ─────────────────────────────────────
    oof_wmae     = {}   # label → float WMAE
    oof_per_pert = {}   # label → (n_train,) per-perturbation WMAE
    oof_arrays   = {}   # label → (n_train, n_genes) (used for per-pert plot)

    for fname, label in OOF_FILES:
        df = _load(fname)
        if df is None:
            print(f"  [{label}] OOF file not found: {fname}")
            continue
        arr  = _oof_to_array(df, train_genes, gene_order)
        mask = np.isfinite(arr).all(axis=1)
        if mask.sum() < 5:
            print(f"  [{label}] Too few valid rows in OOF file — skipping")
            continue
        wmae            = compute_wmae(arr[mask], de_matrix[mask], weight_matrix[mask])
        pp              = compute_wmae_per_pert(arr[mask], de_matrix[mask], weight_matrix[mask])
        oof_wmae[label]     = wmae
        oof_per_pert[label] = pp
        oof_arrays[label]   = arr[mask]
        delta_pct = (BASELINE_WMAE - wmae) / BASELINE_WMAE * 100
        print(f"  OOF WMAE  [{label:12s}]: {wmae:.4f}  ({delta_pct:+.1f}% vs baseline)")

    # Add pre-computed values (e.g. KNN LOO)
    for label, wmae in PRECOMPUTED_WMAE.items():
        if label not in oof_wmae:
            delta_pct = (BASELINE_WMAE - wmae) / BASELINE_WMAE * 100
            oof_wmae[label] = wmae
            print(f"  LOO  WMAE [{label:12s}]: {wmae:.4f}  ({delta_pct:+.1f}% vs baseline)  [pre-computed]")

    print()
    print("=" * 60)
    print(" Loading test submissions …")
    print("=" * 60)

    # ── 2. Load test submissions ───────────────────────────────────────
    sub_arrays = {}   # label → (n_test, n_genes)
    for fname, label in SUBMISSION_FILES:
        df = _load(fname)
        if df is None:
            continue
        sub_arrays[label] = _sub_to_array(df, gene_order)
        print(f"  [{label:14s}]  shape = {sub_arrays[label].shape}")

    sub_labels = list(sub_arrays.keys())
    n_subs     = len(sub_labels)

    # ── 3. Pairwise Pearson correlation ───────────────────────────────
    corr_mat = np.ones((n_subs, n_subs))
    for i, la in enumerate(sub_labels):
        for j, lb in enumerate(sub_labels):
            if i < j:
                r = float(np.corrcoef(sub_arrays[la].ravel(),
                                      sub_arrays[lb].ravel())[0, 1])
                corr_mat[i, j] = r
                corr_mat[j, i] = r

    # ── 4. Mean prediction magnitude ──────────────────────────────────
    sub_mag = {lb: float(np.abs(sub_arrays[lb]).mean()) for lb in sub_labels}

    # ── 5. Build figure ────────────────────────────────────────────────
    print()
    print("Building figure …")

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Perturbation Prediction — Method Comparison",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    ax_bar  = fig.add_subplot(gs[0, 0])     # OOF WMAE bar chart
    ax_heat = fig.add_subplot(gs[0, 1:])    # correlation heatmap (wide)
    ax_pp   = fig.add_subplot(gs[1, :2])    # per-pert WMAE line plot
    ax_mag  = fig.add_subplot(gs[1, 2])     # prediction magnitude

    # ── Panel 1: OOF WMAE bar chart ───────────────────────────────────
    if oof_wmae:
        items      = sorted(oof_wmae.items(), key=lambda x: x[1])
        lbl_bar    = [it[0] for it in items]
        val_bar    = [it[1] for it in items]
        bar_colors = ["#2ecc71" if v < BASELINE_WMAE else "#e74c3c" for v in val_bar]

        bars = ax_bar.barh(lbl_bar, val_bar, color=bar_colors,
                           edgecolor="white", height=0.55)
        ax_bar.axvline(BASELINE_WMAE, color="#555555", linestyle="--",
                       linewidth=1.3, label=f"Baseline={BASELINE_WMAE}")
        ax_bar.set_xlabel("WMAE  (↓ better)", fontsize=9)
        ax_bar.set_title("OOF / LOO Validation WMAE", fontsize=10, fontweight="bold")
        ax_bar.legend(fontsize=8)
        all_vals = val_bar + [BASELINE_WMAE]
        ax_bar.set_xlim(min(all_vals) * 0.983, max(all_vals) * 1.022)
        for bar, val in zip(bars, val_bar):
            ax_bar.text(val + 0.0003, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=8)
    else:
        ax_bar.text(0.5, 0.5,
                    "No OOF predictions found.\n\nRun train.py or train_v2.py first\nto generate oof_gnn.csv.",
                    ha="center", va="center", transform=ax_bar.transAxes,
                    fontsize=9, color="gray")
        ax_bar.set_title("OOF / LOO Validation WMAE", fontsize=10, fontweight="bold")

    # ── Panel 2: Pairwise correlation heatmap ─────────────────────────
    if n_subs >= 2:
        im = ax_heat.imshow(corr_mat, vmin=0.3, vmax=1.0,
                            cmap="YlOrRd", aspect="auto")
        ax_heat.set_xticks(range(n_subs))
        ax_heat.set_yticks(range(n_subs))
        ax_heat.set_xticklabels(sub_labels, rotation=38, ha="right", fontsize=8)
        ax_heat.set_yticklabels(sub_labels, fontsize=8)
        ax_heat.set_title("Test Submission Pairwise Pearson r",
                           fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax_heat, shrink=0.8)
        for i in range(n_subs):
            for j in range(n_subs):
                color = "white" if corr_mat[i, j] > 0.85 else "black"
                ax_heat.text(j, i, f"{corr_mat[i, j]:.2f}",
                             ha="center", va="center", fontsize=7, color=color)
    elif n_subs == 1:
        ax_heat.text(0.5, 0.5,
                     f"Only 1 submission found: {sub_labels[0]}\n\n"
                     "Run more methods to see pairwise correlation.",
                     ha="center", va="center", transform=ax_heat.transAxes,
                     fontsize=9, color="gray")
        ax_heat.set_title("Test Submission Pairwise Pearson r",
                           fontsize=10, fontweight="bold")
    else:
        ax_heat.text(0.5, 0.5,
                     "No submissions found.\n\nRun at least one method first.",
                     ha="center", va="center", transform=ax_heat.transAxes,
                     fontsize=9, color="gray")
        ax_heat.set_title("Test Submission Pairwise Pearson r",
                           fontsize=10, fontweight="bold")

    # ── Panel 3: Per-perturbation WMAE (OOF) ──────────────────────────
    if oof_per_pert:
        # Sort perturbations by mean |DE| across genes (easy → hard)
        sort_idx = np.argsort(np.abs(de_matrix).mean(axis=1))
        x        = np.arange(len(sort_idx))
        for ci, (label, pp) in enumerate(oof_per_pert.items()):
            ax_pp.plot(x, pp[sort_idx],
                       label=label,
                       color=_PALETTE[ci % len(_PALETTE)],
                       alpha=0.85, linewidth=1.1)
        ax_pp.axhline(BASELINE_WMAE, color="#888888", linestyle="--",
                      linewidth=0.9, alpha=0.7, label=f"Baseline ({BASELINE_WMAE})")
        ax_pp.set_xlabel("Training perturbation  (sorted by mean |DE|, easy → hard)",
                          fontsize=9)
        ax_pp.set_ylabel("WMAE", fontsize=9)
        ax_pp.set_title("Per-Perturbation WMAE — OOF",
                         fontsize=10, fontweight="bold")
        ax_pp.legend(fontsize=9)
        ax_pp.set_xlim(0, len(sort_idx) - 1)
    elif oof_wmae:
        ax_pp.text(0.5, 0.5,
                   "Pre-computed totals loaded — per-perturbation breakdown\n"
                   "not available.  Re-run to produce oof CSV files.",
                   ha="center", va="center", transform=ax_pp.transAxes,
                   fontsize=9, color="gray")
        ax_pp.set_title("Per-Perturbation WMAE — OOF",
                         fontsize=10, fontweight="bold")
    else:
        ax_pp.text(0.5, 0.5,
                   "No OOF predictions found.\n\nRun train.py or train_v2.py first.",
                   ha="center", va="center", transform=ax_pp.transAxes,
                   fontsize=9, color="gray")
        ax_pp.set_title("Per-Perturbation WMAE — OOF",
                         fontsize=10, fontweight="bold")

    # ── Panel 4: Mean prediction magnitude (sanity check) ─────────────
    if sub_mag:
        names_m = list(sub_mag.keys())
        vals_m  = [sub_mag[n] for n in names_m]
        bar_m = ax_mag.barh(
            names_m, vals_m,
            color=[_PALETTE[i % len(_PALETTE)] for i in range(len(names_m))],
            edgecolor="white", height=0.55,
        )
        for bar, val in zip(bar_m, vals_m):
            ax_mag.text(val + 1e-5, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=8)
        ax_mag.set_xlabel("Mean |prediction|", fontsize=9)
        ax_mag.set_title("Prediction Magnitude\n(sanity check — should be ~0.01–0.10)",
                          fontsize=10, fontweight="bold")
    else:
        ax_mag.text(0.5, 0.5, "No submissions found.",
                    ha="center", va="center", transform=ax_mag.transAxes,
                    fontsize=9, color="gray")
        ax_mag.set_title("Prediction Magnitude", fontsize=10, fontweight="bold")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = os.path.join(config.FIGURES_DIR, args.output)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # ── Print summary table ────────────────────────────────────────────
    if oof_wmae:
        print()
        print("=" * 50)
        print(f"{'Method':<18}  {'WMAE':>8}  {'vs baseline':>12}")
        print("-" * 50)
        for label, wmae in sorted(oof_wmae.items(), key=lambda x: x[1]):
            delta = (BASELINE_WMAE - wmae) / BASELINE_WMAE * 100
            flag  = "✓" if wmae < BASELINE_WMAE else "✗"
            print(f"  {label:<16}  {wmae:>8.4f}  {delta:>+10.1f}%  {flag}")
        print(f"  {'Baseline':<16}  {BASELINE_WMAE:>8.4f}  {'(reference)':>12}")
        print("=" * 50)


if __name__ == "__main__":
    main()
