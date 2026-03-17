"""
One-command pipeline runner for all NEW advanced methods.

This script runs each step in order, skipping steps whose output files
already exist. Original files (knn_baseline.py, train.py, predict.py,
ensemble.py) are never touched.

New submissions generated
-------------------------
    submission_ridge.csv           — Ridge regression on gene embeddings
    submission_ridge_pairwise.csv  — Ridge with pert×target interaction features
    submission_gnn_v2.csv          — GNN v2 (gene programs + MCDropout TTA)
    submission_cpa.csv             — Compositional Perturbation Autoencoder
    submission_ensemble_v2.csv     — Ensemble of all above

OOF files generated
-------------------
    outputs/submissions/oof_gnn_v2.csv

Figures generated
-----------------
    outputs/figures/training_curves_v2.png
    outputs/figures/method_comparison.png

Usage
-----
    # Full pipeline (takes several hours for GNN v2 on CPU)
    python src/run_pipeline.py

    # Quick run: Ridge + CPA only, skip the slow GNN v2 training
    python src/run_pipeline.py --skip-gnn

    # Skip CPA as well (Ridge only — ~3 minutes total)
    python src/run_pipeline.py --skip-gnn --skip-cpa

    # Re-run everything even if output files already exist
    python src/run_pipeline.py --force

    # Show the comparison figure at the end
    python src/run_pipeline.py --show-figure
"""
from __future__ import annotations
import os
import sys
import argparse
import subprocess

# Resolve src/ directory and project root
_SRC  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)

sys.path.insert(0, _SRC)
import config

PYTHON = sys.executable   # same interpreter that called this script


# ── Helpers ───────────────────────────────────────────────────────────────

def _sub_exists(fname: str) -> bool:
    return os.path.exists(os.path.join(config.SUBMISSIONS_DIR, fname))


def _ckpt_exists(fname: str) -> bool:
    return os.path.exists(os.path.join(config.CHECKPOINTS_DIR, fname))


def _run(cmd: list, desc: str):
    print()
    print("=" * 62)
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print("=" * 62)
    result = subprocess.run(cmd, cwd=_ROOT)
    if result.returncode != 0:
        print(f"\n  [ERROR] Step failed (exit code {result.returncode}).")
        print("  If this is a known issue, re-run with --skip flags to continue.")
        sys.exit(result.returncode)


def _skip(label: str, reason: str = "output already exists"):
    print(f"\n  [SKIP]  {label}  ({reason})")


# ── Pipeline steps ─────────────────────────────────────────────────────────

def step_ridge(force: bool):
    if not force and _sub_exists("submission_ridge.csv"):
        _skip("Step 1 – Ridge baseline")
        return
    _run(
        [PYTHON, "src/ridge_baseline.py"],
        "Step 1 – Ridge baseline  (~2-3 min)",
    )


def step_ridge_pairwise(force: bool):
    if not force and _sub_exists("submission_ridge_pairwise.csv"):
        _skip("Step 2 – Ridge pairwise")
        return
    _run(
        [PYTHON, "src/ridge_baseline.py", "--pairwise"],
        "Step 2 – Ridge pairwise features  (~5-10 min)",
    )


def step_train_gnn_v2(force: bool):
    all_ckpts = all(_ckpt_exists(f"v2_fold{k}_best.pt") for k in range(1, 6))
    if not force and all_ckpts:
        _skip("Step 3 – Train GNN v2")
        return
    _run(
        [PYTHON, "src/train_v2.py"],
        "Step 3 – Train GNN v2 (5-fold, 500 epochs each)  ← SLOW on CPU (~2-6 hrs)",
    )


def step_predict_gnn_v2(force: bool):
    if not force and _sub_exists("submission_gnn_v2.csv"):
        _skip("Step 4 – GNN v2 predictions")
        return
    _run(
        [PYTHON, "src/predict_v2.py"],
        "Step 4 – GNN v2 test predictions  (~2-5 min)",
    )


def step_cpa(force: bool):
    if not force and _sub_exists("submission_cpa.csv"):
        _skip("Step 5 – CPA model")
        return
    _run(
        [PYTHON, "src/cpa_model.py"],
        "Step 5 – CPA: train (200 epochs) + predict  (~10-30 min on CPU)",
    )


def step_ensemble_v2(force: bool):
    if not force and _sub_exists("submission_ensemble_v2.csv"):
        _skip("Step 6 – Ensemble v2")
        return
    _run(
        [PYTHON, "src/ensemble_v2.py"],
        "Step 6 – Ensemble v2  (~1 min)",
    )


def step_compare(show_figure: bool, force: bool):
    cmd = [PYTHON, "src/compare_methods.py"]
    if show_figure:
        cmd.append("--show")
    _run(cmd, "Step 7 – Compare all methods (generates method_comparison.png)")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the full advanced prediction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--skip-gnn",    action="store_true",
                        help="Skip GNN v2 training/prediction (very slow on CPU)")
    parser.add_argument("--skip-cpa",    action="store_true",
                        help="Skip CPA training")
    parser.add_argument("--skip-pairwise", action="store_true",
                        help="Skip pairwise Ridge (faster but less accurate)")
    parser.add_argument("--force",       action="store_true",
                        help="Re-run all steps even if output files exist")
    parser.add_argument("--show-figure", action="store_true",
                        help="Display the comparison figure at the end")
    args = parser.parse_args()

    config.make_dirs()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║  Advanced Perturbation Pipeline  (new methods)  ║")
    print("╚══════════════════════════════════════════════════╝")
    if args.skip_gnn:
        print("  GNN v2 skipped (--skip-gnn).")
    if args.skip_cpa:
        print("  CPA skipped (--skip-cpa).")
    if args.skip_pairwise:
        print("  Pairwise Ridge skipped (--skip-pairwise).")

    # ── Run steps ────────────────────────────────────────────────────
    step_ridge(args.force)

    if not args.skip_pairwise:
        step_ridge_pairwise(args.force)

    if not args.skip_gnn:
        step_train_gnn_v2(args.force)
        step_predict_gnn_v2(args.force)

    if not args.skip_cpa:
        step_cpa(args.force)

    step_ensemble_v2(args.force)
    step_compare(args.show_figure, args.force)

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════╗")
    print("║  Pipeline complete!                      ║")
    print(f"║  Submissions → outputs/submissions/      ║")
    print(f"║  Figure      → outputs/figures/          ║")
    print("╚══════════════════════════════════════════╝")
    print()
    print("  Next steps:")
    print("  1. Open outputs/figures/method_comparison.png")
    print("  2. Submit the best file to Kaggle:")

    candidates = [
        ("submission_ensemble_v2.csv", "Ensemble v2  (recommended first)"),
        ("submission_gnn_v2.csv",      "GNN v2"),
        ("submission_ridge.csv",       "Ridge"),
        ("submission_cpa.csv",         "CPA"),
    ]
    for fname, desc in candidates:
        path = os.path.join(config.SUBMISSIONS_DIR, fname)
        tag  = "✓ ready" if os.path.exists(path) else "✗ not generated"
        print(f"     {tag}  {fname}  — {desc}")


if __name__ == "__main__":
    main()
