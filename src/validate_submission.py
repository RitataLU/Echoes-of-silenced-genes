"""
Pre-submission validation script. Checks model performance locally before Kaggle upload.

Usage:
    # Validate a specific submission
    python src/validate_submission.py --submission submission_gnn.csv
    python src/validate_submission.py --submission submission_ensemble.csv

    # Validate all available submissions
    python src/validate_submission.py --all

    # Compare multiple submissions
    python src/validate_submission.py --compare gnn knn ensemble
"""
from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

sys.path.insert(0, os.path.dirname(__file__))
import config
from data_utils import (
    load_ground_truth, load_pert_ids, compute_wmae, compute_wmae_per_pert
)


def validate_submission_format(sub_df: pd.DataFrame, name: str = "submission") -> bool:
    """
    Check that submission has correct format.

    Returns True if valid, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Submission format check: {name}")
    print(f"{'='*60}")

    # Check shape
    expected_n_perts = 120
    expected_n_genes = 5127
    if sub_df.shape != (expected_n_perts, expected_n_genes + 1):
        print(f"  ✗ Shape mismatch: {sub_df.shape} (expected {expected_n_perts} × {expected_n_genes + 1})")
        return False
    print(f"  ✓ Shape: {sub_df.shape}")

    # Check first column is pert_id
    if sub_df.columns[0] != "pert_id":
        print(f"  ✗ First column is '{sub_df.columns[0]}' (expected 'pert_id')")
        return False
    print(f"  ✓ First column: 'pert_id'")

    # Check for NaN/Inf
    pred_cols = sub_df.iloc[:, 1:]
    nan_count = pred_cols.isna().sum().sum()
    inf_count = np.isinf(pred_cols.values).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  ✗ Found {nan_count} NaN and {inf_count} Inf values")
        return False
    print(f"  ✓ No NaN or Inf values")

    # Check value range (sanity check)
    pred_vals = pred_cols.values
    print(f"  Prediction range: [{pred_vals.min():.4f}, {pred_vals.max():.4f}]")
    print(f"  Mean |pred|: {np.abs(pred_vals).mean():.4f}")

    return True


def compare_to_baseline(sub_df: pd.DataFrame, name: str = "submission") -> Dict[str, float]:
    """
    Compare submission to known baselines.

    Returns dict of baseline comparisons.
    """
    print(f"\n{'='*60}")
    print(f"Baseline comparison: {name}")
    print(f"{'='*60}")

    pred_cols = sub_df.iloc[:, 1:].values.astype(np.float32)

    # Mean prediction magnitude
    mean_abs_pred = np.abs(pred_cols).mean()
    print(f"  Mean |pred|: {mean_abs_pred:.4f}")

    # Standard baselines for reference
    print(f"\n  Reference baselines (from README):")
    print(f"    Mean-DE baseline WMAE: 0.1268")
    print(f"    KNN expected WMAE: 0.110–0.115")
    print(f"    GNN expected WMAE: 0.085–0.095")
    print(f"    Ensemble expected WMAE: 0.075–0.085")

    return {"mean_abs_pred": mean_abs_pred}


def evaluate_oof_performance() -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Load OOF predictions and compute cross-validated WMAE.

    Returns
    -------
    oof_wmae : overall OOF WMAE
    oof_preds: (80, 5127) OOF predictions
    per_pert : (80,) per-perturbation WMAE
    """
    print(f"\n{'='*60}")
    print(f"Out-of-Fold (OOF) Performance Evaluation")
    print(f"{'='*60}")

    oof_path = os.path.join(config.SUBMISSIONS_DIR, "oof_gnn.csv")
    if not os.path.exists(oof_path):
        print(f"  ✗ OOF file not found: {oof_path}")
        print(f"     Run: python src/train.py")
        return None, None, None

    # Load OOF predictions
    oof_df = pd.read_csv(oof_path)
    oof_preds = oof_df.iloc[:, 1:].values.astype(np.float32)

    # Load ground truth
    de_matrix, weight_matrix, _, pert_order = load_ground_truth()

    # Compute WMAE
    oof_wmae = compute_wmae(oof_preds, de_matrix, weight_matrix)
    per_pert_wmae = compute_wmae_per_pert(oof_preds, de_matrix, weight_matrix)

    print(f"  OOF WMAE: {oof_wmae:.4f}")
    print(f"  Improvement over baseline (0.1268): {(0.1268 - oof_wmae) / 0.1268 * 100:+.1f}%")
    print(f"\n  Per-perturbation WMAE:")
    print(f"    Mean: {per_pert_wmae.mean():.4f}")
    print(f"    Std:  {per_pert_wmae.std():.4f}")
    print(f"    Min:  {per_pert_wmae.min():.4f} (best pert: {pert_order[per_pert_wmae.argmin()]})")
    print(f"    Max:  {per_pert_wmae.max():.4f} (worst pert: {pert_order[per_pert_wmae.argmax()]})")

    # Show hardest and easiest perturbations
    worst_indices = np.argsort(per_pert_wmae)[-5:]
    best_indices  = np.argsort(per_pert_wmae)[:5]

    print(f"\n  Top 5 hardest perturbations:")
    for idx in worst_indices[::-1]:
        print(f"    {pert_order[idx]}: {per_pert_wmae[idx]:.4f}")

    print(f"\n  Top 5 easiest perturbations:")
    for idx in best_indices:
        print(f"    {pert_order[idx]}: {per_pert_wmae[idx]:.4f}")

    return oof_wmae, oof_preds, per_pert_wmae


def validate_submission(filename: str) -> Tuple[bool, float]:
    """
    Full validation of a submission file.

    Returns (is_valid, wmae)
    """
    sub_path = os.path.join(config.SUBMISSIONS_DIR, filename)

    if not os.path.exists(sub_path):
        print(f"\n✗ Submission file not found: {sub_path}")
        return False, None

    # Load submission
    sub_df = pd.read_csv(sub_path)

    # Format check
    if not validate_submission_format(sub_df, filename):
        return False, None

    # Baseline comparison
    compare_to_baseline(sub_df, filename)

    print(f"\n{'='*60}")
    print(f"✓ Submission '{filename}' is valid and ready to upload")
    print(f"{'='*60}")

    return True, None


def compare_submissions(filenames: List[str]):
    """Compare multiple submission files."""
    print(f"\n{'='*60}")
    print(f"Submission Comparison")
    print(f"{'='*60}")

    submissions = {}
    for fname in filenames:
        path = os.path.join(config.SUBMISSIONS_DIR, fname)
        if os.path.exists(path):
            submissions[fname] = pd.read_csv(path)
            print(f"  ✓ Loaded: {fname}")
        else:
            print(f"  ✗ Not found: {fname}")

    if len(submissions) < 2:
        print("  Not enough submissions to compare")
        return

    # Pairwise differences
    print(f"\n  Pairwise correlation of predictions:")
    fnames = list(submissions.keys())
    for i, fname1 in enumerate(fnames):
        for fname2 in fnames[i+1:]:
            pred1 = submissions[fname1].iloc[:, 1:].values
            pred2 = submissions[fname2].iloc[:, 1:].values
            correlation = np.corrcoef(pred1.flatten(), pred2.flatten())[0, 1]
            mae_diff = np.abs(pred1 - pred2).mean()
            print(f"    {fname1} vs {fname2}:")
            print(f"      Correlation: {correlation:.4f}")
            print(f"      Mean abs diff: {mae_diff:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate submission(s) before Kaggle upload"
    )
    parser.add_argument(
        "--submission", type=str, default=None,
        help="Specific submission file to validate (e.g., submission_gnn.csv)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Validate all available submissions"
    )
    parser.add_argument(
        "--compare", nargs="+", default=None,
        help="Compare multiple submissions (e.g., --compare gnn knn ensemble)"
    )
    parser.add_argument(
        "--oof", action="store_true",
        help="Evaluate OOF (cross-validation) performance"
    )
    args = parser.parse_args()

    config.make_dirs()

    # Default: validate OOF + specific submission
    if not args.all and not args.compare and not args.submission and not args.oof:
        # Default workflow
        args.oof = True
        args.submission = "submission_gnn.csv"

    # Evaluate OOF
    if args.oof:
        evaluate_oof_performance()

    # Validate specific submission
    if args.submission:
        # Handle shorthand names
        if not args.submission.endswith(".csv"):
            args.submission = f"submission_{args.submission}.csv"
        validate_submission(args.submission)

    # Validate all submissions
    if args.all:
        available = [f for f in os.listdir(config.SUBMISSIONS_DIR) 
                     if f.startswith("submission_") and f.endswith(".csv")]
        print(f"\n{'='*60}")
        print(f"Validating all {len(available)} submissions …")
        print(f"{'='*60}")
        for fname in sorted(available):
            validate_submission(fname)

    # Compare submissions
    if args.compare:
        # Convert shorthand to full names
        fnames = [f"{name}.csv" if not name.endswith(".csv") else name 
                  for name in args.compare]
        fnames = [f if f.startswith("submission_") else f"submission_{f}"
                  for f in fnames]
        compare_submissions(fnames)

    print(f"\n{'='*60}")
    print(f"Validation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
