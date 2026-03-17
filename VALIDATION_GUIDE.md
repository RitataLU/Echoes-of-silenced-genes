"""
PRE-SUBMISSION PERFORMANCE VALIDATION GUIDE
============================================

Before uploading to Kaggle, use these methods to estimate your model's performance locally.
"""

## 1. OUT-OF-FOLD (OOF) CROSS-VALIDATION PERFORMANCE
═══════════════════════════════════════════════════════

**Why it matters**: This is your most reliable performance estimate.
- Trained on 80 perturbations with 5-fold CV
- Each fold trains on ~64 perturbations, validates on ~16
- OOF predictions let you compute WMAE against true labels locally

**Command**:
```bash
# Train models (does 5-fold CV automatically)
python src/train.py

# This creates oof_gnn.csv and prints OOF WMAE
# → Check the overall WMAE and improvement % vs baseline (0.1268)
```

**What to look for**:
- OOF WMAE < 0.1268 = better than mean baseline ✓
- OOF WMAE < 0.100 = competitive GNN
- OOF WMAE < 0.090 = very strong
- Improvement % should be positive

---

## 2. SUBMISSION FORMAT VALIDATION
═════════════════════════════════

**Why it matters**: Kaggle will reject malformed CSVs.

**Command**:
```bash
# Validate specific submission
python src/validate_submission.py --submission submission_gnn.csv

# Or shorthand
python src/validate_submission.py --submission gnn

# Validate all submissions
python src/validate_submission.py --all
```

**What gets checked**:
- Shape: exactly 120 × 5128 (120 perturbations, 5127 genes + pert_id column)
- No NaN or Inf values
- pert_id is first column
- Numeric ranges are reasonable

---

## 3. PER-PERTURBATION BREAKDOWN
═══════════════════════════════

**Why it matters**: Identifies weak spots in your model.

**Inside OOF evaluation**, you'll see:
```
Per-perturbation WMAE:
  Mean: 0.0845
  Std:  0.0123
  Min:  0.0650 (best pert: gene_X)
  Max:  0.1020 (worst pert: gene_Y)

Top 5 hardest perturbations:
  gene_Y: 0.1020
  gene_Z: 0.0980
  ...

Top 5 easiest perturbations:
  gene_X: 0.0650
  gene_A: 0.0670
  ...
```

**What to look for**:
- Std < 0.05 = stable across all perturbations ✓
- Large std = model struggles on some genes → try ensemble
- Look at hardest genes to see if they have features in common

---

## 4. BASELINE COMPARISONS
═════════════════════════

**Reference performance**:
```
Mean-DE baseline (predict mean vector):           WMAE = 0.1268
KNN (STRING similarity, k=5):                     WMAE ≈ 0.110–0.115
GNN (no GO features):                             WMAE ≈ 0.095–0.108
GNN + GO embeddings:                              WMAE ≈ 0.085–0.095
Ensemble (GNN + KNN):                             WMAE ≈ 0.075–0.085
```

**How to compare your KNN baseline**:
```bash
# Generate KNN submission (if not already done)
python src/graph_builder.py --download-string
python src/knn_baseline.py
# → outputs/submissions/submission_knn.csv

# Validate it
python src/validate_submission.py --submission knn
```

---

## 5. ENSEMBLE OPTIMIZATION
═══════════════════════════

**Why it matters**: Blending complementary models often improves performance.

**Commands**:
```bash
# Auto-blend with default alpha (0.7 GNN + 0.3 KNN)
python src/ensemble.py

# Test different blend weights
python src/ensemble.py --alpha 0.5
python src/ensemble.py --alpha 0.8
python src/ensemble.py --alpha 0.9

# Optimize alpha on OOF predictions (if available after training)
python src/ensemble.py --optimise-alpha
```

**Compare ensemble vs individual models**:
```bash
python src/validate_submission.py --compare gnn knn ensemble
```

Output shows correlation and differences between predictions.

---

## 6. SUBMISSION COMPARISON
═══════════════════════════

**If you have multiple candidates**, pick the best:
```bash
# Compare all submissions
python src/validate_submission.py --all

# Compare specific ones
python src/validate_submission.py --compare submission_gnn.csv submission_ensemble.csv submission_knn.csv
```

This shows:
- Pairwise correlations
- Mean absolute differences
- Which one to trust

---

## 7. COMPLETE PRE-SUBMISSION CHECKLIST
═══════════════════════════════════════

```
[ ] Run training: python src/train.py
    → Check OOF WMAE output
    → Verify checkpoints created in outputs/checkpoints/
    
[ ] Review OOF performance:
    → OOF WMAE < baseline (0.1268) ?
    → Per-perturbation spread reasonable ?
    
[ ] Generate predictions: python src/predict.py
    → Creates submission_gnn.csv
    
[ ] Optionally create KNN: python src/knn_baseline.py
    → Creates submission_knn.csv
    
[ ] Optionally blend models: python src/ensemble.py
    → Creates submission_ensemble.csv
    
[ ] Validate submission format:
    python src/validate_submission.py --submission gnn
    
[ ] Compare all submissions (if multiple):
    python src/validate_submission.py --compare gnn knn ensemble
    
[ ] Final check - submission stats:
    - Shape: 120 × 5128 ✓
    - No NaN/Inf ✓
    - First column is pert_id ✓
    - WMAE estimate < 0.100 ✓
    
[ ] Upload winning submission to Kaggle
```

---

## 8. QUICK WORKFLOW EXAMPLES
═════════════════════════════

### Just evaluate what you have (no retraining)
```bash
python src/validate_submission.py --all
```

### Build from scratch + validate
```bash
# Download external data
python src/graph_builder.py --download-string --download-go

# Build node features
python src/node_features.py

# Train GNN
python src/train.py

# Generate submission
python src/predict.py

# Generate KNN baseline
python src/knn_baseline.py

# Blend them
python src/ensemble.py --alpha 0.7

# Validate & compare
python src/validate_submission.py --all
python src/validate_submission.py --compare gnn knn ensemble
```

### Test different ensemble weights
```bash
# Create multiple blends
for alpha in 0.5 0.6 0.7 0.8 0.9; do
  python src/ensemble.py --alpha $alpha --output "submission_blend_${alpha}.csv"
done

# Compare them
python src/validate_submission.py --all
```

---

## Key Insights
═══════════════

1. **OOF is the ground truth for local evaluation**
   - Better than validating on just val split
   - Uses all 80 training examples

2. **Ensemble often helps**
   - KNN and GNN capture different patterns
   - Usually better than either alone
   - Test alpha values 0.5–0.9

3. **Stability matters**
   - If OOF WMAE is 0.085 ± high std, might overfit
   - Check per-perturbation spread

4. **Format validation is mandatory**
   - Kaggle API will reject wrong shapes
   - Always run --submission before uploading

5. **Don't overthink—trust OOF**
   - If OOF WMAE improves, test submission likely will too
   - Competition metric matches WMAE exactly

---

## Troubleshooting
═════════════════

**"No OOF file found"**
→ Run `python src/train.py` to create it

**"Submission validation failed: NaN values detected"**
→ Check that all upstream predictions computed correctly
→ Look for genes with missing features

**"Ensemble WMAE is worse than GNN alone"**
→ Try different alpha values (--alpha 0.6, --alpha 0.8, etc.)
→ KNN may not help for this configuration

**"Very high std in per-perturbation WMAE"**
→ Model struggles on hard genes
→ Try ensemble to diversify predictions
→ Consider re-training with more epochs

---
