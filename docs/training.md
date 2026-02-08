# Training Guide

How TrustECG is trained, including the loss function, class imbalance handling, optimization, and evaluation.

---

## Training Configuration

All training is done in the notebook: [`notebooks/TrustECG_Notebook.ipynb`](../notebooks/TrustECG_Notebook.ipynb)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Batch size | 64 |
| Max epochs | 10 |
| Early stopping | Patience=5 on validation AUROC |
| Hardware | NVIDIA GeForce RTX 2050 |

---

## Loss Function

We use `BCEWithLogitsLoss` (binary cross-entropy with built-in sigmoid) for multi-label classification:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{5} \left[ w_c \cdot y_{ic} \log(\sigma(z_{ic})) + (1 - y_{ic}) \log(1 - \sigma(z_{ic})) \right]$$

Where:
- $y_{ic}$ is the ground truth (0 or 1) for sample $i$, class $c$
- $z_{ic}$ is the raw logit output
- $\sigma$ is the sigmoid function
- $w_c$ is the class-specific positive weight

---

## Handling Class Imbalance

The dataset is imbalanced (NORM is 43.3%, HYP is only 10.4%). We use `pos_weight` in the loss function to give more importance to rare classes.

### Why Square-Root Scaling?

We tried three approaches:

| Strategy | Formula | Result |
|----------|---------|--------|
| No weighting | $w_c = 1$ | Model biased toward NORM, poor HYP detection |
| Full inverse | $w_c = \frac{N_{\text{neg}}}{N_{\text{pos}}}$ | Overfitting to rare classes, unstable training |
| **Square-root** | $w_c = \sqrt{\frac{N_{\text{neg}}}{N_{\text{pos}}}}$ | Best balance across all classes |

### Computed Weights

```python
pos_counts = labels_train.sum(axis=0)      # Count of positives per class
neg_counts = len(labels_train) - pos_counts # Count of negatives per class
pos_weight = torch.sqrt(neg_counts / pos_counts)
```

| Class | Positive | Negative | pos_weight |
|-------|----------|----------|------------|
| NORM | 7,549 | 9,892 | 1.14 |
| MI | 3,307 | 14,134 | 2.07 |
| STTC | 4,071 | 13,370 | 1.81 |
| CD | 3,912 | 13,529 | 1.86 |
| HYP | 1,796 | 15,645 | 2.95 |

HYP gets ~3× the weight of NORM, balanced enough to improve detection without destabilizing training.

---

## Training Loop

The training loop is implemented in the notebook with early stopping:

```python
best_val_auroc = 0
patience_counter = 0

for epoch in range(num_epochs):
    # --- Train ---
    model.train()
    for batch in train_loader:
        signals, labels = batch
        optimizer.zero_grad()
        output = model(signals)
        loss = criterion(output["logits"], labels)
        loss.backward()
        optimizer.step()

    # --- Validate ---
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            signals, labels = batch
            output = model(signals)
            val_preds.append(torch.sigmoid(output["logits"]))
            val_labels.append(labels)

    val_auroc = compute_auroc(val_preds, val_labels)

    # --- Early Stopping ---
    if val_auroc > best_val_auroc:
        best_val_auroc = val_auroc
        torch.save(model.state_dict(), "best_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 5:
            break

    # --- LR Scheduling ---
    scheduler.step(val_auroc)
```

### Training Progress

<p align="center">
  <img src="../figures/05_training_curves.png" alt="Training Curves" width="700"/>
</p>

- Training converged within 10 epochs
- Best model saved at epoch 9
- No significant overfitting (train and val curves track closely)

---

## Evaluation Metrics

### Primary: AUROC (Macro)

Area Under the ROC Curve, averaged across all 5 classes. Threshold-independent metric that measures discrimination ability.

### Secondary: F1-Score (Macro)

Harmonic mean of precision and recall at threshold = 0.5.

### Per-Class Metrics

Computed using `sklearn.metrics.roc_auc_score` with `average=None` for per-class breakdown.

---

## Results

### Overall

| Metric | Validation | Test |
|--------|------------|------|
| **AUROC (Macro)** | 92.1% | 91.2% |
| **F1-Score (Macro)** | 69.4% | 69.4% |

### Per-Class Test Performance

| Class | AUROC | F1 |
|-------|-------|-----|
| NORM | 93.5% | 82.0% |
| MI | 93.9% | 70.6% |
| STTC | 93.5% | 66.6% |
| CD | 91.0% | 71.2% |
| HYP | 84.9% | 56.7% |

<p align="center">
  <img src="../figures/04_auroc_per_class.png" alt="AUROC Per Class" width="600"/>
</p>

### ROC Curves

<p align="center">
  <img src="../figures/06_roc_curves.png" alt="ROC Curves" width="600"/>
</p>

### Confusion Matrices

<p align="center">
  <img src="../figures/07_confusion_matrices.png" alt="Confusion Matrices" width="700"/>
</p>

---

## Observations

**Strong performers (>93% AUROC):**
- NORM, MI, STTC: These conditions have clear ECG patterns and sufficient training data

**Weaker performer:**
- HYP (84.9%): Rarest class (10.4%), subtler ECG patterns. Could benefit from per-class threshold optimization or additional features

**F1 gap from AUROC:**
- F1 scores are lower because we use a fixed threshold of 0.5. Per-class threshold optimization on the validation set could significantly improve F1

---

## Reproducing Training

1. Open `notebooks/TrustECG_Notebook.ipynb`
2. Run all cells in order
3. Model is saved to `notebooks/best_model.pt`
4. Copy to `checkpoints/trustecg_model.pt` for the dashboard

Alternatively, the model configuration is stored in `checkpoints/model_config.json`:

```json
{
  "num_leads": 12,
  "sequence_length": 1000,
  "num_classes": 5,
  "encoder_channels": [32, 64, 128],
  "dropout": 0.3,
  "best_val_auroc": 0.9213
}
```

---

**Next**: [Explainability](explainability.md) to understand how attention weights create interpretable predictions.
