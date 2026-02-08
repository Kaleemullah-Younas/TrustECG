# Explainability

How TrustECG provides interpretable predictions through attention mechanisms and occlusion analysis.

---

## Why Explainability Matters

In healthcare AI, a prediction alone isn't enough. Doctors need to understand *why* the model flagged something. TrustECG builds explainability directly into the architecture (not as a post-hoc add-on), using attention mechanisms that learn which parts of the ECG matter most.

---

## Explainability Methods

| Method | Level | What It Shows | Computation |
|--------|-------|---------------|-------------|
| **Lead Attention** | Per-lead | Which of 12 leads contributed most | Built-in (free) |
| **Temporal Attention** | Per-timestep | Which time segments matter per lead | Built-in (free) |
| **Occlusion Sensitivity** | Per-lead per-class | Impact of masking each lead | 12 forward passes |
| **Attention Heatmap** | Combined | 2D view of attention across leads × time | Built-in (free) |

---

## 1. Lead Attention (β weights)

The lead attention module assigns an importance weight $\beta_l$ to each of the 12 ECG leads:

$$\beta_l = \text{softmax}\left(\text{score}(z_l)\right) \quad \text{for } l \in \{I, II, III, aVR, aVL, aVF, V1, ..., V6\}$$

Weights sum to 1.0 across all leads. Higher weight = more influence on the prediction.

### Visualization: Radar Chart

<p align="center">
  <img src="../figures/08_lead_importance.png" alt="Lead Importance" width="500"/>
</p>

```python
# Access lead attention weights
output = model(x, return_attention=True)
lead_weights = output["lead_attention"]  # Shape: (batch, 12)

# Plot radar chart
fig = go.Figure(go.Scatterpolar(
    r=lead_weights.squeeze().tolist() + [lead_weights[0, 0].item()],
    theta=LEAD_NAMES + [LEAD_NAMES[0]],
    fill="toself",
))
```

### Clinical Validation

The learned attention patterns align with cardiology knowledge:

| Condition | Expected Important Leads | Why |
|-----------|-------------------------|-----|
| Inferior MI | II, III, aVF | These leads view the inferior wall |
| Anterior MI | V1-V4 | These leads view the anterior wall |
| Lateral MI | I, aVL, V5, V6 | These leads view the lateral wall |
| HYP | V1-V6 (chest leads) | Voltage criteria measured in chest leads |
| CD | II (rhythm lead) | Lead II best shows P waves and rhythm |

---

## 2. Temporal Attention (α weights)

For each lead, temporal attention assigns importance to each of the 125 encoded time steps (corresponding to the 1000-sample, 10-second ECG compressed through the CNN encoder):

$$\alpha_t = \text{softmax}\left(\text{score}(h_t)\right) \quad \text{for } t \in \{1, ..., 125\}$$

### What the Model Focuses On

In a typical ECG, the model learns to attend to:

| ECG Feature | Approximate Time | Clinical Significance |
|-------------|------------------|----------------------|
| **QRS complex** | ~0.06-0.12s per beat | Ventricular depolarization, conduction |
| **ST segment** | ~0.12-0.20s after QRS | Ischemia, MI detection |
| **T wave** | ~0.20-0.40s after QRS | Repolarization abnormalities |
| **P wave** | ~0.08-0.12s before QRS | Atrial activity, hypertrophy |

### Visualization: Heatmap

The temporal attention heatmap shows attention intensity across all leads and time:

```python
output = model(x, return_attention=True)
temporal_attn = output["temporal_attention"]  # Shape: (batch, 12, 125)

fig = go.Figure(go.Heatmap(
    z=temporal_attn.squeeze().numpy(),
    y=LEAD_NAMES,
    colorscale=[[0, "#1A1F26"], [0.5, "#E63946"], [1, "#FECACA"]],
))
```

Hot spots in the heatmap indicate where the model concentrated its attention. Consistent hot spots across leads suggest a global feature (like a wide QRS), while lead-specific hot spots suggest focal findings.

---

## 3. Occlusion Sensitivity

This method measures how much the prediction changes when each lead is masked (set to zero):

$$\text{Importance}_l = |P(\text{original}) - P(\text{occluded}_l)|$$

### Algorithm

```python
# Baseline prediction
baseline = model(signal)["probs"]

# Occlude each lead and measure change
importance = np.zeros((12, 5))
for i in range(12):
    occluded = signal.copy()
    occluded[i, :] = 0  # Zero out lead i
    pred = model(occluded)["probs"]
    importance[i] = np.abs(baseline - pred)
```

### Interpretation

| Importance Value | Meaning |
|-----------------|---------|
| 0.0 | Masking this lead had no effect (not important) |
| 0.01-0.05 | Minor contribution |
| 0.05-0.15 | Moderate contribution |
| > 0.15 | Critical lead for this prediction |

### Advantages Over Attention

Occlusion analysis provides a different perspective than attention weights:

- **Attention** shows what the model *looked at*
- **Occlusion** shows what the model *needs* (removing it changes the answer)

These often agree but can differ. If a lead has high attention but low occlusion importance, the model looked at it but could reach the same conclusion without it.

---

## 4. Combined Attention Heatmap

The full attention picture combines temporal and lead attention into one visualization:

```
Effective attention(lead l, time t) = β_l × α_{l,t}
```

This creates a 12 × 125 heatmap showing the overall importance of each (lead, time) combination.

---

## Using Explainability in the Dashboard

The **Explainability** page in the Streamlit dashboard provides all four visualizations:

1. **Load an ECG** on the "Analyze ECG" page
2. **Switch to "Explainability"** page
3. View:
   - Lead importance radar chart (instant)
   - Temporal attention heatmap (instant)
   - Prediction confidence bars (instant)
4. Click **"Run Analysis"** for occlusion sensitivity (takes a few seconds)

---

## Programmatic Access

```python
import torch
from src.app.streamlit_app import ExplainableECGNet, ECGPreprocessor

model = ExplainableECGNet()
model.load_state_dict(torch.load("checkpoints/trustecg_model.pt", map_location="cpu"))
model.eval()

# Get all attention weights in one forward pass
x = torch.randn(1, 12, 1000)  # Your preprocessed ECG
output = model(x, return_attention=True)

probs = output["probs"]                    # (1, 5) class probabilities
lead_attn = output["lead_attention"]        # (1, 12) lead importance
temporal_attn = output["temporal_attention"] # (1, 12, 125) per-lead temporal importance

# Most important lead
top_lead_idx = lead_attn.argmax(dim=1).item()
print(f"Most important lead: {LEAD_NAMES[top_lead_idx]}")

# Most attended time step for a specific lead
lead_idx = 1  # Lead II
top_time = temporal_attn[0, lead_idx].argmax().item()
print(f"Lead II: most attended time step = {top_time}/125")
```

---

**Next**: [Dashboard Guide](dashboard.md) to understand the Streamlit application.
