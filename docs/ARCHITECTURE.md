# Model Architecture

Complete technical breakdown of ExplainableECGNet, the neural network at the heart of TrustECG.

---

## Overview

ExplainableECGNet is a CNN + Attention hybrid designed for multi-label ECG classification with built-in interpretability.

| Property          | Value                                           |
| ----------------- | ----------------------------------------------- |
| Parameters        | 276,421                                         |
| Input             | (batch, 12, 1000) — 12 leads × 1000 time steps  |
| Output            | 5 probabilities (one per diagnostic class)      |
| Attention outputs | Temporal weights (12, 125) + Lead weights (12,) |

<p align="center">
  <img src="../figures/High-level-architecture-diagram.png" alt="Architecture" width="800"/>
</p>

---

## Architecture Diagram

```
Input: (B, 12, 1000)
       │
       ▼ reshape to (B×12, 1, 1000) — process each lead independently
       │
┌──────┴──────┐
│ Lead-wise   │   ResidualBlock1D(1→32, stride=2)  → (B×12, 32, 500)
│ CNN Encoder │   ResidualBlock1D(32→64, stride=2) → (B×12, 64, 250)
│             │   ResidualBlock1D(64→128, stride=2)→ (B×12, 128, 125)
└──────┬──────┘
       │ reshape to (B, 12, 125, 128) then permute to (B, 12, 125, 128)
       │
       ▼ for each lead i ∈ [0..11]:
┌──────┴──────┐
│  Temporal   │   Input: (B, 125, 128)
│  Attention  │   → attention scores α ∈ (B, 125)
│             │   → weighted sum → lead_feature ∈ (B, 128)
└──────┬──────┘
       │ stack 12 lead features → (B, 12, 128)
       │
┌──────┴──────┐
│    Lead     │   Input: (B, 12, 128)
│  Attention  │   → attention scores β ∈ (B, 12)
│             │   → weighted sum → context ∈ (B, 128)
└──────┬──────┘
       │
┌──────┴──────┐
│ Classifier  │   Linear(128→128) → ReLU → Dropout
│   Head      │   Linear(128→64)  → ReLU → Dropout
│             │   Linear(64→5)    → Sigmoid
└──────┬──────┘
       │
       ▼
Output: probs (B, 5), temporal_attention (B, 12, 125), lead_attention (B, 12)
```

---

## Component Details

### 1. ResidualBlock1D

Each block contains two 1D convolutions with skip connections (inspired by ResNet):

```
Input ──→ Conv1d → BN → ReLU → Dropout → Conv1d → BN ──→ (+) → ReLU → Output
  │                                                         ↑
  └──────────────── Skip Connection (1×1 conv if needed) ───┘
```

**Parameters:**

| Property      | Value                         |
| ------------- | ----------------------------- |
| Kernel size   | 7                             |
| Stride        | 2 (halves temporal dimension) |
| Dropout       | 0.1 within blocks             |
| Activation    | ReLU                          |
| Normalization | BatchNorm1d                   |

**Skip connection**: Uses a 1×1 convolution + BatchNorm when input and output dimensions differ (channel count or temporal length).

```python
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
```

**Three blocks in sequence:**

| Block   | Input     | Output     | Temporal dim |
| ------- | --------- | ---------- | ------------ |
| Block 1 | (1, 1000) | (32, 500)  | 1000 → 500   |
| Block 2 | (32, 500) | (64, 250)  | 500 → 250    |
| Block 3 | (64, 250) | (128, 125) | 250 → 125    |

### 2. Temporal Attention

For each lead, temporal attention learns which of the 125 time steps are most important:

$$\alpha_t = \text{softmax}\left(w_2 \cdot \tanh(W_1 \cdot h_t + b_1) + b_2\right)$$

$$\text{context} = \sum_{t=1}^{125} \alpha_t \cdot h_t$$

```python
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):  # feature_dim = 128
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),  # 128 → 64
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),             # 64 → 1
        )

    def forward(self, x):  # x: (B, 125, 128)
        attn_weights = self.attention(x)        # (B, 125, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (x * attn_weights).sum(dim=1)  # (B, 128)
        return context, attn_weights.squeeze(-1)  # (B, 128), (B, 125)
```

**What it learns**: Focus on QRS complexes, ST segments, and T waves while ignoring baseline segments.

### 3. Lead Attention

After temporal attention produces one 128-dim feature per lead, lead attention learns which leads matter most:

$$\beta_l = \text{softmax}\left(w_2 \cdot \tanh(W_1 \cdot z_l + b_1) + b_2\right)$$

$$\text{context} = \sum_{l=1}^{12} \beta_l \cdot z_l$$

Same architecture as temporal attention but applied over the lead dimension:

```python
class LeadAttention(nn.Module):
    def __init__(self, feature_dim, num_leads=12):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),  # 128 → 64
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),             # 64 → 1
        )

    def forward(self, x):  # x: (B, 12, 128)
        attn_weights = self.attention(x)        # (B, 12, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (x * attn_weights).sum(dim=1)  # (B, 128)
        return context, attn_weights.squeeze(-1)  # (B, 128), (B, 12)
```

**What it learns**: For MI detection, focus on leads II, III, aVF (inferior), V1-V4 (anterior). For HYP, focus on chest leads V1-V6.

### 4. Classification Head

A three-layer MLP with sigmoid output for multi-label classification:

```python
self.classifier = nn.Sequential(
    nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 5),   # 5 classes
)
# Sigmoid applied separately: probs = torch.sigmoid(logits)
```

Sigmoid (not softmax) because classes are independent. A patient can have MI + STTC simultaneously.

---

## Forward Pass

```python
def forward(self, x, return_attention=False):
    batch_size = x.shape[0]

    # 1. Reshape: treat each lead as a separate sample
    x = x.view(batch_size * 12, 1, 1000)      # (B*12, 1, 1000)

    # 2. Encode all leads through shared CNN
    encoded = self.lead_encoder(x)              # (B*12, 128, 125)

    # 3. Reshape back to (B, 12, 128, 125) then permute
    encoded = encoded.view(batch_size, 12, 128, -1)
    encoded = encoded.permute(0, 1, 3, 2)      # (B, 12, 125, 128)

    # 4. Temporal attention per lead
    lead_features = []
    temporal_weights_all = []
    for i in range(12):
        lead_seq = encoded[:, i, :, :]          # (B, 125, 128)
        feat, weights = self.temporal_attention(lead_seq)
        lead_features.append(feat)               # (B, 128)
        temporal_weights_all.append(weights)     # (B, 125)

    lead_features = torch.stack(lead_features, dim=1)         # (B, 12, 128)
    temporal_weights_all = torch.stack(temporal_weights_all, dim=1)  # (B, 12, 125)

    # 5. Lead attention
    context, lead_weights = self.lead_attention(lead_features)  # (B, 128), (B, 12)

    # 6. Classify
    logits = self.classifier(context)           # (B, 5)
    probs = torch.sigmoid(logits)               # (B, 5)

    if return_attention:
        return {
            "logits": logits,
            "probs": probs,
            "temporal_attention": temporal_weights_all,   # (B, 12, 125)
            "lead_attention": lead_weights,               # (B, 12)
        }
    return {"logits": logits, "probs": probs}
```

---

## Design Decisions

| Decision                 | Rationale                                                                                                                                                      |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Lead-wise encoding**   | Each lead has the same structure (PQRST waves). Processing independently lets the model learn lead-specific features before comparing across leads.            |
| **Shared encoder**       | All 12 leads pass through the same CNN. Reduces parameters and enforces consistent feature extraction.                                                         |
| **Dual attention**       | Temporal attention finds important time points (QRS, ST segments). Lead attention finds important leads. Together they provide two levels of interpretability. |
| **Sigmoid over softmax** | Multi-label: conditions are independent. A patient can have MI + STTC at the same time.                                                                        |
| **Small model (276K)**   | Deliberately compact for fast inference and deployment on consumer hardware.                                                                                   |

---

## Tensor Shapes Reference

For a single sample flowing through the network:

| Stage                  | Shape             | Description                      |
| ---------------------- | ----------------- | -------------------------------- |
| Input                  | (1, 12, 1000)     | batch, leads, time               |
| After reshape          | (12, 1, 1000)     | leads as batch, channel, time    |
| After encoder          | (12, 128, 125)    | leads, features, compressed time |
| After reshape          | (1, 12, 125, 128) | batch, leads, time, features     |
| Per-lead temporal attn | (1, 128) per lead | weighted feature per lead        |
| Stacked lead features  | (1, 12, 128)      | all lead features                |
| After lead attn        | (1, 128)          | global context vector            |
| Classifier output      | (1, 5)            | class probabilities              |

---

**Next**: [Training Guide](training.md) to understand how the model is trained.
