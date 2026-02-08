# API Reference

Complete reference for all classes and functions in TrustECG.

All code lives in `src/app/streamlit_app.py`.

---

## Model Classes

### `ExplainableECGNet`

The main neural network for multi-label ECG classification.

```python
class ExplainableECGNet(nn.Module)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_leads` | int | 12 | Number of ECG leads |
| `sequence_length` | int | 1000 | Number of time samples per lead |
| `num_classes` | int | 5 | Number of output classes |
| `encoder_channels` | list[int] | [32, 64, 128] | Channel sizes for ResBlock layers |
| `dropout` | float | 0.3 | Dropout rate |

**Methods:**

#### `forward(x, return_attention=False)`

Run inference on ECG input.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | Tensor (B, 12, 1000) | Preprocessed ECG signal |
| `return_attention` | bool | If True, include attention weights in output |

**Returns** `dict`:

| Key | Shape | Always | Description |
|-----|-------|--------|-------------|
| `logits` | (B, 5) | Yes | Raw logit scores |
| `probs` | (B, 5) | Yes | Sigmoid probabilities |
| `temporal_attention` | (B, 12, 125) | If `return_attention=True` | Per-lead temporal weights |
| `lead_attention` | (B, 12) | If `return_attention=True` | Per-lead importance weights |

**Example:**

```python
model = ExplainableECGNet()
model.load_state_dict(torch.load("checkpoints/trustecg_model.pt", map_location="cpu"))
model.eval()

x = torch.randn(1, 12, 1000)
output = model(x, return_attention=True)

probs = output["probs"]                      # (1, 5)
lead_weights = output["lead_attention"]        # (1, 12)
temporal_weights = output["temporal_attention"] # (1, 12, 125)
```

---

### `ResidualBlock1D`

1D residual convolutional block with skip connections.

```python
class ResidualBlock1D(nn.Module)
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | int | — | Input channel count |
| `out_channels` | int | — | Output channel count |
| `kernel_size` | int | 7 | Convolution kernel size |
| `stride` | int | 1 | Stride (2 = halves temporal dim) |
| `dropout` | float | 0.1 | Dropout between convolutions |

**Forward:**

| Input | Output |
|-------|--------|
| (B, in_channels, T) | (B, out_channels, T // stride) |

---

### `TemporalAttention`

Learns importance weights across the time dimension.

```python
class TemporalAttention(nn.Module)
```

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `feature_dim` | int | Feature dimension (128) |

**Forward:**

| Input | Output |
|-------|--------|
| (B, T, D) | context (B, D), weights (B, T) |

---

### `LeadAttention`

Learns importance weights across the 12 ECG leads.

```python
class LeadAttention(nn.Module)
```

**Constructor Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `feature_dim` | int | Feature dimension (128) |
| `num_leads` | int | Number of leads (12) |

**Forward:**

| Input | Output |
|-------|--------|
| (B, L, D) | context (B, D), weights (B, L) |

---

## Preprocessing

### `ECGPreprocessor`

Applies bandpass filtering and z-score normalization to raw ECG signals.

```python
class ECGPreprocessor
```

**Constructor Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lowcut` | float | 0.5 | Low cutoff frequency (Hz) |
| `highcut` | float | 40.0 | High cutoff frequency (Hz) |
| `fs` | int | 100 | Sampling rate (Hz) |

**`__call__(signal)`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `signal` | ndarray (12, 1000) | Raw ECG signal (leads × time) |

**Returns**: `ndarray (12, 1000)` float32, preprocessed signal.

**Example:**

```python
preprocessor = ECGPreprocessor()
raw_signal = record.p_signal.T     # (12, 1000)
clean_signal = preprocessor(raw_signal)  # (12, 1000) float32
```

---

## Data Loading Functions

### `load_ecg_by_id(ecg_id)`

Load and preprocess an ECG record from the PTB-XL dataset.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ecg_id` | int | PTB-XL record ID (1-21800) |

**Returns**: `ndarray (12, 1000)` preprocessed ECG signal.

Falls back to `generate_demo_ecg()` if loading fails.

---

### `generate_demo_ecg()`

Generate a synthetic 12-lead ECG signal for testing.

**Returns**: `ndarray (12, 1000)` float32.

Uses `np.random.seed(42)` for reproducibility. Creates Gaussian peaks simulating QRS complexes at regular intervals.

---

### `run_prediction(signal)`

Run model inference on a preprocessed ECG signal.

| Parameter | Type | Description |
|-----------|------|-------------|
| `signal` | ndarray (12, 1000) | Preprocessed ECG |

**Returns**: `tuple(ndarray, dict)` — probabilities (5,) and attention dict with keys `"temporal"` and `"lead"`.

---

## Visualization Functions

### `plot_ecg(signal, title, highlight_leads)`

Create a 12-lead ECG plot using Plotly.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | ndarray (12, 1000) | — | ECG signal |
| `title` | str | "12-Lead ECG" | Plot title |
| `highlight_leads` | list[str] | None | Lead names to highlight in red |

**Returns**: `plotly.graph_objects.Figure`

---

### `plot_prediction_bars(probs)`

Create a horizontal bar chart of prediction confidences.

| Parameter | Type | Description |
|-----------|------|-------------|
| `probs` | ndarray (5,) | Class probabilities |

**Returns**: `plotly.graph_objects.Figure` with 0.5 threshold line.

---

### `plot_lead_radar(attention_dict)`

Create a polar radar chart of lead importance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `attention_dict` | dict | Dict with "lead" key containing attention tensor |

**Returns**: `plotly.graph_objects.Figure`

---

### `plot_attention_heatmap(attention_dict)`

Create a temporal attention heatmap across all leads.

| Parameter | Type | Description |
|-----------|------|-------------|
| `attention_dict` | dict | Dict with "temporal" key containing attention tensor |

**Returns**: `plotly.graph_objects.Figure` (12 × 125 heatmap)

---

## Page Rendering Functions

### `render_sidebar()`

Render the navigation sidebar with model status and dataset info.

**Returns**: `str` — selected page name (e.g., "🏠 Dashboard").

---

### `render_dashboard()`

Render the home page with hero header, metrics, features, and class info.

---

### `render_analyze()`

Render the ECG analysis page with input controls and prediction results.

---

### `render_explainability()`

Render the explainability page with attention visualizations and occlusion analysis.

---

### `main()`

Application entry point. Initializes session state, renders sidebar, and routes to the selected page.

---

## Constants

### `CLASS_NAMES`

```python
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
```

### `LEAD_NAMES`

```python
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
```

### `CLASS_INFO`

Dictionary mapping each class to display metadata:

| Key | Fields |
|-----|--------|
| `"NORM"` | icon: ✅, color: #10B981, name: "Normal" |
| `"MI"` | icon: 💔, color: #EF4444, name: "Myocardial Infarction" |
| `"STTC"` | icon: 📉, color: #F59E0B, name: "ST/T Change" |
| `"CD"` | icon: ⚡, color: #8B5CF6, name: "Conduction Disturbance" |
| `"HYP"` | icon: 💪, color: #06B6D4, name: "Hypertrophy" |

---

## File Map

| File | What It Contains |
|------|-----------------|
| `src/app/streamlit_app.py` | All classes and functions listed above |
| `notebooks/TrustECG_Notebook.ipynb` | Training pipeline (not importable as module) |
| `checkpoints/trustecg_model.pt` | Trained model weights |
| `checkpoints/model_config.json` | Model hyperparameters |
| `.streamlit/config.toml` | Dashboard theme configuration |
