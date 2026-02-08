# Getting Started

This guide walks you through installing TrustECG, running the dashboard, and making your first ECG prediction.

---

## Prerequisites

| Requirement    | Version                         |
| -------------- | ------------------------------- |
| Python         | 3.10 or higher                  |
| pip / uv       | Latest                          |
| Git            | Any recent version              |
| GPU (optional) | CUDA 11.8+ for faster inference |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/TrustECG.git
cd TrustECG
```

### 2. Create a Virtual Environment

**Using uv (recommended):**

```bash
uv venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate   # Linux / macOS
```

**Using standard venv:**

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate   # Linux / macOS
```

### 3. Install Dependencies

```bash
# With uv
uv sync

# Or with pip
pip install -e .
```

This installs all dependencies defined in `pyproject.toml`:

- **Core ML**: PyTorch, PyTorch Lightning, torchmetrics
- **Data**: NumPy, Pandas, SciPy, wfdb, scikit-learn
- **Explainability**: SHAP, LIME, Captum
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit

### 4. Verify Installation

```bash
python -c "import torch; import streamlit; print('Ready!')"
```

---

## Dataset

The PTB-XL dataset should be in the `dataset/` directory. If you cloned the repo with the dataset included, you're good to go. Otherwise:

```bash
# Download from PhysioNet (~2GB)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

**Required files:**

```
dataset/
├── ptbxl_database.csv      # Patient metadata and labels
├── scp_statements.csv      # SCP diagnostic code definitions
└── records100/             # 100Hz ECG recordings (21,801 files)
    ├── 00000/
    ├── 01000/
    └── ...
```

---

## Running the Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

Opens at **http://localhost:8501**

### First Steps

1. Click **"Load Model"** in the sidebar (loads `checkpoints/trustecg_model.pt`)
2. Navigate to **"Analyze ECG"** page
3. Enter an ECG ID (e.g., `9`) and click **"Load ECG"**
4. View the 12-lead ECG plot and prediction results
5. Switch to **"Explainability"** page for attention visualizations

### Useful ECG IDs to Try

| ECG ID | Expected Result        |
| ------ | ---------------------- |
| `9`    | Normal ECG             |
| `42`   | Mixed conditions       |
| `100`  | Multi-label example    |
| `500`  | Try different patterns |

---

## Making Predictions Programmatically

```python
import torch
import numpy as np
from src.app.streamlit_app import ExplainableECGNet, ECGPreprocessor

# 1. Load model
model = ExplainableECGNet()
state_dict = torch.load("checkpoints/trustecg_model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 2. Load ECG
import wfdb
record = wfdb.rdrecord("dataset/records100/00000/00009_lr")
signal = record.p_signal.T  # Shape: (12, 1000)

# 3. Preprocess
preprocessor = ECGPreprocessor()
signal = preprocessor(signal)  # Bandpass filter + z-normalization

# 4. Predict
x = torch.from_numpy(signal[np.newaxis, ...])  # Shape: (1, 12, 1000)
with torch.no_grad():
    output = model(x, return_attention=True)

# 5. Read results
probs = output["probs"].numpy().squeeze()
classes = ["NORM", "MI", "STTC", "CD", "HYP"]
for cls, prob in zip(classes, probs):
    print(f"{cls}: {prob:.1%}")

# 6. Access attention weights (for explainability)
lead_attention = output["lead_attention"]       # Shape: (1, 12)
temporal_attention = output["temporal_attention"] # Shape: (1, 12, 125)
```

---

## Model Checkpoints

| File                            | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| `checkpoints/trustecg_model.pt` | Trained model state dict (best validation AUROC) |
| `checkpoints/model_config.json` | Model hyperparameters and training metadata      |
| `notebooks/best_model.pt`       | Same weights saved from training notebook        |

---

## Troubleshooting

| Problem                     | Solution                                                                 |
| --------------------------- | ------------------------------------------------------------------------ |
| `ModuleNotFoundError: wfdb` | Run `pip install wfdb`                                                   |
| Model loads as "untrained"  | Check that `checkpoints/trustecg_model.pt` exists                        |
| ECG loading fails           | Verify `dataset/records100/` directory exists                            |
| CUDA out of memory          | Model is small (276K params), this is unlikely. Try `map_location="cpu"` |
| Streamlit not found         | Run `pip install streamlit`                                              |

---

**Next**: [Data Pipeline](data-pipeline.md) to understand how ECG data is loaded and preprocessed.
