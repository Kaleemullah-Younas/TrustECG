# 🫀 TrustECG: Explainable Multi-Label ECG Classification

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.1+-792ee5.svg?style=for-the-badge&logo=lightning&logoColor=white)](https://lightning.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-ff4b4b.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**AI-powered cardiac diagnosis with attention-based explainability**

[Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Results](#-results) • [License](#-license)

</div>

---

## 📖 Overview

**TrustECG** is an explainable AI system for multi-label ECG classification, designed to detect 5 cardiac conditions from 12-lead electrocardiogram recordings. Built on the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) (21,801 clinical ECGs), it combines state-of-the-art deep learning with comprehensive explainability to build trust in AI-assisted cardiac diagnosis.

### Why TrustECG?

- **Clinical Relevance**: Trained on the largest publicly available ECG dataset with cardiologist-verified annotations
- **Explainable AI**: Understand _why_ the model makes predictions through attention visualization
- **Multi-Label**: Detect multiple conditions simultaneously (patients often have co-occurring cardiac issues)
- **Production-Ready**: Professional Streamlit dashboard for clinical demonstration

---

## ✨ Key Features

| Feature                               | Description                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| 🎯 **Multi-Label Classification**     | Simultaneously detect 5 cardiac conditions from a single ECG |
| 🔍 **Attention-Based Explainability** | Temporal and lead-wise attention visualization               |
| 📊 **Comprehensive XAI**              | SHAP, LIME, Grad-CAM, and occlusion sensitivity analysis     |
| 🖥️ **Professional Dashboard**         | Feature-rich Streamlit application with dark theme           |
| 📈 **High Performance**               | 92.1% Val AUROC, 91.2% Test AUROC                            |
| ⚡ **Fast Inference**                 | Real-time predictions on consumer hardware                   |

---

## 🏥 Diagnostic Classes

| Class       | Condition              | Description                              | Prevalence |
| ----------- | ---------------------- | ---------------------------------------- | ---------- |
| ✅ **NORM** | Normal                 | No significant cardiac abnormalities     | 43.3%      |
| 💔 **MI**   | Myocardial Infarction  | Signs of heart attack or ischemic damage | 19.0%      |
| 📉 **STTC** | ST/T Change            | Abnormalities in ST segment or T wave    | 23.3%      |
| ⚡ **CD**   | Conduction Disturbance | Abnormal electrical conduction patterns  | 22.5%      |
| 💪 **HYP**  | Hypertrophy            | Signs of enlarged heart chambers         | 10.4%      |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     12-Lead ECG Input                           │
│                  (12 leads × 1000 samples @ 100Hz)              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Preprocessing Pipeline                          │
│         Bandpass Filter (0.5-40 Hz) + Z-Normalization          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Lead-wise ResBlock1D Encoder                       │
│                 [32 → 64 → 128] channels                        │
│                    with skip connections                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Temporal Attention Block                        │
│            Learns which time points are important               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Lead Attention Block                          │
│          Learns which leads contribute most                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Label Classification Head                    │
│             [128 → 64 → 5] with Sigmoid output                  │
│            Outputs: NORM, MI, STTC, CD, HYP                     │
└─────────────────────────────────────────────────────────────────┘
```

**Model Statistics:**

- **Parameters**: 276,421 (1.1 MB)
- **Inference Time**: ~10ms on CPU, ~2ms on GPU
- **Input**: 12-lead ECG (12 × 1000 samples)
- **Output**: 5-class multi-label probabilities

---

## 📊 Results

### Performance Metrics

| Metric               | Validation | Test  |
| -------------------- | ---------- | ----- |
| **AUROC (Macro)**    | 92.1%      | 91.2% |
| **F1-Score (Macro)** | 69.4%      | 69.4% |

### Per-Class Test AUROC

| NORM  | MI    | STTC  | CD    | HYP   |
| ----- | ----- | ----- | ----- | ----- |
| 93.5% | 93.9% | 93.5% | 91.0% | 84.9% |

### Training Details

- **Epochs**: 10 (early stopping at epoch 9)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau
- **Loss**: BCEWithLogitsLoss with sqrt-scaled class weights
- **Hardware**: NVIDIA GeForce RTX 2050

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU acceleration)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/trustecg.git
cd trustecg

# Create virtual environment (using uv - recommended)
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Download PTB-XL Dataset

The PTB-XL dataset is already included in the `dataset/` directory. If you need to re-download:

```bash
# Download from PhysioNet (requires ~2GB)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

---

## 💻 Usage

### Run the Streamlit Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

The dashboard opens at `http://localhost:8501` with three pages:

| Page                  | Description                                         |
| --------------------- | --------------------------------------------------- |
| 🏠 **Dashboard**      | Overview, metrics, and feature highlights           |
| 🔮 **Analyze ECG**    | Load ECGs from PTB-XL dataset for prediction        |
| 🔍 **Explainability** | Attention maps, lead importance, occlusion analysis |

### Using the Analyze Page

1. **Load Model**: Click "🚀 Load Model" in the sidebar
2. **Enter ECG ID**: Input a number from 1-21800 (patient record ID)
3. **Load ECG**: Click "📥 Load ECG" to load and preprocess the signal
4. **View Results**: See predictions with confidence scores and explanations

**Example ECG IDs to try:**

- `9` - Normal ECG
- `42` - Various conditions
- `100` - Multi-label example

### Programmatic Usage

```python
import torch
import numpy as np
from src.app.streamlit_app import ExplainableECGNet, ECGPreprocessor

# Load model
model = ExplainableECGNet()
state_dict = torch.load("checkpoints/trustecg_model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Load and preprocess ECG
import wfdb
record = wfdb.rdrecord("dataset/records100/00000/00009_lr")
signal = record.p_signal.T  # (12, 1000)

preprocessor = ECGPreprocessor()
signal = preprocessor(signal)

# Predict
x = torch.from_numpy(signal[np.newaxis, ...])
with torch.no_grad():
    output = model(x, return_attention=True)
    probs = output["probs"].numpy().squeeze()

# Results
classes = ["NORM", "MI", "STTC", "CD", "HYP"]
for cls, prob in zip(classes, probs):
    print(f"{cls}: {prob:.1%}")
```

---

## 📁 Project Structure

```
TrustECG/
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration
├── src/
│   ├── app/
│   │   └── streamlit_app.py # Main Streamlit dashboard
│   ├── data/
│   │   ├── datamodule.py    # PyTorch Lightning DataModule
│   │   ├── dataset.py       # ECG Dataset class
│   │   └── preprocessing.py # Signal preprocessing
│   ├── models/
│   │   ├── ecg_net.py       # ExplainableECGNet architecture
│   │   ├── attention.py     # Attention mechanisms
│   │   └── blocks.py        # Residual blocks
│   ├── training/
│   │   ├── trainer.py       # Training script
│   │   ├── callbacks.py     # Custom callbacks
│   │   └── metrics.py       # Evaluation metrics
│   ├── explainability/
│   │   ├── attention_viz.py # Attention visualization
│   │   ├── gradcam.py       # Grad-CAM implementation
│   │   ├── lime_explainer.py
│   │   └── shap_explainer.py
│   └── feedback/
│       ├── feedback_store.py     # Human feedback storage
│       └── uncertainty_sampling.py
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory data analysis
│   └── 02_demo_colab.ipynb  # Google Colab demo
├── checkpoints/
│   └── trustecg_model.pt    # Trained model weights
├── dataset/
│   ├── ptbxl_database.csv   # Patient metadata
│   ├── scp_statements.csv   # Diagnostic codes
│   └── records100/          # 100Hz ECG recordings
├── configs/
│   └── default.yaml         # Training configuration
├── figures/                 # Generated visualizations
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

---

## 🧪 Explainability Methods

### 1. Temporal Attention

Shows which **time points** in the ECG are most important for classification.

### 2. Lead Attention

Visualizes which of the **12 ECG leads** contribute most to the prediction.

### 3. Occlusion Sensitivity

Measures prediction change when each lead is **masked to zero**, indicating its importance.

### 4. Attention Heatmap

Combined view of temporal attention across all leads for pattern identification.

---

## 🔧 Configuration

### Streamlit Theme (`.streamlit/config.toml`)

```toml
[theme]
primaryColor = "#E63946"           # Heart red accent
backgroundColor = "#0F1419"         # Dark background
secondaryBackgroundColor = "#1A1F26"
textColor = "#F8F9FA"
```

### Training Configuration (`configs/default.yaml`)

```yaml
model:
  encoder_channels: [32, 64, 128]
  dropout: 0.3

training:
  epochs: 20
  lr: 0.001
  batch_size: 64
  weight_decay: 0.01
```

---

## 📚 References

- **PTB-XL Dataset**: Wagner, P., et al. "PTB-XL, a large publicly available electrocardiography dataset." _Scientific Data_ 7.1 (2020): 1-15.
- **Attention Mechanisms**: Vaswani, A., et al. "Attention is all you need." _NeurIPS_ 2017.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [PTB-XL Dataset](https://physionet.org/content/ptb-xl/) by PhysioNet
- [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)
- [Streamlit](https://streamlit.io/) for the dashboard framework
- [Plotly](https://plotly.com/) for interactive visualizations

---

<div align="center">

**Made with ❤️ by the TrustECG Team**

</div>
