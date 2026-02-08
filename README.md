<div align="center">

# TrustECG

### Explainable AI for 12-Lead ECG Classification

<br/>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.5+-792ee5.svg?style=for-the-badge&logo=lightning&logoColor=white)](https://lightning.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-ff4b4b.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

**Explainable AI for cardiac diagnosis you can trust**

[Quick Start](#-quick-start) · [Results](#-results) · [How It Works](#-how-it-works) · [Explore the Docs](#-documentation)

</div>

---

## What is TrustECG?

TrustECG is an **explainable** deep learning system that classifies 12-lead ECG recordings into 5 cardiac conditions — and shows doctors _exactly why_ it made each prediction through built-in attention visualization.

> Trained on [PTB-XL](https://physionet.org/content/ptb-xl/), the largest open clinical ECG dataset (21,801 recordings, cardiologist-verified).

<details>
<summary><b>Why does this matter?</b></summary>
<br/>

- **Trust**: Clinicians need to _understand_ AI decisions, not blindly follow them
- **Multi-label**: Patients often have multiple cardiac conditions — TrustECG detects them simultaneously
- **Speed**: Real-time predictions (~10ms/ECG) with comprehensive visual explanations
- **Ready to use**: Professional Streamlit dashboard with dark cardiac theme

</details>

---

## Conditions Detected

|          | Condition              | What It Means                      | Prevalence |
| -------- | ---------------------- | ---------------------------------- | ---------- |
| **NORM** | Normal                 | No significant abnormalities       | 43.3%      |
| **MI**   | Myocardial Infarction  | Heart attack / ischemic damage     | 19.0%      |
| **STTC** | ST/T Change            | ST segment or T wave abnormalities | 23.3%      |
| **CD**   | Conduction Disturbance | Abnormal electrical conduction     | 22.5%      |
| **HYP**  | Hypertrophy            | Enlarged heart chambers            | 10.4%      |

> **Deep dive →** [Data Pipeline docs](docs/data-pipeline.md) for dataset details, preprocessing math, and train/val/test splits.

---

## Results

<table>
<tr>
<td width="50%">

### Headline Numbers

| Metric       | Val   | Test  |
| ------------ | ----- | ----- |
| **AUROC**    | 92.1% | 91.2% |
| **F1-Score** | 69.4% | 69.4% |

**Per-Class AUROC**

| NORM  | MI    | STTC  | CD    | HYP   |
| ----- | ----- | ----- | ----- | ----- |
| 93.5% | 93.9% | 93.5% | 91.0% | 84.9% |

</td>
<td width="50%">

<img src="figures/04_auroc_per_class.png" alt="AUROC Per Class" width="100%"/>

</td>
</tr>
</table>

<details>
<summary><b>More figures: ROC curves, confusion matrices, training curves</b></summary>
<br/>

<p align="center">
  <img src="figures/06_roc_curves.png" alt="ROC Curves" width="700"/>
</p>

<p align="center">
  <img src="figures/07_confusion_matrices.png" alt="Confusion Matrices" width="700"/>
</p>

<p align="center">
  <img src="figures/05_training_curves.png" alt="Training Curves" width="700"/>
</p>

</details>

> **Deep dive →** [Training docs](docs/training.md) for loss function, class weighting strategy, and full evaluation details.

---

## How It Works

<p align="center">
  <img src="figures/High-level-architecture-diagram.png" alt="Architecture" width="750"/>
</p>

```
12-lead ECG → Bandpass Filter → Z-score Norm → ResBlocks [32→64→128] → Temporal Attention → Lead Attention → 5 Probabilities
```

**ExplainableECGNet** — 276,421 parameters — processes a 12×1000 input through residual convolutional blocks with dual attention (temporal + lead) to produce multi-label predictions _and_ interpretability maps in one forward pass.

> **Deep dive →** [Architecture docs](docs/ARCHITECTURE.md) for tensor shapes at every layer, design decisions, and module breakdown.

---

## Explainability

What separates TrustECG from a black box — the model tells you _where it looked_ and _why_.

<table>
<tr>
<td width="55%">

| Method                    | Shows                                   |
| ------------------------- | --------------------------------------- |
| **Temporal Attention**    | Important time points (QRS, ST, T wave) |
| **Lead Attention**        | Which of 12 leads mattered most         |
| **Occlusion Sensitivity** | Impact of masking each lead             |
| **Attention Heatmap**     | Combined lead × time view               |

</td>
<td width="45%">

<img src="figures/08_lead_importance.png" alt="Lead Importance" width="100%"/>

</td>
</tr>
</table>

> **Deep dive →** [Explainability docs](docs/explainability.md) for method details and clinical validation.

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/yourusername/TrustECG.git && cd TrustECG
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e .

# 2. Launch the dashboard
streamlit run src/app/streamlit_app.py
```

Then: **Load Model** → enter an ECG ID (try `9`, `42`, or `100`) → **Load ECG** → switch to the **Explainability** page.

> **Need more help?** See [Getting Started](docs/getting-started.md) for detailed setup, troubleshooting, and first-prediction walkthrough.

<details>
<summary><b>Programmatic usage (Python)</b></summary>

```python
import torch, wfdb
from src.app.streamlit_app import ExplainableECGNet, ECGPreprocessor

model = ExplainableECGNet()
model.load_state_dict(torch.load("checkpoints/trustecg_model.pt", map_location="cpu"))
model.eval()

record = wfdb.rdrecord("dataset/records100/00000/00009_lr")
x = torch.from_numpy(ECGPreprocessor()(record.p_signal.T)[None, ...])

with torch.no_grad():
    probs = model(x, return_attention=True)["probs"].squeeze()

for cls, p in zip(["NORM","MI","STTC","CD","HYP"], probs):
    print(f"{cls}: {p:.1%}")
```

> **Full API →** [API Reference](docs/api-reference.md)

</details>

---

## Project Structure

```
TrustECG/
├── src/app/streamlit_app.py        # All-in-one: model + preprocessing + dashboard
├── notebooks/TrustECG_Notebook.ipynb  # Training & evaluation pipeline
├── checkpoints/                    # Trained weights + config
├── dataset/                        # PTB-XL (21,801 ECGs)
├── figures/                        # Visualizations
├── reports/                        # PDF + DOCX project report
├── docs/                           # Full documentation suite
└── pyproject.toml                  # Dependencies
```

---

## Documentation

> All deep-dive documentation lives in [`docs/`](docs/index.md).

|                  | Guide                                                  | What You'll Learn                             |
| ---------------- | ------------------------------------------------------ | --------------------------------------------- |
| :house:          | [Documentation Home](docs/index.md)                    | Overview & navigation                         |
| :rocket:         | [Getting Started](docs/getting-started.md)             | Install, setup, first prediction              |
| :bar_chart:      | [Data Pipeline](docs/data-pipeline.md)                 | PTB-XL preprocessing, splits, class weighting |
| :brain:          | [Architecture](docs/ARCHITECTURE.md)                   | ExplainableECGNet layer-by-layer breakdown    |
| :dart:           | [Training](docs/training.md)                           | Loss function, metrics, reproducibility       |
| :mag:            | [Explainability](docs/explainability.md)               | Attention maps, occlusion, clinical meaning   |
| :computer:       | [Dashboard](docs/dashboard.md)                         | Streamlit pages, theming, deployment          |
| :books:          | [API Reference](docs/api-reference.md)                 | Every class, function, parameter              |
| :page_facing_up: | [Project Report](reports/TrustECG%20Report.pdf)        | Full methodology & findings                   |
| :notebook:       | [Training Notebook](notebooks/TrustECG_Notebook.ipynb) | End-to-end code pipeline                      |

---

## Tech Stack

|               |                                  |                       |
| ------------- | -------------------------------- | --------------------- |
| **Model**     | ExplainableECGNet · 276K params  | PyTorch + Lightning   |
| **Data**      | PTB-XL · 21,801 ECGs · 12 leads  | wfdb + scipy          |
| **App**       | Streamlit dashboard · dark theme | Plotly visualizations |
| **Inference** | ~10ms/ECG · CPU or GPU           |                       |

---

## References

1. Wagner et al. — _"PTB-XL, a large publicly available electrocardiography dataset."_ Scientific Data, 2020.
2. Vaswani et al. — _"Attention is all you need."_ NeurIPS, 2017.
3. He et al. — _"Deep residual learning for image recognition."_ CVPR, 2016.

---

<div align="center">

**TrustECG** — Explainable AI for cardiac diagnosis you can trust

</div>
