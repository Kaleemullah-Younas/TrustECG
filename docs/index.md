# TrustECG Documentation

Welcome to the TrustECG documentation. This guide covers every aspect of the project, from setup to model internals.

---

## Table of Contents

| Document                              | Description                                                   |
| ------------------------------------- | ------------------------------------------------------------- |
| [Getting Started](getting-started.md) | Installation, setup, and running your first prediction        |
| [Data Pipeline](data-pipeline.md)     | PTB-XL dataset, preprocessing, and data loading               |
| [Model Architecture](architecture.md) | ExplainableECGNet design, layers, and tensor flow             |
| [Training Guide](training.md)         | Training loop, loss function, class imbalance, and evaluation |
| [Explainability](explainability.md)   | Attention visualization, occlusion analysis, lead importance  |
| [Dashboard Guide](dashboard.md)       | Streamlit app pages, features, and customization              |
| [API Reference](api-reference.md)     | Complete class and function reference                         |

---

## Project at a Glance

|                 |                                                                     |
| --------------- | ------------------------------------------------------------------- |
| **What**        | Multi-label 12-lead ECG classification with built-in explainability |
| **Dataset**     | PTB-XL (21,801 clinical ECGs, 5 diagnostic classes)                 |
| **Model**       | ExplainableECGNet (276K parameters)                                 |
| **Performance** | 92.1% Val AUROC, 91.2% Test AUROC                                   |
| **Interface**   | Professional Streamlit dashboard with dark cardiac theme            |

---

## Quick Links

- **Run the app**: `streamlit run src/app/streamlit_app.py`
- **Training notebook**: [notebooks/TrustECG_Notebook.ipynb](../notebooks/TrustECG_Notebook.ipynb)
- **Trained weights**: [checkpoints/trustecg_model.pt](../checkpoints/trustecg_model.pt)
- **Full report**: [reports/TrustECG Report.pdf](../reports/TrustECG%20Report.pdf)

---

## How the Documentation is Organized

**Learning path** (recommended reading order):

```
Getting Started → Data Pipeline → Model Architecture → Training → Explainability → Dashboard
```

**Quick reference**: Jump straight to [API Reference](api-reference.md) if you already understand the project and need class/function details.

---

## Contributing

All core logic lives in two files:

| File                                | What It Contains                                                       |
| ----------------------------------- | ---------------------------------------------------------------------- |
| `src/app/streamlit_app.py`          | Model architecture, preprocessing, inference, visualization, dashboard |
| `notebooks/TrustECG_Notebook.ipynb` | Full training pipeline, evaluation, and analysis                       |

The Streamlit app is self-contained by design so it can be deployed as a single file without module import issues.
