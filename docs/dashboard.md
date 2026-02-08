# Dashboard Guide

The TrustECG Streamlit dashboard provides an interactive interface for ECG analysis and model explainability.

---

## Running the Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

Opens at **http://localhost:8501**

---

## Theme & Design

The dashboard uses a custom dark theme with cardiac-red accents:

| Element        | Color            | Code      |
| -------------- | ---------------- | --------- |
| Primary accent | Cardiac red      | `#E63946` |
| Background     | Deep dark        | `#0F1419` |
| Cards / panels | Slightly lighter | `#1A1F26` |
| Text           | Clean white      | `#F8F9FA` |

Configured in `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#E63946"
backgroundColor = "#0F1419"
secondaryBackgroundColor = "#1A1F26"
textColor = "#F8F9FA"
font = "sans serif"
```

The app also injects custom CSS for:

- Gradient hero header
- Hover-animated metric cards
- Styled alert boxes (success, warning, info)
- Custom scrollbar
- Hidden Streamlit branding (menu, footer, header)

---

## Pages

### 1. Dashboard (Home)

The landing page provides a project overview.

**Components:**

- **Hero header**: Gradient banner with project name and tagline
- **Metric cards**: Val AUROC (92.1%), Test AUROC (91.2%), Records (21,801), Parameters (276K)
- **Feature cards**: Multi-label classification, Explainable AI, Clinical-grade data
- **Class table**: Diagnostic classes with descriptions
- **Distribution chart**: Donut chart showing class prevalence

### 2. Analyze ECG

The main analysis page for loading and classifying ECGs.

**Left panel** — Input:

- ECG ID input (1-21800)
- "Load ECG" button → loads from PTB-XL dataset
- "Demo ECG" button → generates synthetic ECG
- 12-lead ECG visualization (Plotly interactive chart)

**Right panel** — Results:

- Status alert (green for normal, red for abnormalities)
- Prediction confidence bars (horizontal bar chart with 0.5 threshold line)
- Condition details (expanded info for conditions > 30% probability)

### 3. Explainability

Visualization of model attention and feature importance.

**Components:**

- **Lead importance radar**: Polar chart showing attention weight per lead
- **Prediction confidence**: Bar chart (same as Analyze page)
- **Temporal attention heatmap**: 12 × 125 heatmap showing attention across leads and time
- **Occlusion sensitivity**: On-demand analysis (button click) showing per-lead, per-class importance

---

## Sidebar

The sidebar is persistent across all pages:

| Element          | Function                             |
| ---------------- | ------------------------------------ |
| Logo + branding  | TrustECG heart icon and name         |
| Navigation radio | Switch between 3 pages               |
| Model status     | Load model button / ready indicator  |
| Dataset info     | Record count and class count metrics |
| Version          | "TrustECG v1.0"                      |

---

## Session State

The app uses `st.session_state` to persist data across interactions:

| Key           | Type                          | Description                            |
| ------------- | ----------------------------- | -------------------------------------- |
| `model`       | `ExplainableECGNet` or `None` | Loaded model instance                  |
| `predictions` | `np.ndarray` or `None`        | Last prediction probabilities (5,)     |
| `current_ecg` | `np.ndarray` or `None`        | Current ECG signal (12, 1000)          |
| `ecg_id`      | `int` or `str` or `None`      | Current ECG identifier                 |
| `attention`   | `dict` or `None`              | Attention weights from last prediction |

---

## ECG Loading

Two methods to load ECGs:

### From PTB-XL Dataset

```python
def load_ecg_by_id(ecg_id: int):
    folder = (ecg_id - 1) // 1000 * 1000
    path = f"dataset/records100/{folder:05d}/{ecg_id:05d}_lr"
    record = wfdb.rdrecord(path)
    return preprocessor(record.p_signal.T)  # Returns (12, 1000) float32
```

### Demo ECG (Synthetic)

Generates a synthetic ECG with:

- Gaussian QRS-like peaks at regular intervals
- Random baseline noise
- 12 leads with slight amplitude variation

Useful for testing without the dataset.

---

## Visualizations

All charts use Plotly with a transparent background theme:

### 12-Lead ECG Plot

- 6×2 subplot grid (one per lead)
- Blue traces (default), red for highlighted leads
- Time axis: 0-10 seconds
- Grid lines at 10% opacity

### Prediction Bars

- Horizontal bar chart
- Color-coded by class (green for NORM, red for MI, etc.)
- Dashed vertical line at 0.5 threshold
- Text labels showing percentage

### Lead Radar Chart

- Polar/radar chart with 12 angular positions
- Fill color: semi-transparent cardiac red
- Values are attention weights (sum to 1.0)

### Temporal Heatmap

- 12 rows (leads) × 125 columns (compressed time steps)
- Custom colorscale: dark → red → light pink
- Interactive hover for exact values

### Occlusion Matrix

- 12 columns (leads) × 5 rows (classes)
- Cell values: absolute change in prediction probability
- Annotated with numeric values

---

## Customization

### Changing the Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#YOUR_COLOR"
backgroundColor = "#YOUR_BG"
```

### Modifying CSS

The CSS is embedded in `streamlit_app.py` inside the `st.markdown()` call at the top. Key classes:

| CSS Class        | Used For                  |
| ---------------- | ------------------------- |
| `.hero-header`   | Top gradient banner       |
| `.metric-card`   | Stats cards               |
| `.feature-card`  | Feature description boxes |
| `.success-alert` | Green notification boxes  |
| `.warning-alert` | Red notification boxes    |
| `.info-alert`    | Blue notification boxes   |
| `.pred-row`      | Prediction result rows    |

### Adding a New Page

1. Add a new option to the sidebar radio buttons
2. Create a `render_new_page()` function
3. Add routing in the `main()` function:

```python
def main():
    page = render_sidebar()
    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "🔮 Analyze ECG":
        render_analyze()
    elif page == "🔍 Explainability":
        render_explainability()
    elif page == "📊 New Page":
        render_new_page()
```

---

## Deployment

The app is self-contained in a single file (`src/app/streamlit_app.py`) with all model architecture, preprocessing, and visualization code. This makes deployment straightforward:

### Streamlit Cloud

1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Set main file: `src/app/streamlit_app.py`
4. Ensure `checkpoints/trustecg_model.pt` is committed

### Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8501
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501"]
```

---

**Next**: [API Reference](api-reference.md) for complete class and function documentation.
