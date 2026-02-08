"""
TrustECG - Explainable Multi-Label ECG Classification
Professional Streamlit Dashboard
Author: TrustECG Team
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="TrustECG | AI-Powered ECG Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== CONSTANTS ====================
CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

CLASS_INFO = {
    "NORM": {
        "icon": "✅",
        "color": "#10B981",
        "name": "Normal",
        "desc": "No significant cardiac abnormalities detected",
    },
    "MI": {
        "icon": "💔",
        "color": "#EF4444",
        "name": "Myocardial Infarction",
        "desc": "Signs of heart attack or ischemic damage to heart muscle",
    },
    "STTC": {
        "icon": "📉",
        "color": "#F59E0B",
        "name": "ST/T Change",
        "desc": "Abnormalities in ST segment or T wave morphology",
    },
    "CD": {
        "icon": "⚡",
        "color": "#8B5CF6",
        "name": "Conduction Disturbance",
        "desc": "Abnormal electrical conduction patterns in the heart",
    },
    "HYP": {
        "icon": "💪",
        "color": "#06B6D4",
        "name": "Hypertrophy",
        "desc": "Signs of enlarged heart chambers (thickened walls)",
    },
}

# ==================== PROFESSIONAL CSS ====================
st.markdown(
    """
<style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #E63946 0%, #1D3557 100%);
        padding: 3rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(230, 57, 70, 0.3);
    }
    
    .hero-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .hero-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.75rem;
        font-weight: 300;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1A1F26 0%, #252D37 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.4);
        border-color: #E63946;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #E63946 0%, #F97316 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: #1A1F26;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }
    
    .feature-card h4 {
        color: #E63946;
        margin-bottom: 1rem;
    }
    
    /* Alert boxes */
    .success-alert {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    .info-alert {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%);
        border: 1px solid #3B82F6;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
    }
    
    /* Prediction bar */
    .pred-row {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        transition: background 0.2s;
    }
    
    .pred-row:hover {
        background: rgba(255,255,255,0.05);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1419 0%, #1A1F26 100%);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1A1F26;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #E63946;
        border-radius: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==================== MODEL ARCHITECTURE ====================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.Tanh(), nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (x * attn_weights).sum(dim=1)
        return context, attn_weights.squeeze(-1)


class LeadAttention(nn.Module):
    def __init__(self, feature_dim, num_leads=12):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.Tanh(), nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (x * attn_weights).sum(dim=1)
        return context, attn_weights.squeeze(-1)


class ExplainableECGNet(nn.Module):
    def __init__(
        self,
        num_leads=12,
        sequence_length=1000,
        num_classes=5,
        encoder_channels=[32, 64, 128],
        dropout=0.3,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        encoder_layers = []
        in_ch = 1
        for out_ch in encoder_channels:
            encoder_layers.append(ResidualBlock1D(in_ch, out_ch, stride=2, dropout=dropout))
            in_ch = out_ch
        self.lead_encoder = nn.Sequential(*encoder_layers)

        encoded_len = sequence_length
        for _ in encoder_channels:
            encoded_len = (encoded_len + 1) // 2
        self.encoded_len = encoded_len
        self.feature_dim = encoder_channels[-1]

        self.temporal_attention = TemporalAttention(self.feature_dim)
        self.lead_attention = LeadAttention(self.feature_dim, num_leads)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_attention=False):
        batch_size = x.shape[0]
        x = x.view(batch_size * self.num_leads, 1, self.sequence_length)
        encoded = self.lead_encoder(x)
        encoded = encoded.view(batch_size, self.num_leads, self.feature_dim, -1)
        encoded = encoded.permute(0, 1, 3, 2)

        lead_features, temporal_weights_all = [], []
        for i in range(self.num_leads):
            lead_seq = encoded[:, i, :, :]
            lead_feat, temp_weights = self.temporal_attention(lead_seq)
            lead_features.append(lead_feat)
            temporal_weights_all.append(temp_weights)

        lead_features = torch.stack(lead_features, dim=1)
        temporal_weights_all = torch.stack(temporal_weights_all, dim=1)
        context, lead_weights = self.lead_attention(lead_features)
        logits = self.classifier(context)
        probs = torch.sigmoid(logits)

        if return_attention:
            return {
                "logits": logits,
                "probs": probs,
                "temporal_attention": temporal_weights_all,
                "lead_attention": lead_weights,
            }
        return {"logits": logits, "probs": probs}


# ==================== PREPROCESSING ====================
class ECGPreprocessor:
    """Bandpass filter (0.5-40 Hz) + Z-score normalization."""

    def __init__(self, lowcut=0.5, highcut=40.0, fs=100):
        self.lowcut, self.highcut, self.fs = lowcut, highcut, fs

    def __call__(self, signal):
        nyquist = 0.5 * self.fs
        b, a = butter(N=2, Wn=[self.lowcut / nyquist, self.highcut / nyquist], btype="band")
        filtered = np.array([filtfilt(b, a, signal[i]) for i in range(signal.shape[0])])
        mean, std = filtered.mean(), filtered.std() + 1e-8
        return ((filtered - mean) / std).astype(np.float32)


preprocessor = ECGPreprocessor()


# ==================== SESSION STATE ====================
def init_session():
    for key in ["model", "predictions", "current_ecg", "ecg_id", "attention"]:
        if key not in st.session_state:
            st.session_state[key] = None


# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    model = ExplainableECGNet()
    checkpoint_paths = [Path("checkpoints/trustecg_model.pt"), Path("notebooks/best_model.pt")]

    for path in checkpoint_paths:
        if path.exists():
            try:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                return model, path.name
            except Exception:
                continue
    return model, "untrained"


# ==================== DATA LOADING ====================
def load_ecg_by_id(ecg_id: int):
    """Load and preprocess ECG from PTB-XL dataset."""
    try:
        import wfdb

        folder = (ecg_id - 1) // 1000 * 1000
        path = f"dataset/records100/{folder:05d}/{ecg_id:05d}_lr"
        record = wfdb.rdrecord(path)
        return preprocessor(record.p_signal.T)
    except Exception as e:
        st.error(f"Error loading ECG {ecg_id}: {e}")
        return generate_demo_ecg()


def generate_demo_ecg():
    """Generate synthetic demo ECG signal."""
    np.random.seed(42)
    signal = np.zeros((12, 1000))
    for i in range(12):
        base = np.random.randn(1000) * 0.05
        for beat in np.linspace(0, 10, 12):
            idx = int(beat * 100)
            if idx < 980:
                qrs = np.exp(-(np.arange(-20, 20) ** 2) / 20) * (0.5 + 0.3 * np.random.rand())
                signal[i, idx : idx + 40] += qrs
        signal[i] += base
    return signal.astype(np.float32)


def run_prediction(signal):
    """Run model inference on ECG signal."""
    model = st.session_state.model
    if model is None:
        return None, None

    model.eval()
    x = torch.from_numpy(np.asarray(signal, dtype=np.float32)[np.newaxis, ...])
    with torch.no_grad():
        output = model(x, return_attention=True)
        return output["probs"].numpy().squeeze(), {
            "temporal": output.get("temporal_attention"),
            "lead": output.get("lead_attention"),
        }


# ==================== VISUALIZATION ====================
def plot_ecg(signal, title="12-Lead ECG", highlight_leads=None):
    """Create professional 12-lead ECG plot."""
    fig = make_subplots(
        rows=6, cols=2, subplot_titles=LEAD_NAMES, vertical_spacing=0.05, horizontal_spacing=0.06
    )
    t = np.linspace(0, 10, signal.shape[1])

    for i, lead in enumerate(LEAD_NAMES):
        row, col = (i % 6) + 1, (i // 6) + 1
        color = "#E63946" if highlight_leads and lead in highlight_leads else "#3B82F6"
        fig.add_trace(
            go.Scatter(
                x=t,
                y=signal[i],
                mode="lines",
                line=dict(color=color, width=1.2),
                name=lead,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        height=550,
        showlegend=False,
        margin=dict(t=60, b=30, l=50, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,31,38,0.8)",
        font=dict(color="white"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(255,255,255,0.1)",
        tickfont=dict(size=9),
        title_font=dict(size=10),
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=9)
    )
    return fig


def plot_prediction_bars(probs):
    """Create prediction confidence bar chart."""
    colors = [CLASS_INFO[c]["color"] for c in CLASS_NAMES]

    fig = go.Figure(
        go.Bar(
            x=probs,
            y=[f"{CLASS_INFO[c]['icon']} {c}" for c in CLASS_NAMES],
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
            textfont=dict(color="white", size=12),
        )
    )

    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text="Threshold",
        annotation_position="top",
    )

    fig.update_layout(
        height=280,
        margin=dict(t=20, b=20, l=10, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(range=[0, 1.1], showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


def plot_lead_radar(attention_dict):
    """Create lead importance radar chart."""
    if attention_dict and attention_dict.get("lead") is not None:
        lead_attn = attention_dict["lead"].cpu().numpy().squeeze()
        if lead_attn.ndim > 1:
            lead_attn = lead_attn.mean(axis=0)
        values = list(lead_attn) + [lead_attn[0]]
    else:
        values = [0.5] * 13

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=LEAD_NAMES + [LEAD_NAMES[0]],
            fill="toself",
            fillcolor="rgba(230, 57, 70, 0.3)",
            line=dict(color="#E63946", width=2),
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(t=30, b=30, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(26,31,38,0.8)",
            radialaxis=dict(
                visible=True,
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="white", size=8),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color="white", size=10)
            ),
        ),
        font=dict(color="white"),
    )
    return fig


def plot_attention_heatmap(attention_dict):
    """Create temporal attention heatmap."""
    if attention_dict and attention_dict.get("temporal") is not None:
        temp_attn = attention_dict["temporal"].cpu().numpy().squeeze()
        if temp_attn.ndim == 1:
            temp_attn = np.repeat(temp_attn[np.newaxis, :], 12, axis=0)
    else:
        temp_attn = np.random.rand(12, 125)

    fig = go.Figure(
        go.Heatmap(
            z=temp_attn,
            y=LEAD_NAMES,
            colorscale=[[0, "#1A1F26"], [0.5, "#E63946"], [1, "#FECACA"]],
            showscale=True,
            colorbar=dict(tickfont=dict(color="white")),
        )
    )

    fig.update_layout(
        title=dict(text="Temporal Attention by Lead", font=dict(color="white", size=14)),
        height=350,
        margin=dict(t=50, b=40, l=60, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Time Step", tickfont=dict(color="white"), title_font=dict(color="white")),
        yaxis=dict(title="Lead", tickfont=dict(color="white"), title_font=dict(color="white")),
    )
    return fig


# ==================== PAGES ====================
def render_sidebar():
    """Render professional sidebar."""
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align: center; padding: 1.5rem 0;">
            <span style="font-size: 4rem;">🫀</span>
            <h1 style="color: #E63946; margin: 0.5rem 0; font-size: 1.8rem; font-weight: 700;">
                TrustECG
            </h1>
            <p style="color: #9CA3AF; font-size: 0.9rem; margin: 0;">
                AI-Powered ECG Analysis
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Navigation
        page = st.radio(
            "",
            ["🏠 Dashboard", "🔮 Analyze ECG", "🔍 Explainability"],
            label_visibility="collapsed",
        )

        st.divider()

        # Model status
        st.markdown("##### ⚙️ Model Status")
        if st.session_state.model is None:
            if st.button("🚀 Load Model", type="primary", use_container_width=True):
                with st.spinner("Initializing AI..."):
                    model, checkpoint = load_model()
                    st.session_state.model = model
                    st.success(f"✓ {checkpoint}")
        else:
            st.success("✓ Model Ready", icon="✅")

        st.divider()

        # Info
        st.markdown("##### 📊 Dataset Info")
        col1, col2 = st.columns(2)
        col1.metric("Records", "21,801")
        col2.metric("Classes", "5")

        st.divider()
        st.caption("TrustECG v1.0 | © 2026")

    return page


def render_dashboard():
    """Render home dashboard."""
    # Hero header
    st.markdown(
        """
    <div class="hero-header">
        <h1>🫀 TrustECG</h1>
        <p>Explainable AI for Multi-Label ECG Classification</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("92.1%", "Validation AUROC"),
        ("91.2%", "Test AUROC"),
        ("21,801", "ECG Records"),
        ("276K", "Parameters"),
    ]

    for col, (val, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Features section
    st.markdown("### 🎯 Key Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="feature-card">
            <h4>🏥 Multi-Label Classification</h4>
            <p style="color: #9CA3AF;">
                Simultaneously detect 5 cardiac conditions from a single ECG:
                Normal, MI, ST/T Changes, Conduction Disturbance, and Hypertrophy.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="feature-card">
            <h4>🔍 Explainable AI</h4>
            <p style="color: #9CA3AF;">
                Understand model decisions through temporal attention,
                lead importance visualization, and occlusion sensitivity analysis.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="feature-card">
            <h4>📈 Clinical-Grade Data</h4>
            <p style="color: #9CA3AF;">
                Trained on PTB-XL, the largest publicly available ECG dataset
                with 21,801 cardiologist-verified 12-lead recordings.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Diagnostic classes
    st.markdown("### 📋 Diagnostic Classes")
    col1, col2 = st.columns([3, 2])

    with col1:
        df = pd.DataFrame(
            {
                "Class": [f"{CLASS_INFO[c]['icon']} {c}" for c in CLASS_NAMES],
                "Condition": [CLASS_INFO[c]["name"] for c in CLASS_NAMES],
                "Description": [CLASS_INFO[c]["desc"] for c in CLASS_NAMES],
            }
        )
        st.dataframe(df, hide_index=True, use_container_width=True)

    with col2:
        fig = go.Figure(
            go.Pie(
                labels=CLASS_NAMES,
                values=[9438, 4134, 5078, 4891, 2258],
                hole=0.5,
                marker_colors=[CLASS_INFO[c]["color"] for c in CLASS_NAMES],
                textinfo="label+percent",
                textfont=dict(color="white", size=11),
            )
        )
        fig.update_layout(
            height=280,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_analyze():
    """Render ECG analysis page."""
    st.markdown("## 🔮 ECG Analysis")

    if st.session_state.model is None:
        st.markdown(
            """
        <div class="warning-alert">
            <strong>⚠️ Model Not Loaded</strong><br>
            Please load the model from the sidebar to begin analysis.
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### 📥 Load ECG from PTB-XL Dataset")

        st.markdown(
            """
        <div class="info-alert">
            <strong>💡 How to use:</strong> Enter an ECG ID (1-21800) from the PTB-XL dataset.
            The ID corresponds to the patient record number.<br>
            <em>Example: ID <strong>9</strong> loads file <code>records100/00000/00009_lr</code></em>
        </div>
        """,
            unsafe_allow_html=True,
        )

        ecg_id = st.number_input(
            "ECG ID",
            min_value=1,
            max_value=21800,
            value=9,
            help="Enter patient record ID from PTB-XL dataset",
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            load_btn = st.button("📥 Load ECG", type="primary", use_container_width=True)
        with col_btn2:
            demo_btn = st.button("🎲 Demo ECG", use_container_width=True)

        if load_btn:
            with st.spinner("Loading ECG..."):
                signal = load_ecg_by_id(ecg_id)
                st.session_state.current_ecg = signal
                st.session_state.ecg_id = ecg_id
                st.success(f"✓ Loaded ECG #{ecg_id}")

        if demo_btn:
            signal = generate_demo_ecg()
            st.session_state.current_ecg = signal
            st.session_state.ecg_id = "Demo"

        # Display ECG
        if st.session_state.current_ecg is not None:
            fig = plot_ecg(st.session_state.current_ecg, f"ECG Record #{st.session_state.ecg_id}")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🎯 Prediction Results")

        if st.session_state.current_ecg is not None:
            with st.spinner("Analyzing..."):
                probs, attention = run_prediction(st.session_state.current_ecg)

            if probs is not None:
                st.session_state.predictions = probs
                st.session_state.attention = attention

                # Result summary
                detected = [CLASS_NAMES[i] for i, p in enumerate(probs) if p > 0.5]

                if not detected or detected == ["NORM"]:
                    st.markdown(
                        """
                    <div class="success-alert">
                        <strong>✅ Normal ECG</strong><br>
                        No significant cardiac abnormalities detected.
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    abnormal = [c for c in detected if c != "NORM"]
                    st.markdown(
                        f"""
                    <div class="warning-alert">
                        <strong>⚠️ Conditions Detected</strong><br>
                        {', '.join([f"{CLASS_INFO[c]['icon']} {CLASS_INFO[c]['name']}" for c in abnormal])}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Prediction bars
                st.plotly_chart(plot_prediction_bars(probs), use_container_width=True)

                # Condition details
                st.markdown("##### 📝 Condition Details")
                for cls, prob in zip(CLASS_NAMES, probs):
                    if prob > 0.3:  # Show conditions with >30% probability
                        info = CLASS_INFO[cls]
                        status = "🔴" if prob > 0.5 else "🟡"
                        st.markdown(
                            f"""
                        **{status} {info['icon']} {info['name']}** ({prob:.1%})  
                        <small style="color: #9CA3AF;">{info['desc']}</small>
                        """,
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(
                """
            <div class="info-alert">
                <strong>👆 Load an ECG</strong><br>
                Enter an ECG ID and click "Load ECG" to begin analysis.
            </div>
            """,
                unsafe_allow_html=True,
            )


def render_explainability():
    """Render explainability page."""
    st.markdown("## 🔍 Model Explainability")

    if st.session_state.predictions is None:
        st.markdown(
            """
        <div class="warning-alert">
            <strong>⚠️ No Analysis Available</strong><br>
            Please analyze an ECG first on the "Analyze ECG" page.
        </div>
        """,
            unsafe_allow_html=True,
        )
        return

    signal = st.session_state.current_ecg
    attention = st.session_state.attention

    st.markdown(
        """
    <div class="info-alert">
        <strong>🧠 Understanding AI Decisions</strong><br>
        TrustECG uses attention mechanisms to focus on the most clinically relevant parts of the ECG.
        This page visualizes which leads and time segments influenced the prediction.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Attention visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📡 Lead Importance")
        st.caption("Which ECG leads contributed most to the prediction")
        st.plotly_chart(plot_lead_radar(attention), use_container_width=True)

    with col2:
        st.markdown("#### 📊 Prediction Confidence")
        st.caption("Model confidence for each diagnostic class")
        st.plotly_chart(
            plot_prediction_bars(st.session_state.predictions), use_container_width=True
        )

    # Heatmap
    st.markdown("#### 🔥 Temporal Attention Heatmap")
    st.caption("Where the model focused across time for each lead")
    st.plotly_chart(plot_attention_heatmap(attention), use_container_width=True)

    # Occlusion analysis
    st.markdown("---")
    st.markdown("#### 🔬 Occlusion Sensitivity Analysis")
    st.caption("Measures prediction change when each lead is masked (higher = more important)")

    if st.button("▶️ Run Analysis", type="primary"):
        with st.spinner("Running occlusion analysis..."):
            importance = np.zeros((12, 5))
            model = st.session_state.model
            model.eval()

            x = torch.from_numpy(signal[np.newaxis, ...].astype(np.float32))
            with torch.no_grad():
                baseline = model(x)["probs"].numpy().squeeze()

            for i in range(12):
                occluded = signal.copy()
                occluded[i, :] = 0
                x = torch.from_numpy(occluded[np.newaxis, ...].astype(np.float32))
                with torch.no_grad():
                    pred = model(x)["probs"].numpy().squeeze()
                importance[i] = np.abs(baseline - pred)

            fig = go.Figure(
                go.Heatmap(
                    z=importance.T,
                    x=LEAD_NAMES,
                    y=CLASS_NAMES,
                    colorscale=[[0, "#1A1F26"], [0.5, "#E63946"], [1, "#FECACA"]],
                    text=[[f"{v:.3f}" for v in row] for row in importance.T],
                    texttemplate="%{text}",
                    textfont=dict(color="white", size=10),
                    colorbar=dict(tickfont=dict(color="white")),
                )
            )
            fig.update_layout(
                title=dict(text="Lead Importance per Class", font=dict(color="white")),
                height=300,
                margin=dict(t=50, b=40, l=60, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    title="ECG Lead", tickfont=dict(color="white"), title_font=dict(color="white")
                ),
                yaxis=dict(
                    title="Class", tickfont=dict(color="white"), title_font=dict(color="white")
                ),
            )
            st.plotly_chart(fig, use_container_width=True)


# ==================== MAIN ====================
def main():
    init_session()
    page = render_sidebar()

    if page == "🏠 Dashboard":
        render_dashboard()
    elif page == "🔮 Analyze ECG":
        render_analyze()
    elif page == "🔍 Explainability":
        render_explainability()


if __name__ == "__main__":
    main()
