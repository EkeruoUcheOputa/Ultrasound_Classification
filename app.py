import sys
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. System Setup & Path Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transforms import tf_cnn, tf_vit
from src.models import build_resnet, build_effnet, build_vit, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Page Configuration (Updated Favicon)
st.set_page_config(
    page_title="Breast Cancer CADx System",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Executive Medical Theme & CSS Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main Background & Executive Typography */
    .stApp {
        background: #0b1320;
        color: #e2e8f0;
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Matured Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b;
    }

    /* Professional Clinical Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 6px solid #0d9488;
        padding: 2.2rem 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 2rem;
        text-align: left;
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 0.6rem;
        line-height: 1.5;
    }

    /* Matured Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: border-color 0.2s ease-in-out;
    }
    .metric-card:hover {
        border-color: #0d9488;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #64748b;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 0.3rem;
    }

    /* Academic Group Credit Card */
    .group-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .group-header {
        color: #14b8a6;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.8px;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
    }
    .member-item {
        color: #cbd5e1;
        font-size: 0.82rem;
        padding: 0.3rem 0;
        border-bottom: 1px solid #334155;
    }
    .member-item:last-child {
        border-bottom: none;
    }
    .member-id {
        color: #38bdf8;
        font-weight: 600;
        float: right;
    }

    /* Clean File Uploader Frame */
    div[data-testid="stFileUploader"] {
        background: #1e293b;
        border: 1px dashed #475569;
        border-radius: 10px;
        padding: 1.2rem;
    }
    
    /* Section Headers */
    h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 3. Cached Resource Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ensemble_artifacts():
    res_path = PROJECT_ROOT / "artifacts" / "best_resnet_50.pth"
    eff_path = PROJECT_ROOT / "artifacts" / "best_efficientnet_b4.pth"
    vit_path = PROJECT_ROOT / "artifacts" / "best_vit_b16.pth"
    meta_path = PROJECT_ROOT / "artifacts" / "stacked_meta_learner_c2.joblib"

    res_model = build_resnet(res_path, device, extract_features=False)
    eff_model = build_effnet(eff_path, device, extract_features=False)
    vit_model = build_vit(vit_path, device, extract_features=False)
    meta_learner = joblib.load(meta_path)

    return res_model, eff_model, vit_model, meta_learner

# -----------------------------------------------------------------------------
# 4. Main Application Interface
# -----------------------------------------------------------------------------
def main():
    # Clinical Header Banner with Asclepius Icon
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">⚕️ Computer-Aided Diagnosis (CADx) System</h1>
            <p class="hero-subtitle">Deep Feature Ensemble & Stacked Meta-Learning Infrastructure for B-Mode Ultrasound Breast Cancer Risk Classification</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load artifacts
    with st.spinner("⚡ Initializing Neural Networks & Meta-Learner..."):
        try:
            res_model, eff_model, vit_model, meta_learner = load_ensemble_artifacts()
        except Exception as e:
            st.error(f"Error loading model artifacts: {e}")
            st.stop()

    # Sidebar Architecture Panel
    st.sidebar.markdown("### 🔬 System Pipeline")
    st.sidebar.markdown(
        """
        - **ResNet-50** *(384x384 Deep Feature Extractor)*
        - **EfficientNet-B4** *(384x384 Scaled CNN)*
        - **ViT-B/16** *(224x224 Vision Transformer)*
        - **Stacked Classifier** *(L2 Regularized C=2.0 Meta-Learner)*
        """
    )
    st.sidebar.markdown("---")
    
    # Academic Note
    st.sidebar.info("💡 **Academic Note:** This platform provides clinical decision support and requires validation by a qualified radiologist.")
    
    # Group Developers Panel
    st.sidebar.markdown(
        """
        <div class="group-card">
            <div class="group-header">👥 PROJECT DEVELOPERS (GROUP ONE)</div>
            <div class="member-item">EKERUO UCHE OPUTA <span class="member-id">DU/C/236</span></div>
            <div class="member-item">NYAM JESSY NNENNA <span class="member-id">DU/C/130</span></div>
            <div class="member-item">OLAFISOYE TEMITOPE <span class="member-id">DU/C/260</span></div>
            <div class="member-item">UZOETOH LOVETH CHIDIMMA <span class="member-id">DU/C/321</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Image Input Section
    st.markdown("---")
    col_upload, col_preview = st.columns([1.1, 0.9])

    with col_upload:
        st.markdown("### 📥 Image Acquisition")
        uploaded_file = st.file_uploader(
            "Upload B-mode Ultrasound Scan (PNG, JPG, JPEG)", 
            type=["png", "jpg", "jpeg"]
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with col_preview:
            st.markdown("### 🖼️ Diagnostic Frame")
            st.image(image, caption="Active B-Mode Ultrasound Scan", use_container_width=True)

        # Preprocessing & Tensors
        t_cnn = tf_cnn(image).unsqueeze(0).to(device)
        t_vit = tf_vit(image).unsqueeze(0).to(device)

        # Model Inference Passes
        with torch.no_grad():
            p_res = torch.softmax(res_model(t_cnn), dim=1).cpu().numpy()[0]
            p_eff = torch.softmax(eff_model(t_cnn), dim=1).cpu().numpy()[0]
            p_vit = torch.softmax(vit_model(t_vit), dim=1).cpu().numpy()[0]

            stacked_9d = np.hstack([p_res, p_eff, p_vit]).reshape(1, -1)

        # Meta-Learner Prediction
        final_probs = meta_learner.predict_proba(stacked_9d)[0]
        final_pred_idx = np.argmax(final_probs)
        predicted_class = CLASS_NAMES[final_pred_idx].upper()
        confidence = final_probs[final_pred_idx] * 100

        st.markdown("<br><h2 style='text-align: center;'>📊 Diagnostic Inference & Risk Stratification</h2>", unsafe_allow_html=True)

        # Clinical Status Display
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Diagnosis</div>
                    <div class="metric-value" style="color: #38bdf8;">{predicted_class}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Ensemble Confidence</div>
                    <div class="metric-value" style="color: #14b8a6;">{confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:
            if predicted_class == "MALIGNANT":
                badge_bg = "rgba(225, 29, 72, 0.15)"
                badge_border = "#f43f5e"
                badge_text = "🚨 HIGH RISK: Malignant"
            elif predicted_class == "BENIGN":
                badge_bg = "rgba(217, 119, 6, 0.15)"
                badge_border = "#f59e0b"
                badge_text = "⚠️ MODERATE: Benign"
            else:
                badge_bg = "rgba(16, 185, 129, 0.15)"
                badge_border = "#10b981"
                badge_text = "✅ LOW RISK: Normal Tissue"

            st.markdown(
                f"""
                <div class="metric-card" style="background: {badge_bg}; border: 1px solid {badge_border};">
                    <div class="metric-label">Clinical Stratification</div>
                    <div class="metric-value" style="color: {badge_border}; font-size: 1.35rem; margin-top: 0.5rem;">{badge_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Matured Analytics Visualizations
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.markdown("### 🎯 Meta-Learner Gauge Score")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Confidence Index ({predicted_class})", 'font': {'color': '#94a3b8', 'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#64748b"},
                    'bar': {'color': "#0d9488"},
                    'bgcolor': "#1e293b",
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(244, 63, 94, 0.2)"},
                        {'range': [50, 80], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [80, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f1f5f9"},
                height=340
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with v_col2:
            st.markdown("### 🧬 Multi-Model Ensemble Breakdown")
            df_breakdown = pd.DataFrame({
                "Class": [c.capitalize() for c in CLASS_NAMES],
                "ResNet-50": p_res,
                "EfficientNet-B4": p_eff,
                "ViT-B/16": p_vit
            }).melt(id_vars="Class", var_name="Model", value_name="Probability")

            fig_grouped = px.bar(
                df_breakdown,
                x="Class",
                y="Probability",
                color="Model",
                barmode="group",
                range_y=[0, 1],
                text_auto=".1%",
                color_discrete_sequence=["#38bdf8", "#0d9488", "#64748b"]
            )
            fig_grouped.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f1f5f9"},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=340
            )
            st.plotly_chart(fig_grouped, use_container_width=True)

        # Tabulated Metrics
        with st.expander("📋 Tabular Multi-Model Prediction Matrix"):
            df_table = pd.DataFrame(
                [p_res, p_eff, p_vit, final_probs], 
                index=["ResNet-50", "EfficientNet-B4", "ViT-B/16", "Stacked Meta-Learner"],
                columns=[c.capitalize() for c in CLASS_NAMES]
            )
            st.dataframe(df_table.style.format("{:.4f}").highlight_max(axis=1, color="#1e3a8a"))

if __name__ == "__main__":
    main()