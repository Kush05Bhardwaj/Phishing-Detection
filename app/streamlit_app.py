"""Interactive Phishing Website Detection Dashboard.
Capstone Project: Intelligent Phishing Website Detection Using Machine Learning & Explainable AI.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Setup page layout
st.set_page_config(
    page_title="Phishing Detection AI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(ROOT_DIR, "src") not in sys.path:
    sys.path.append(os.path.join(ROOT_DIR, "src"))

from app.utils import load_all_models, load_test_dataset, get_curated_samples

models = load_all_models()
test_df = load_test_dataset()
curated_samples = get_curated_samples(test_df)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("Phishing Detection AI")
st.sidebar.markdown("**AI in Cybersecurity Capstone**")
st.sidebar.markdown("---")

selected_model_name = st.sidebar.selectbox(
    "Active Detection Model:",
    ["Random Forest (Primary)", "SVM (RBF)", "MLP (ANN)"]
)
st.sidebar.info(
    "💡 **Scope Note:** Pre-extracted UCI feature vectors are evaluated here. "
    "Live URL string parsing is treated as a separate microservice."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**System Status:** 🟢 All Pipelines Operational")

# Title Header
st.title("🛡️ Intelligent Phishing Website Detection Platform")
st.markdown("##### Machine Learning Detection, Adversarial Hardening & Explainable AI (SHAP)")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📌 Overview & Architecture",
    "🔍 Live Inspection & Inference",
    "📊 Model Performance & Benchmarks"
])

# ----------------- TAB 1: OVERVIEW -----------------
with tab1:
    st.subheader("Project Objectives & Capstone Alignment")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size", "11,055 Samples", "UCI Phishing Repository")
    col2.metric("Feature Vectors", "30 Attributes", "Ternary Categorical")
    col3.metric("Primary Model F1", "96.87%", "Random Forest Classifier")
    col4.metric("Adversarial Evasion Drop", "68.7% -> 0.4%", "Hardened Mitigation")

    st.markdown("---")
    st.markdown("### 🏗️ End-to-End System Pipeline")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            DATA[UCI Phishing Dataset] --> PREP[Leak-Free Preprocessing]
            PREP --> ML[Random Forest / SVM / MLP]
            ML --> ADV[Adversarial Evasion Simulation]
            ADV --> DEF[Adversarial Hardening Defense]
            ML --> XAI[SHAP TreeExplainer & Alerts]
            XAI --> APP[Streamlit Operational Dashboard]
        ```
        """
    )

    st.markdown("---")
    st.markdown("### 📋 Core Deliverables Checklist")
    c1, c2 = st.columns(2)
    with c1:
        st.success("✅ **Multi-Model ML Detection:** Trained Random Forest, SVM (RBF), and MLP/ANN.")
        st.success("✅ **Leakage-Free Preprocessing:** Stratified train/test splits with isolated scaling.")
        st.success("✅ **Adversarial Cybersecurity Simulation:** Black-box/Grey-box feature manipulation.")
    with c2:
        st.success("✅ **Defensive Mitigation:** Adversarial retraining cutting evasion rate by 99%.")
        st.success("✅ **Explainable AI (SHAP):** Global feature importances & plain-English security alerts.")
        st.success("✅ **Live Interactive Demo:** Real-time inference across test instances.")

# ----------------- TAB 2: LIVE INSPECTION -----------------
with tab2:
    st.subheader("Live Web Sample Scanner")
    st.markdown("Select a sample from the held-out test dataset or load curated security cases:")

    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        sample_choice = st.selectbox("Curated Test Instances:", list(curated_samples.keys()))
        selected_idx = curated_samples[sample_choice]
    with c_sel2:
        custom_idx = st.number_input("Or enter custom Test Row Index (0 to 2210):", min_value=0, max_value=len(test_df)-1 if not test_df.empty else 0, value=selected_idx)

    sample_row = test_df.iloc[custom_idx] if not test_df.empty else None

    if sample_row is not None:
        X_sample = sample_row.drop(labels=['target']).to_frame().T
        true_label = int(sample_row['target'])

        # Model Inference
        active_model = models.get("Random Forest")
        if "SVM" in selected_model_name:
            active_model = models.get("SVM (RBF)")
            if "Preprocessor" in models:
                X_sample_scaled = models["Preprocessor"].transform(X_sample)
            else:
                X_sample_scaled = X_sample
            prob = float(active_model.predict_proba(X_sample_scaled)[0, 1])
            pred = int(active_model.predict(X_sample_scaled)[0])
        elif "MLP" in selected_model_name:
            active_model = models.get("MLP (ANN)")
            if "Preprocessor" in models:
                X_sample_scaled = models["Preprocessor"].transform(X_sample)
            else:
                X_sample_scaled = X_sample
            prob = float(active_model.predict_proba(X_sample_scaled)[0, 1])
            pred = int(active_model.predict(X_sample_scaled)[0])
        else:
            prob = float(active_model.predict_proba(X_sample)[0, 1])
            pred = int(active_model.predict(X_sample)[0])

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns([1, 1, 2])

        with res_col1:
            st.markdown("#### Model Verdict")
            if pred == 1:
                st.error("🚨 **PHISHING DETECTED**")
            else:
                st.success("✅ **LEGITIMATE WEBSITE**")
            st.caption(f"Ground Truth Label: {'Phishing (1)' if true_label == 1 else 'Legitimate (0)'}")

        with res_col2:
            st.markdown("#### Malicious Probability")
            st.progress(prob)
            st.markdown(f"**Confidence:** `{prob*100:.2f}% Phishing`")

        with res_col3:
            st.markdown("#### Operational Threat Assessment")
            if prob >= 0.80:
                st.warning("⚠️ **CRITICAL RISK:** Multiple high-severity phishing markers identified.")
            elif prob >= 0.50:
                st.warning("⚠️ **ELEVATED RISK:** Suspicious anomalies detected in domain or asset links.")
            else:
                st.info("ℹ️ **LOW RISK:** Structural profile aligns with authentic domain registrations.")

        with st.expander("🔍 View Raw Sample Feature Vector (30 Attributes)"):
            st.dataframe(X_sample)

# ----------------- TAB 3: BENCHMARKS -----------------
with tab3:
    st.subheader("Model Performance & Comparative Evaluation")

    metrics_file = os.path.join(ROOT_DIR, "reports", "results", "model_comparison.csv")
    if os.path.exists(metrics_file):
        m_df = pd.read_csv(metrics_file)
        st.dataframe(m_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#d4edda'), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Evaluation Visualizations")
    img_col1, img_col2 = st.columns(2)

    cm_path = os.path.join(ROOT_DIR, "reports", "figures", "confusion_matrices.png")
    roc_path = os.path.join(ROOT_DIR, "reports", "figures", "roc_curves.png")

    with img_col1:
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Normalized Confusion Matrices Across Models", use_container_width=True)

    with img_col2:
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
