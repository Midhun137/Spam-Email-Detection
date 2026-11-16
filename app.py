"""Batch 15: Spam Email Detection Using Ensemble Learning.

Streamlit Academic Demonstration Dashboard
- View 1: Analyze Email (Interactive single-message classifier with sub-model breakdown)
- View 2: Model Performance (Interactive Plotly evaluation benchmarks, ablation & curves)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import load

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/spam_ensemble_pipeline.joblib")
METADATA_PATH = Path("models/metadata.json")
METRICS_PATH = Path("reports/metrics.json")
ABLATION_PATH = Path("reports/ablation_metrics.json")
CURVES_PATH = Path("reports/curves_data.json")
DATASET_SUMMARY_PATH = Path("data/processed/dataset_summary.json")

# Fictional example presets for immediate viva demonstration
EXAMPLE_SPAM = {
    "sender": "winner@prize-rewards-direct.biz",
    "subject": "URGENT: Claim your $1,000,000 lottery cash prize now!!!",
    "body": (
        "CONGRATULATIONS! You have been selected as our grand cash prize winner!\n"
        "Click http://claim-prize-rewards.biz/win immediately to verify your identity\n"
        "and claim your cash instantly. Call 1-800-555-CASH now! Limited spots available!"
    ),
}

EXAMPLE_LEGIT = {
    "sender": "david.chen@engineering-corp.io",
    "subject": "Quarterly project sync meeting notes and deployment roadmap",
    "body": (
        "Hi team,\n\n"
        "Please find the summary of our quarterly review attached. We have completed\n"
        "phase 1 integration tests, and staging deployment is scheduled for tomorrow at 10 AM.\n\n"
        "Let me know if anyone has feedback before the rollout.\n\n"
        "Best regards,\nDavid"
    ),
}


# ---------------------------------------------------------------------------
# Data and Model Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_pipeline(path: Path = MODEL_PATH):
    """Load serialized scikit-learn pipeline."""
    if not path.exists():
        return None
    try:
        return load(path)
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_json_artifact(path: Path) -> Optional[Dict[str, Any]]:
    """Safely load cached JSON reports."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page Setup and Scoped Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Email Detection | Ensemble Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Clean typography & panel styling */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Hero Header */
    .app-header {
        padding: 0.8rem 1.2rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 8px;
        color: #f8fafc;
        margin-bottom: 1.2rem;
        border: 1px solid #334155;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0 0 0.2rem 0;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    .app-header p {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0;
    }

    /* Content Cards */
    .content-panel {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* Prediction Result Cards */
    .verdict-box-spam {
        background-color: #fff1f2;
        border: 1.5px solid #f43f5e;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-box-legit {
        background-color: #f0fdf4;
        border: 1.5px solid #10b981;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-badge-spam {
        display: inline-block;
        background-color: #e11d48;
        color: #ffffff;
        font-size: 0.95rem;
        font-weight: 700;
        padding: 0.25rem 0.85rem;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .verdict-badge-legit {
        display: inline-block;
        background-color: #059669;
        color: #ffffff;
        font-size: 0.95rem;
        font-weight: 700;
        padding: 0.25rem 0.85rem;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Metric Callout Card */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        border-left: 4px solid #2563eb;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    .stat-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Muted instruction container */
    .guide-box {
        background-color: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 8px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        color: #64748b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header & Navigation
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🛡️ Spam Email Detection — Soft-Voting Ensemble</h1>
        <p>Batch 15 Supervised Academic ML Project | Naive Bayes + Logistic Regression + Calibrated Linear SVM</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top navigation
current_view = st.radio(
    "Navigation Views",
    options=["Analyze Email", "Model Performance"],
    horizontal=True,
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Sidebar Information
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Status")
    pipeline = get_model_pipeline()
    if pipeline is not None:
        st.success("Trained Ensemble Pipeline Active")
    else:
        st.error("Model Artifact Missing")
        st.info("Train the model using:\n`python -m src.evaluate`")

    st.divider()

    st.markdown("### 🧠 Ensemble Architecture")
    st.markdown(
        """
        - **Naive Bayes**: $\\alpha = 0.1$ (Weight: 1.0)
        - **Logistic Reg**: $C = 1.0$ (Weight: 0.25)
        - **Linear SVM**: Calibrated Sigmoid (Weight: 1.5)
        - **Decision Rule**: Soft Voting $\\ge 0.50$
        """
    )

    st.divider()

    st.markdown("### 🔒 Privacy Guarantee")
    st.caption(
        "Email text submitted in this demo is processed strictly in-memory "
        "and is never logged, stored, or transmitted to any external server."
    )


# ===========================================================================
# VIEW 1: ANALYZE EMAIL
# ===========================================================================
if current_view == "Analyze Email":
    # Session state initialization for form inputs
    if "input_sender" not in st.session_state:
        st.session_state.input_sender = ""
    if "input_subject" not in st.session_state:
        st.session_state.input_subject = ""
    if "input_body" not in st.session_state:
        st.session_state.input_body = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("#### ✉️ Email Submission")

        # Preset Loader Actions
        p_col1, p_col2 = st.columns(2)
        if p_col1.button("📥 Load Spam Example", use_container_width=True):
            st.session_state.input_sender = EXAMPLE_SPAM["sender"]
            st.session_state.input_subject = EXAMPLE_SPAM["subject"]
            st.session_state.input_body = EXAMPLE_SPAM["body"]
            st.session_state.last_result = None
            st.rerun()

        if p_col2.button("📥 Load Legitimate Example", use_container_width=True):
            st.session_state.input_sender = EXAMPLE_LEGIT["sender"]
            st.session_state.input_subject = EXAMPLE_LEGIT["subject"]
            st.session_state.input_body = EXAMPLE_LEGIT["body"]
            st.session_state.last_result = None
            st.rerun()

        # Main Input Form
        with st.form("prediction_form", clear_on_submit=False):
            sender_input = st.text_input(
                "Sender Address (Optional)",
                value=st.session_state.input_sender,
                placeholder="e.g., alex@organization.com or winner@lottery-promo.biz",
                help="Sender domain and missing-sender flags are derived from this address.",
            )
            subject_input = st.text_input(
                "Subject Line (Optional)",
                value=st.session_state.input_subject,
                placeholder="e.g., Project update or Urgent account verification",
            )
            body_input = st.text_area(
                "Email Body Text",
                value=st.session_state.input_body,
                height=180,
                placeholder="Type or paste the message body here...",
            )

            analyze_btn = st.form_submit_button(
                "🔍 Analyze Email", type="primary", use_container_width=True
            )

        if analyze_btn:
            # Sync session state with the latest submitted values to preserve user edits across reruns
            st.session_state.input_sender = sender_input
            st.session_state.input_subject = subject_input
            st.session_state.input_body = body_input

            if not body_input.strip() and not subject_input.strip():
                st.warning("Please provide at least a subject line or message body to analyze.")
            elif pipeline is None:
                st.error(
                    "Trained model artifact not found at `models/spam_ensemble_pipeline.joblib`. "
                    "Run `python -m src.evaluate` to generate the production model."
                )
            else:
                input_df = pd.DataFrame(
                    [
                        {
                            "subject": subject_input,
                            "sender": sender_input,
                            "sender_address": sender_input,
                            "body": body_input,
                        }
                    ]
                )

                # Execute pipeline inference
                preparer = pipeline.named_steps["preparer"]
                preprocessor = pipeline.named_steps["preprocessor"]
                classifier = pipeline.named_steps["classifier"]

                prep_df = preparer.transform(input_df)
                X_sparse = preprocessor.transform(prep_df)

                pred_class = int(classifier.predict(X_sparse)[0])
                pred_probs = classifier.predict_proba(X_sparse)[0]
                spam_prob = float(pred_probs[1])

                # Extract sub-estimator probabilities
                sub_probs = {}
                sub_estimators = classifier.named_estimators_
                if "nb" in sub_estimators:
                    sub_probs["Multinomial Naive Bayes (w=1.0)"] = float(
                        sub_estimators["nb"].predict_proba(X_sparse)[0][1]
                    )
                if "lr" in sub_estimators:
                    sub_probs["Logistic Regression (w=0.25)"] = float(
                        sub_estimators["lr"].predict_proba(X_sparse)[0][1]
                    )
                if "svm" in sub_estimators:
                    sub_probs["Calibrated Linear SVM (w=1.5)"] = float(
                        sub_estimators["svm"].predict_proba(X_sparse)[0][1]
                    )

                # Inspect text recognition using fitted vectorizer
                vectorizer = preprocessor.named_transformers_["text"]
                analyzer = vectorizer.build_analyzer()
                vocab = vectorizer.vocabulary_

                subj_tokens = analyzer(subject_input) if subject_input.strip() else []
                subj_rec = [t for t in subj_tokens if t in vocab]
                body_tokens = analyzer(body_input) if body_input.strip() else []
                body_rec = [t for t in body_tokens if t in vocab]
                total_tokens = len(subj_tokens) + len(body_tokens)
                total_rec = len(subj_rec) + len(body_rec)
                coverage_ratio = total_rec / max(total_tokens, 1)

                is_limited = False
                if total_tokens > 0:
                    if total_rec <= 2:
                        is_limited = True
                    elif len(body_tokens) >= 8 and len(body_rec) <= 1:
                        is_limited = True
                    elif total_tokens >= 8 and coverage_ratio < 0.18:
                        is_limited = True

                st.session_state.last_result = {
                    "is_spam": bool(pred_class == 1),
                    "spam_prob": spam_prob,
                    "sub_probs": sub_probs,
                    "is_limited_text": is_limited,
                    "text_recognition": {
                        "total_tokens": total_tokens,
                        "total_rec": total_rec,
                        "body_tokens": len(body_tokens),
                        "body_rec": len(body_rec),
                        "coverage_ratio": coverage_ratio,
                    },
                    "extracted_features": {
                        "sender_domain": prep_df["sender_domain"].iloc[0],
                        "url_count": int(prep_df["url_count"].iloc[0]),
                        "exclamation_count": int(prep_df["exclamation_count"].iloc[0]),
                        "uppercase_ratio": float(prep_df["uppercase_ratio"].iloc[0]),
                        "digit_ratio": float(prep_df["digit_ratio"].iloc[0]),
                        "has_html": int(prep_df["has_html"].iloc[0]),
                    },
                }

    with col_result:
        st.markdown("#### 📊 Classification Verdict")

        result = st.session_state.last_result

        if result is None:
            st.markdown(
                """
                <div class="guide-box">
                    <p style="font-size: 1.05rem; font-weight: 600; color: #475569; margin-bottom: 0.4rem;">
                        No Email Analyzed Yet
                    </p>
                    <p style="font-size: 0.88rem; margin: 0;">
                        Enter an email on the left and click <strong>Analyze Email</strong>, or click one of the <strong>Load Example</strong> buttons above to run an instant demonstration.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            is_spam = result["is_spam"]
            spam_p = result["spam_prob"]
            legit_p = 1.0 - spam_p

            if is_spam:
                st.markdown(
                    f"""
                    <div class="verdict-box-spam">
                        <span class="verdict-badge-spam">Predicted: Spam</span>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #9f1239; margin-top: 0.5rem; margin-bottom: 0.25rem;">
                            Estimated Spam Probability: {spam_p*100:.1f}%
                        </div>
                        <div style="font-size: 0.9rem; color: #881337; font-weight: 500;">
                            Decision threshold: 50.0% (0.50)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-box-legit">
                        <span class="verdict-badge-legit">Predicted: Legitimate</span>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #065f46; margin-top: 0.5rem; margin-bottom: 0.25rem;">
                            Estimated Spam Probability: {spam_p*100:.1f}%
                        </div>
                        <div style="font-size: 0.9rem; color: #047857; font-weight: 500;">
                            Decision threshold: 50.0% (0.50)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Warning if very little recognizable text was found
            if result.get("is_limited_text", False):
                st.warning("⚠️ **Limited text recognized. Treat this prediction with caution.**")

            # Visual Probability Gauge with Threshold Marker
            gauge_fig = go.Figure()
            gauge_fig.add_trace(
                go.Bar(
                    x=[spam_p * 100],
                    y=["Ensemble"],
                    orientation="h",
                    marker=dict(color="#e11d48" if is_spam else "#0d9488"),
                    text=[f"{spam_p*100:.1f}%"],
                    textposition="inside",
                    textfont=dict(color="white", size=13, weight="bold"),
                    hoverinfo="none",
                )
            )
            # Threshold vertical line at 50%
            gauge_fig.add_vline(
                x=50,
                line_width=2.5,
                line_dash="dash",
                line_color="#475569",
                annotation_text="Decision Threshold (50%)",
                annotation_position="top",
                annotation_font=dict(size=10, color="#334155"),
            )
            gauge_fig.update_layout(
                xaxis=dict(range=[0, 100], ticksuffix="%", title=None, showgrid=True),
                yaxis=dict(showticklabels=False),
                height=90,
                margin=dict(l=10, r=15, t=25, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})

            # Sub-Model Breakdown Horizontal Bar Chart
            st.markdown("##### 🔬 Sub-Model Voting Contributions")
            sub_probs = result.get("sub_probs", {})
            if sub_probs:
                models_plot = list(sub_probs.keys()) + ["Soft-Voting Ensemble"]
                probs_plot = [sub_probs[k] * 100 for k in sub_probs.keys()] + [spam_p * 100]
                colors_plot = [
                    "#f43f5e" if p >= 50 else "#10b981" for p in probs_plot[:-1]
                ] + ["#2563eb"]  # Distinctive blue for Ensemble

                sub_fig = go.Figure(
                    go.Bar(
                        x=probs_plot,
                        y=models_plot,
                        orientation="h",
                        marker=dict(color=colors_plot),
                        text=[f"{p:.1f}%" for p in probs_plot],
                        textposition="outside",
                        textfont=dict(size=11, weight="bold"),
                        hoverinfo="text",
                        hovertext=[f"{m}: {p:.2f}% spam probability" for m, p in zip(models_plot, probs_plot)],
                    )
                )
                sub_fig.add_vline(x=50, line_dash="dot", line_color="#94a3b8", line_width=1.5)
                sub_fig.update_layout(
                    xaxis=dict(range=[0, 108], ticksuffix="%", title="Spam Probability"),
                    yaxis=dict(autorange="reversed"),
                    height=200,
                    margin=dict(l=10, r=25, t=10, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(sub_fig, use_container_width=True, config={"displayModeBar": False})

            # Derived Characteristics Expander
            with st.expander("🔎 Extracted Message Signals", expanded=True):
                feats = result["extracted_features"]
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                domain_disp = feats["sender_domain"]
                if len(domain_disp) > 16:
                    domain_disp = domain_disp[:14] + "..."
                m_c1.metric("Sender Domain", domain_disp, help=feats["sender_domain"])
                m_c2.metric("URL Count", feats["url_count"])
                m_c3.metric("Exclamations", feats["exclamation_count"])
                m_c4.metric("Upper Ratio", f"{feats['uppercase_ratio']*100:.1f}%")

                t_info = result.get("text_recognition", {})
                if t_info:
                    st.caption(
                        f"**Text Recognition**: {t_info.get('total_rec', 0)} of {t_info.get('total_tokens', 0)} "
                        f"tokens recognized in fitted TF-IDF vocabulary ({t_info.get('coverage_ratio', 0.0)*100:.1f}% coverage)."
                    )


# ===========================================================================
# VIEW 2: MODEL PERFORMANCE DASHBOARD
# ===========================================================================
elif current_view == "Model Performance":
    metrics_data = load_json_artifact(METRICS_PATH)
    ablation_data = load_json_artifact(ABLATION_PATH)
    curves_data = load_json_artifact(CURVES_PATH)
    dataset_data = load_json_artifact(DATASET_SUMMARY_PATH)

    if not metrics_data:
        st.error(
            "Evaluation metrics file `reports/metrics.json` not found. "
            "Please run `python -m src.evaluate` to generate the complete evaluation benchmarks."
        )
        st.stop()

    test_results = metrics_data.get("test_results", {})
    split_stats = metrics_data.get("split_stats", {})

    # Top Compact KPI Cards
    ens_metrics = test_results.get("Ensemble (Soft - Tuned Weights)", {})
    kpi_f1 = ens_metrics.get("f1", 0.0)
    kpi_prec = ens_metrics.get("precision", 0.0)
    kpi_rec = ens_metrics.get("recall", 0.0)
    kpi_acc = ens_metrics.get("accuracy", 0.0)
    test_size = split_stats.get("test_size", 901)
    test_ham = split_stats.get("test_ham_count", 569)
    test_spam = split_stats.get("test_spam_count", 332)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Ensemble Spam F1 (Primary)</div>
                <div class="stat-value">{kpi_f1:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Spam Precision</div>
                <div class="stat-value">{kpi_prec*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Spam Recall</div>
                <div class="stat-value">{kpi_rec*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Test Evaluation Set</div>
                <div class="stat-value">{test_size} <span style="font-size:0.8rem;color:#64748b;font-weight:normal;">({test_ham}H / {test_spam}S)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Section 1: Model Comparison
    # -----------------------------------------------------------------------
    st.markdown("#### 1. Model Comparison on Held-Out Test Data")

    metric_choice = st.selectbox(
        "Select metric to compare:",
        options=["Spam F1 (Primary)", "Spam Precision", "Spam Recall", "Accuracy"],
        index=0,
    )

    metric_key_map = {
        "Spam F1 (Primary)": "f1",
        "Spam Precision": "precision",
        "Spam Recall": "recall",
        "Accuracy": "accuracy",
    }
    target_metric_key = metric_key_map[metric_choice]

    # Build model comparison data
    model_rows = []
    for m_name, m_dict in test_results.items():
        model_rows.append(
            {
                "Model": m_name,
                "Score": m_dict.get(target_metric_key, 0.0),
                "IsEnsemble": "Ensemble" in m_name,
            }
        )
    df_compare = pd.DataFrame(model_rows).sort_values("Score", ascending=True)

    colors = [
        "#2563eb" if r["Model"] == "Ensemble (Soft - Tuned Weights)"
        else ("#0284c7" if r["IsEnsemble"] else ("#94a3b8" if "Dummy" in r["Model"] else "#475569"))
        for _, r in df_compare.iterrows()
    ]

    comp_fig = go.Figure(
        go.Bar(
            x=df_compare["Score"],
            y=df_compare["Model"],
            orientation="h",
            marker=dict(color=colors),
            text=[f"{s:.4f}" for s in df_compare["Score"]],
            textposition="outside",
            textfont=dict(size=11, weight="bold"),
            hoverinfo="text",
            hovertext=[f"{r['Model']}: {r['Score']:.4f}" for _, r in df_compare.iterrows()],
        )
    )
    comp_fig.update_layout(
        xaxis=dict(
            range=[0, 1.05],
            title=metric_choice,
            showgrid=True,
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(title=None),
        height=280,
        margin=dict(l=20, r=30, t=15, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(comp_fig, use_container_width=True)

    # Factual conclusion sentence based on measured data
    st.caption(
        "💡 **Key Finding**: The **Tuned Soft-Voting Ensemble** achieved the highest overall Spam F1 score of **0.9713** (97.89% accuracy), "
        "while **Calibrated Linear SVM** was the strongest individual classifier with an F1 score of **0.9665** and 97.84% precision."
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 2: Confusion Matrix Inspection
    # -----------------------------------------------------------------------
    st.markdown("#### 2. Confusion Matrix & Error Pattern Analysis")

    cm_col1, cm_col2 = st.columns([1.1, 0.9], gap="large")

    with cm_col1:
        cm_model_choice = st.selectbox(
            "Select model to inspect confusion matrix:",
            options=[
                "Ensemble (Soft - Tuned Weights)",
                "Calibrated Linear SVM",
                "Multinomial Naive Bayes",
                "Logistic Regression",
                "Ensemble (Soft - Equal Weights)",
            ],
            index=0,
        )

        selected_cm_dict = test_results.get(cm_model_choice, {})
        cm_matrix = np.array(selected_cm_dict.get("confusion_matrix", [[0, 0], [0, 0]]))

        tn, fp = cm_matrix[0, 0], cm_matrix[0, 1]
        fn, tp = cm_matrix[1, 0], cm_matrix[1, 1]

        z_text = [
            [f"<b>{tn}</b><br>(True Ham)", f"<b>{fp}</b><br>(False Pos)"],
            [f"<b>{fn}</b><br>(False Neg)", f"<b>{tp}</b><br>(True Spam)"],
        ]

        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm_matrix,
                x=["Predicted Ham", "Predicted Spam"],
                y=["Actual Ham", "Actual Spam"],
                text=z_text,
                texttemplate="%{text}",
                textfont={"size": 14},
                colorscale="Blues",
                showscale=False,
                hoverinfo="text",
                hovertext=[
                    [
                        f"True Negatives: {tn} legitimate emails correctly delivered to Inbox",
                        f"False Positives: {fp} legitimate emails incorrectly flagged as Spam",
                    ],
                    [
                        f"False Negatives: {fn} spam emails missed and allowed into Inbox",
                        f"True Positives: {tp} spam emails correctly caught and blocked",
                    ],
                ],
            )
        )
        cm_fig.update_layout(
            xaxis=dict(title="Predicted Label", side="bottom"),
            yaxis=dict(title="Actual Ground Truth", autorange="reversed"),
            height=260,
            margin=dict(l=10, r=10, t=15, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    with cm_col2:
        st.markdown("##### ⚖️ Error Distribution & Trade-Offs")
        st.markdown(
            f"""
            - **False Positives (Type I Error)**: `{fp}` legitimate emails were misclassified as spam.
              In email filtering, false positives are the most critical operational hazard because users may miss crucial client contracts or personal correspondence.
            - **False Negatives (Type II Error)**: `{fn}` spam emails slipped past the filter into the inbox.
            - **True Positives**: `{tp}` spam emails intercepted out of `{tp + fn}` total spam messages (**{selected_cm_dict.get('recall', 0.0)*100:.2f}% recall**).
            - **True Negatives**: `{tn}` legitimate emails safely routed to the inbox (**{tn / (tn + fp)*100:.2f}% specificity**).
            """
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Section 3: Feature Ablation Comparison
    # -----------------------------------------------------------------------
    st.markdown("#### 3. Feature Group Ablation Study")
    st.markdown(
        "Demonstrates the incremental contribution of text TF-IDF, sender domain metadata, and structural message characteristics."
    )

    if ablation_data:
        ab_names = list(ablation_data.keys())
        ab_f1s = [ablation_data[k]["f1"] for k in ab_names]
        ab_precs = [ablation_data[k]["precision"] for k in ab_names]
        ab_recs = [ablation_data[k]["recall"] for k in ab_names]

        # Calculate percentage point changes relative to Text Only
        baseline_f1 = ab_f1s[0]
        f1_diffs = [
            f"Baseline ({s:.4f})" if i == 0 else f"{s:.4f} ({(s - baseline_f1)*100:+.2f} pp)"
            for i, s in enumerate(ab_f1s)
        ]

        ab_fig = go.Figure()
        ab_fig.add_trace(
            go.Bar(
                name="Spam Precision",
                x=ab_names,
                y=ab_precs,
                marker_color="#0284c7",
                text=[f"{p:.3f}" for p in ab_precs],
                textposition="outside",
            )
        )
        ab_fig.add_trace(
            go.Bar(
                name="Spam Recall",
                x=ab_names,
                y=ab_recs,
                marker_color="#059669",
                text=[f"{r:.3f}" for r in ab_recs],
                textposition="outside",
            )
        )
        ab_fig.add_trace(
            go.Bar(
                name="Spam F1 (Primary)",
                x=ab_names,
                y=ab_f1s,
                marker_color="#2563eb",
                text=f1_diffs,
                textposition="outside",
            )
        )

        ab_fig.update_layout(
            barmode="group",
            yaxis=dict(range=[0.92, 1.01], title="Score", gridcolor="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320,
            margin=dict(l=20, r=20, t=25, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(ab_fig, use_container_width=True)

        st.caption(
            "📌 **Ablation Insight**: Incorporating sender domain categories alone caused a slight drop (-0.62 pp) due to domain sparsity in unseen test senders. "
            "However, combining sender data with **9 structural message characteristics** (URL density, capitalization, punctuation, HTML presence) pushed Spam F1 to its peak of **0.9713** (+1.23 pp gain)."
        )
    else:
        st.info("Feature ablation results not found in `reports/ablation_metrics.json`.")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 4: Precision–Recall & ROC Curves
    # -----------------------------------------------------------------------
    st.markdown("#### 4. Precision–Recall & ROC Curves")

    pr_col1, pr_col2 = st.columns(2, gap="large")

    with pr_col1:
        st.markdown("##### Precision–Recall Trajectories")
        if curves_data:
            pr_fig = go.Figure()
            color_map = {
                "Ensemble (Soft - Tuned)": "#2563eb",
                "Calibrated Linear SVM": "#7c3aed",
                "Multinomial Naive Bayes": "#0d9488",
                "Logistic Regression": "#d97706",
            }
            for model_name, c_data in curves_data.items():
                is_ens = "Ensemble" in model_name
                pr_fig.add_trace(
                    go.Scatter(
                        x=c_data["recall"],
                        y=c_data["precision"],
                        mode="lines",
                        name=f"{model_name} (AP: {c_data['ap']:.4f})",
                        line=dict(
                            color=color_map.get(model_name, "#64748b"),
                            width=3.2 if is_ens else 1.8,
                        ),
                    )
                )
            # Baseline test set prevalence
            pr_fig.add_hline(
                y=0.3685,
                line_dash="dot",
                line_color="#94a3b8",
                annotation_text="Spam Prevalence (36.9%)",
                annotation_position="bottom right",
            )
            pr_fig.update_layout(
                xaxis=dict(title="Recall (Spam Caught)", range=[0.80, 1.01], gridcolor="#f1f5f9"),
                yaxis=dict(title="Precision (Accuracy of Flagged)", range=[0.85, 1.01], gridcolor="#f1f5f9"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0),
                height=300,
                margin=dict(l=15, r=15, t=15, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(pr_fig, use_container_width=True)
        else:
            st.info("Curve data not found in `reports/curves_data.json`.")

    with pr_col2:
        st.markdown("##### Receiver Operating Characteristic (ROC)")
        if curves_data:
            roc_fig = go.Figure()
            for model_name, c_data in curves_data.items():
                is_ens = "Ensemble" in model_name
                roc_fig.add_trace(
                    go.Scatter(
                        x=c_data["fpr"],
                        y=c_data["tpr"],
                        mode="lines",
                        name=f"{model_name} (AUC: {c_data['auc']:.4f})",
                        line=dict(
                            color=color_map.get(model_name, "#64748b"),
                            width=3.2 if is_ens else 1.8,
                        ),
                    )
                )
            roc_fig.add_shape(
                type="line",
                line=dict(dash="dash", color="#cbd5e1", width=1),
                x0=0,
                x1=1,
                y0=0,
                y1=1,
            )
            roc_fig.update_layout(
                xaxis=dict(title="False Positive Rate", range=[-0.01, 0.15], gridcolor="#f1f5f9"),
                yaxis=dict(title="True Positive Rate", range=[0.85, 1.01], gridcolor="#f1f5f9"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="left", x=0),
                height=300,
                margin=dict(l=15, r=15, t=15, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(roc_fig, use_container_width=True)
        else:
            st.info("Curve data not found in `reports/curves_data.json`.")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 5: Dataset Breakdown
    # -----------------------------------------------------------------------
    st.markdown("#### 5. Dataset Class Distribution")

    dataset_choice = st.radio(
        "Select dataset partition to display:",
        options=["Full Cleaned Corpus (N = 4,533)", "Held-Out Test Set (N = 901)"],
        horizontal=True,
    )

    if "Full" in dataset_choice:
        ham_cnt = 2722
        spam_cnt = 1811
        tot_cnt = ham_cnt + spam_cnt
        part_label = "Full Cleaned Apache SpamAssassin Corpus"
    else:
        ham_cnt = test_ham
        spam_cnt = test_spam
        tot_cnt = ham_cnt + spam_cnt
        part_label = "Held-Out Test Partition (Zero Near-Duplicate Leakage)"

    data_fig = go.Figure()
    data_fig.add_trace(
        go.Bar(
            y=[part_label],
            x=[ham_cnt],
            name="Legitimate Email (Ham)",
            orientation="h",
            marker=dict(color="#0d9488"),
            text=[f"{ham_cnt:,} ({ham_cnt/tot_cnt*100:.1f}%)"],
            textposition="inside",
            textfont=dict(color="white", weight="bold"),
        )
    )
    data_fig.add_trace(
        go.Bar(
            y=[part_label],
            x=[spam_cnt],
            name="Unsolicited Email (Spam)",
            orientation="h",
            marker=dict(color="#e11d48"),
            text=[f"{spam_cnt:,} ({spam_cnt/tot_cnt*100:.1f}%)"],
            textposition="inside",
            textfont=dict(color="white", weight="bold"),
        )
    )
    data_fig.update_layout(
        barmode="stack",
        height=130,
        margin=dict(l=10, r=20, t=10, b=25),
        xaxis=dict(title="Number of Emails", gridcolor="#f1f5f9"),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(data_fig, use_container_width=True)
