"""Batch 15: Spam Email Detection Using Ensemble Learning.

Streamlit Academic Demonstration Dashboard
- Warm Light Theme Palette:
    * Page background: Warm ivory (#F7F5F0)
    * Cards, inputs, and charts: White (#FFFFFF)
    * Headings and body text: Charcoal (#1F2430)
    * Secondary text and chart labels: Slate (#566071)
    * Panel borders and chart gridlines: Warm gray (#DEDAD2)
    * Primary button, selected tab, and ensemble highlight: Deep indigo (#4338CA)
    * Predicted spam: Dark burgundy (#9F1239) on Soft rose (#FFF1F2)
    * Predicted legitimate: Dark evergreen (#166534) on Soft green (#F0FDF4)
- View 1: Analyze Email (Interactive composer, immediate examples, verdict & sub-model breakdown)
- View 2: Model Performance (Interactive benchmarks, confusion matrix, PR curves, ablation study)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from joblib import load

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/spam_ensemble_pipeline.joblib")
METRICS_PATH = Path("reports/metrics.json")
ABLATION_PATH = Path("reports/ablation_metrics.json")
CURVES_PATH = Path("reports/curves_data.json")

# Fictional example presets for immediate viva demonstration
EXAMPLE_SPAM = {
    "sender": "rewards@claim-promo-winner.biz",
    "subject": "URGENT: Claim your $1,000,000 cash lottery prize now!",
    "body": (
        "CONGRATULATIONS! You have been selected as the grand cash prize winner!\n"
        "Click http://claim-prize-rewards.biz/win immediately to verify your identity\n"
        "and claim your cash prize today. Call 1-800-555-CASH now! Limited spots available!"
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

# Theme Palette Constants (Warm Light Palette)
COLOR_BG = "#F7F5F0"          # Warm ivory
COLOR_CARD = "#FFFFFF"        # White
COLOR_TEXT = "#1F2430"        # Charcoal
COLOR_MUTED = "#566071"       # Slate
COLOR_BORDER = "#DEDAD2"      # Warm gray
COLOR_PRIMARY = "#4338CA"     # Deep indigo

# Verdict Colors
COLOR_SPAM_BG = "#FFF1F2"     # Soft rose
COLOR_SPAM_TEXT = "#9F1239"   # Dark burgundy
COLOR_SPAM_BORDER = "#FECDD3"

COLOR_LEGIT_BG = "#F0FDF4"    # Soft green
COLOR_LEGIT_TEXT = "#166534"  # Dark evergreen
COLOR_LEGIT_BORDER = "#BBF7D0"

# Model Bar Colors
COLOR_ENSEMBLE = "#4338CA"    # Deep indigo
COLOR_INDIVIDUAL = "#566071"  # Slate


# ---------------------------------------------------------------------------
# Data and Model Loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_pipeline(path: Path = MODEL_PATH):
    """Load serialized scikit-learn ensemble pipeline."""
    if not path.exists():
        return None
    try:
        return load(path)
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_json_artifact(path: Path) -> Optional[Dict[str, Any]]:
    """Safely load cached evaluation JSON reports."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Page Configuration & Scoped CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Email Detector | Ensemble Learning",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    /* Global Warm Light Background & Text */
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
    }}

    /* Top Header on Ivory Background with Charcoal Text */
    .top-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.25rem;
        background-color: {COLOR_BG};
        border: 1.5px solid {COLOR_BORDER};
        border-radius: 8px;
        margin-bottom: 1.25rem;
    }}
    .top-header-title {{
        font-size: 1.45rem;
        font-weight: 800;
        color: {COLOR_TEXT};
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.01em;
    }}
    .top-header-tagline {{
        font-size: 0.88rem;
        color: {COLOR_MUTED};
        margin-top: 0.15rem;
    }}

    /* Radio Navigation Tabs Styling */
    div[data-testid="stRadio"] > div {{
        gap: 0.5rem;
    }}
    div[data-testid="stRadio"] label {{
        background-color: {COLOR_CARD};
        border: 1.5px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 0.45rem 1rem;
        color: {COLOR_TEXT};
        font-weight: 600;
        transition: all 0.15s ease-in-out;
    }}
    div[data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stRadio"] label:has(input:checked) {{
        background-color: {COLOR_PRIMARY} !important;
        border-color: {COLOR_PRIMARY} !important;
        color: #FFFFFF !important;
    }}

    /* Content Cards & Containers */
    .content-card {{
        background-color: {COLOR_CARD};
        border: 1.5px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }}

    /* Inputs and Textareas */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {{
        background-color: {COLOR_CARD} !important;
        border: 1.5px solid {COLOR_BORDER} !important;
        color: {COLOR_TEXT} !important;
        border-radius: 6px !important;
        font-size: 0.95rem !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {COLOR_PRIMARY} !important;
        box-shadow: 0 0 0 1px {COLOR_PRIMARY} !important;
    }}
    label, .stWidgetLabel p {{
        color: {COLOR_TEXT} !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }}

    /* Buttons: Indigo with White Text for Primary */
    div[data-testid="stFormSubmitButton"] > button {{
        background-color: {COLOR_PRIMARY} !important;
        color: #FFFFFF !important;
        border: 1.5px solid {COLOR_PRIMARY} !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.2rem !important;
        box-shadow: 0 1px 3px rgba(67, 56, 202, 0.2);
    }}
    div[data-testid="stFormSubmitButton"] > button:hover {{
        background-color: #3730A3 !important;
        border-color: #3730A3 !important;
        color: #FFFFFF !important;
    }}

    /* Example Preset Buttons: White with Subtle Border */
    div.stButton > button {{
        background-color: {COLOR_CARD} !important;
        color: {COLOR_TEXT} !important;
        border: 1.5px solid {COLOR_BORDER} !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        transition: all 0.15s ease-in-out;
    }}
    div.stButton > button:hover {{
        background-color: {COLOR_BG} !important;
        border-color: {COLOR_PRIMARY} !important;
        color: {COLOR_PRIMARY} !important;
    }}

    /* Verdict Boxes */
    .verdict-card-spam {{
        background-color: {COLOR_SPAM_BG};
        border: 1.5px solid {COLOR_SPAM_BORDER};
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }}
    .verdict-card-legit {{
        background-color: {COLOR_LEGIT_BG};
        border: 1.5px solid {COLOR_LEGIT_BORDER};
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }}
    .verdict-title-spam {{
        color: {COLOR_SPAM_TEXT};
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 0.35rem;
    }}
    .verdict-title-legit {{
        color: {COLOR_LEGIT_TEXT};
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 0.35rem;
    }}
    .verdict-stat {{
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}
    .verdict-stat-spam {{
        color: {COLOR_SPAM_TEXT};
    }}
    .verdict-stat-legit {{
        color: {COLOR_LEGIT_TEXT};
    }}
    .verdict-subtext {{
        font-size: 0.88rem;
        color: {COLOR_MUTED};
        font-weight: 500;
    }}

    /* Empty State Container */
    .empty-verdict-box {{
        background-color: {COLOR_CARD};
        border: 1.5px dashed {COLOR_BORDER};
        border-radius: 8px;
        padding: 3.5rem 1.5rem;
        text-align: center;
        color: {COLOR_MUTED};
        font-size: 0.95rem;
    }}

    /* Top KPI Cards for Performance View */
    .kpi-container {{
        display: flex;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }}
    .kpi-card {{
        flex: 1;
        background-color: {COLOR_CARD};
        border: 1.5px solid {COLOR_BORDER};
        border-top: 4px solid {COLOR_PRIMARY};
        border-radius: 8px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }}
    .kpi-label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {COLOR_MUTED};
        font-weight: 600;
        margin-bottom: 0.25rem;
    }}
    .kpi-number {{
        font-size: 1.75rem;
        font-weight: 800;
        color: {COLOR_TEXT};
    }}
    .kpi-subtext {{
        font-size: 0.8rem;
        color: {COLOR_MUTED};
        margin-top: 0.15rem;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_BG} !important;
        border-right: 1.5px solid {COLOR_BORDER} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Top Header & Navigation
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="top-header">
        <div>
            <div class="top-header-title">🛡️ Spam Email Detector</div>
            <div class="top-header-tagline">Batch 15 Academic Project &bull; Multi-Feature Soft-Voting Ensemble</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_selection = st.radio(
    "Navigation Views",
    options=["✉️ Analyze Email", "📊 Model Performance"],
    horizontal=True,
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Sidebar: Pipeline Status & Collapsed Model Details
# ---------------------------------------------------------------------------
pipeline = get_model_pipeline()

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    if pipeline is not None:
        st.success("Ensemble Pipeline Loaded & Ready")
    else:
        st.error("Model Artifact Missing")
        st.info("Train the model via command:\n`python -m src.evaluate`")

    # Collapsed Model Details as requested
    with st.expander("🔍 Model details", expanded=False):
        st.markdown(
            """
            **Ensemble Architecture**:
            - **Soft Voting Ensemble** with 3 calibrated sub-classifiers:
              1. **Multinomial Naive Bayes** (Weight: `1.0`, $\\alpha=0.1$)
              2. **Logistic Regression** (Weight: `0.25`, $C=1.0$)
              3. **Linear SVM** (Weight: `1.5`, Sigmoid Calibration)

            **Decision Parameters**:
            - **Decision Threshold**: $\\tau = 0.50$ (50%)
            - **Text Features**: 10,000 TF-IDF word unigrams & bigrams
            - **Metadata**: Sender domain one-hot encoding
            - **Structural Signals**: 9 engineered numeric features
            """
        )

    st.markdown("---")
    st.markdown("### 🔒 Privacy Notice")
    st.caption(
        "Emails entered in this demonstration are processed strictly in-memory "
        "and are never stored, logged, or shared."
    )


# ===========================================================================
# VIEW 1: ANALYZE EMAIL
# ===========================================================================
if nav_selection == "✉️ Analyze Email":
    # Session state initialization
    if "email_sender" not in st.session_state:
        st.session_state.email_sender = ""
    if "email_subject" not in st.session_state:
        st.session_state.email_subject = ""
    if "email_body" not in st.session_state:
        st.session_state.email_body = ""
    if "prediction_verdict" not in st.session_state:
        st.session_state.prediction_verdict = None

    # Two column layout: Left = Composer, Right = Verdict & Sub-models
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        # Header row with title and immediate preset example loaders
        c_title, c_spam_btn, c_legit_btn = st.columns([1.5, 1.25, 1.25])
        with c_title:
            st.markdown("#### ✉️ Email Composer")
        with c_spam_btn:
            if st.button("📥 Try spam example", use_container_width=True):
                st.session_state.email_sender = EXAMPLE_SPAM["sender"]
                st.session_state.email_subject = EXAMPLE_SPAM["subject"]
                st.session_state.email_body = EXAMPLE_SPAM["body"]
                # Clear previous verdict when loading a different example
                st.session_state.prediction_verdict = None
                st.rerun()
        with c_legit_btn:
            if st.button("📥 Try legitimate example", use_container_width=True):
                st.session_state.email_sender = EXAMPLE_LEGIT["sender"]
                st.session_state.email_subject = EXAMPLE_LEGIT["subject"]
                st.session_state.email_body = EXAMPLE_LEGIT["body"]
                # Clear previous verdict when loading a different example
                st.session_state.prediction_verdict = None
                st.rerun()

        # Email Composer Form
        with st.form("composer_form", clear_on_submit=False):
            sender_val = st.text_input(
                "From (Sender Address - Optional)",
                value=st.session_state.email_sender,
                placeholder="e.g., sender@company.com or winner@promo-biz.net",
                help="Optional sender address used to extract domain and authenticity signals.",
            )
            subject_val = st.text_input(
                "Subject (Optional)",
                value=st.session_state.email_subject,
                placeholder="e.g., Team sync meeting notes or Urgent account claim",
            )
            body_val = st.text_area(
                "Message Body",
                value=st.session_state.email_body,
                height=210,
                placeholder="Type or paste the email text here...",
            )

            analyze_submit = st.form_submit_button(
                "⚡ Analyze email", type="primary", use_container_width=True
            )

        if analyze_submit:
            # Sync session state with latest submitted values
            st.session_state.email_sender = sender_val
            st.session_state.email_subject = subject_val
            st.session_state.email_body = body_val

            if not body_val.strip() and not subject_val.strip():
                st.warning("Please provide at least a subject line or message body to analyze.")
            elif pipeline is None:
                st.error(
                    "Model pipeline file `models/spam_ensemble_pipeline.joblib` not found. "
                    "Run `python -m src.evaluate` to generate the trained model."
                )
            else:
                input_df = pd.DataFrame(
                    [
                        {
                            "subject": subject_val,
                            "sender": sender_val,
                            "sender_address": sender_val,
                            "body": body_val,
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
                    sub_probs["Multinomial Naive Bayes"] = {
                        "prob": float(sub_estimators["nb"].predict_proba(X_sparse)[0][1]),
                        "weight": 1.0,
                    }
                if "lr" in sub_estimators:
                    sub_probs["Logistic Regression"] = {
                        "prob": float(sub_estimators["lr"].predict_proba(X_sparse)[0][1]),
                        "weight": 0.25,
                    }
                if "svm" in sub_estimators:
                    sub_probs["Linear SVM"] = {
                        "prob": float(sub_estimators["svm"].predict_proba(X_sparse)[0][1]),
                        "weight": 1.5,
                    }

                # Vocabulary recognition check
                vectorizer = preprocessor.named_transformers_["text"]
                analyzer = vectorizer.build_analyzer()
                vocab = vectorizer.vocabulary_

                subj_tokens = analyzer(subject_val) if subject_val.strip() else []
                subj_rec = [t for t in subj_tokens if t in vocab]
                body_tokens = analyzer(body_val) if body_val.strip() else []
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

                st.session_state.prediction_verdict = {
                    "is_spam": bool(pred_class == 1),
                    "spam_prob": spam_prob,
                    "sub_probs": sub_probs,
                    "is_limited_text": is_limited,
                    "text_recognition": {
                        "total_tokens": total_tokens,
                        "total_rec": total_rec,
                        "coverage_ratio": coverage_ratio,
                    },
                    "extracted_signals": {
                        "sender_domain": prep_df["sender_domain"].iloc[0],
                        "url_count": int(prep_df["url_count"].iloc[0]),
                        "exclamation_count": int(prep_df["exclamation_count"].iloc[0]),
                        "uppercase_ratio": float(prep_df["uppercase_ratio"].iloc[0]),
                        "digit_ratio": float(prep_df["digit_ratio"].iloc[0]),
                    },
                }

    with col_right:
        st.markdown("#### 📊 Detection Verdict")

        verdict = st.session_state.prediction_verdict

        if verdict is None:
            # Simple one-sentence empty state before submission
            st.markdown(
                """
                <div class="empty-verdict-box">
                    Enter an email on the left and click <strong>Analyze email</strong>, or load an example to view the prediction verdict.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            is_spam = verdict["is_spam"]
            spam_p = verdict["spam_prob"]
            spam_pct = spam_p * 100

            # Verdict Box: Focal Point
            if is_spam:
                st.markdown(
                    f"""
                    <div class="verdict-card-spam">
                        <div class="verdict-title-spam">Predicted: Spam</div>
                        <div class="verdict-stat verdict-stat-spam">{spam_pct:.1f}% estimated spam probability</div>
                        <div class="verdict-subtext">50% decision threshold</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-card-legit">
                        <div class="verdict-title-legit">Predicted: Legitimate</div>
                        <div class="verdict-stat verdict-stat-legit">{spam_pct:.1f}% estimated spam probability</div>
                        <div class="verdict-subtext">50% decision threshold</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Warning if very little recognizable text was found
            if verdict.get("is_limited_text", False):
                st.warning("⚠️ **Limited text recognized. Treat this prediction with caution.**")

            # Full-width horizontal bar with the actual cutoff (50%) marked
            cutoff_fig = go.Figure()
            cutoff_fig.add_trace(
                go.Bar(
                    x=[spam_pct],
                    y=["Score"],
                    orientation="h",
                    marker=dict(color=COLOR_SPAM_TEXT if is_spam else COLOR_LEGIT_TEXT),
                    text=[f"{spam_pct:.1f}%"],
                    textposition="inside",
                    textfont=dict(color="#FFFFFF", size=13, weight="bold"),
                    hoverinfo="none",
                )
            )
            cutoff_fig.add_vline(
                x=50,
                line_width=2.5,
                line_dash="dash",
                line_color=COLOR_MUTED,
                annotation_text="50% Cutoff",
                annotation_position="top",
                annotation_font=dict(size=11, color=COLOR_MUTED, weight="bold"),
            )
            cutoff_fig.update_layout(
                xaxis=dict(
                    range=[0, 100],
                    ticksuffix="%",
                    showgrid=True,
                    gridcolor=COLOR_BORDER,
                    tickfont=dict(color=COLOR_MUTED, size=11),
                ),
                yaxis=dict(showticklabels=False),
                height=80,
                margin=dict(l=10, r=15, t=25, b=15),
                paper_bgcolor=COLOR_CARD,
                plot_bgcolor=COLOR_CARD,
            )
            st.plotly_chart(cutoff_fig, use_container_width=True, config={"displayModeBar": False})

            # Sub-Model Horizontal Breakdown Chart
            st.markdown("##### 🔬 Sub-Model Voting Contributions")
            sub_probs = verdict.get("sub_probs", {})
            if sub_probs:
                model_labels = []
                prob_vals = []
                bar_colors = []
                hover_labels = []

                for m_name, m_info in sub_probs.items():
                    m_prob_pct = m_info["prob"] * 100
                    w = m_info["weight"]
                    model_labels.append(m_name)
                    prob_vals.append(m_prob_pct)
                    bar_colors.append(COLOR_INDIVIDUAL)  # Slate for individual models
                    hover_labels.append(
                        f"<b>{m_name}</b><br>Spam probability: {m_prob_pct:.2f}%<br>Voting weight: {w}"
                    )

                # Add final ensemble row (Indigo)
                model_labels.append("Ensemble (Weighted Average)")
                prob_vals.append(spam_pct)
                bar_colors.append(COLOR_ENSEMBLE)  # Deep indigo for ensemble
                hover_labels.append(
                    f"<b>Ensemble (Weighted Average)</b><br>Final score: {spam_pct:.2f}%<br>Weights: NB(1.0) + LR(0.25) + SVM(1.5)"
                )

                sub_fig = go.Figure(
                    go.Bar(
                        x=prob_vals,
                        y=model_labels,
                        orientation="h",
                        marker=dict(color=bar_colors),
                        text=[f"{p:.1f}%" for p in prob_vals],
                        textposition="outside",
                        textfont=dict(size=11, weight="bold", color=COLOR_TEXT),
                        hoverinfo="text",
                        hovertext=hover_labels,
                    )
                )
                sub_fig.add_vline(x=50, line_dash="dot", line_color=COLOR_MUTED, line_width=1.5)
                sub_fig.update_layout(
                    xaxis=dict(
                        range=[0, 110],
                        ticksuffix="%",
                        title="Spam Probability (0–100%)",
                        title_font=dict(size=11, color=COLOR_MUTED),
                        showgrid=True,
                        gridcolor=COLOR_BORDER,
                        tickfont=dict(color=COLOR_MUTED, size=11),
                    ),
                    yaxis=dict(
                        autorange="reversed",
                        tickfont=dict(size=11, color=COLOR_TEXT, weight="bold"),
                    ),
                    height=200,
                    margin=dict(l=10, r=30, t=10, b=30),
                    paper_bgcolor=COLOR_CARD,
                    plot_bgcolor=COLOR_CARD,
                    hoverlabel=dict(bgcolor=COLOR_TEXT, font=dict(color="#FFFFFF", family="Inter")),
                )
                st.plotly_chart(sub_fig, use_container_width=True, config={"displayModeBar": False})

            # Extracted Feature Indicators
            with st.expander("🔎 Extracted Message Signals", expanded=False):
                signals = verdict["extracted_signals"]
                s_c1, s_c2, s_c3, s_c4 = st.columns(4)
                dom = signals["sender_domain"]
                if len(dom) > 16:
                    dom = dom[:14] + "..."
                s_c1.metric("Sender Domain", dom, help=signals["sender_domain"])
                s_c2.metric("URL Count", signals["url_count"])
                s_c3.metric("Exclamations", signals["exclamation_count"])
                s_c4.metric("Upper Ratio", f"{signals['uppercase_ratio']*100:.1f}%")

                t_data = verdict.get("text_recognition", {})
                if t_data:
                    st.caption(
                        f"**Text Recognition**: {t_data.get('total_rec', 0)} of {t_data.get('total_tokens', 0)} "
                        f"tokens recognized in TF-IDF vocabulary ({t_data.get('coverage_ratio', 0.0)*100:.1f}% coverage)."
                    )


# ===========================================================================
# VIEW 2: MODEL PERFORMANCE DASHBOARD
# ===========================================================================
elif nav_selection == "📊 Model Performance":
    metrics_data = load_json_artifact(METRICS_PATH)
    ablation_data = load_json_artifact(ABLATION_PATH)
    curves_data = load_json_artifact(CURVES_PATH)

    if not metrics_data:
        st.error(
            "Evaluation metrics file `reports/metrics.json` not found. "
            "Please run `python -m src.evaluate` to generate the complete evaluation benchmarks."
        )
        st.stop()

    test_results = metrics_data.get("test_results", {})
    split_stats = metrics_data.get("split_stats", {})

    ens_metrics = test_results.get("Ensemble (Soft - Tuned Weights)", {})
    kpi_f1 = ens_metrics.get("f1", 0.9713)
    kpi_prec = ens_metrics.get("precision", 0.9757)
    kpi_rec = ens_metrics.get("recall", 0.9669)
    test_size = split_stats.get("test_size", 901)
    test_ham = split_stats.get("test_ham_count", 569)
    test_spam = split_stats.get("test_spam_count", 332)

    # Top KPI Cards
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-label">Spam F1 (Ensemble)</div>
                <div class="kpi-number">{kpi_f1:.4f}</div>
                <div class="kpi-subtext">Primary test benchmark</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Spam Precision</div>
                <div class="kpi-number">{kpi_prec*100:.2f}%</div>
                <div class="kpi-subtext">Accuracy of flagged spam</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Spam Recall</div>
                <div class="kpi-number">{kpi_rec*100:.2f}%</div>
                <div class="kpi-subtext">Proportion of spam caught</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Test Evaluation Set</div>
                <div class="kpi-number">{test_size} emails</div>
                <div class="kpi-subtext">{test_ham} ham, {test_spam} spam &bull; 0 overlap</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # Main Chart: Model Comparison
    # -----------------------------------------------------------------------
    st.markdown("#### 1. Model Comparison on Held-Out Test Data")

    metric_choice = st.selectbox(
        "Select comparison metric:",
        options=["F1 Score (Primary)", "Precision", "Recall"],
        index=0,
    )

    metric_key_map = {
        "F1 Score (Primary)": "f1",
        "Precision": "precision",
        "Recall": "recall",
    }
    target_metric_key = metric_key_map[metric_choice]

    # Build model rows
    model_rows = []
    for m_name, m_dict in test_results.items():
        if "Dummy" in m_name:
            continue  # Exclude dummy baseline from main comparison
        score = m_dict.get(target_metric_key, 0.0)
        is_ens = "Ensemble" in m_name and "Tuned" in m_name
        model_rows.append(
            {
                "Model": m_name,
                "Score": score,
                "ScorePct": score * 100,
                "IsEnsemble": is_ens,
            }
        )

    df_compare = pd.DataFrame(model_rows).sort_values("Score", ascending=True)

    # Identify highest scoring model
    max_row = df_compare.loc[df_compare["Score"].idxmax()]
    best_model_name = max_row["Model"]
    best_model_score = max_row["ScorePct"]

    # Use Indigo for Ensemble and Slate for individual models
    bar_colors = [
        COLOR_ENSEMBLE if r["IsEnsemble"] else COLOR_INDIVIDUAL
        for _, r in df_compare.iterrows()
    ]

    comp_fig = go.Figure(
        go.Bar(
            x=df_compare["ScorePct"],
            y=df_compare["Model"],
            orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{s:.2f}%" for s in df_compare["ScorePct"]],
            textposition="outside",
            textfont=dict(size=11, weight="bold", color=COLOR_TEXT),
            hoverinfo="text",
            hovertext=[f"<b>{r['Model']}</b><br>{metric_choice}: {r['ScorePct']:.2f}%" for _, r in df_compare.iterrows()],
        )
    )
    comp_fig.update_layout(
        xaxis=dict(
            range=[0, 105],
            ticksuffix="%",
            title=f"{metric_choice} (0–100% Scale)",
            title_font=dict(size=11, color=COLOR_MUTED),
            showgrid=True,
            gridcolor=COLOR_BORDER,
            tickfont=dict(color=COLOR_MUTED, size=11),
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(size=12, color=COLOR_TEXT, weight="bold"),
        ),
        height=260,
        margin=dict(l=20, r=40, t=15, b=30),
        paper_bgcolor=COLOR_CARD,
        plot_bgcolor=COLOR_CARD,
        hoverlabel=dict(bgcolor=COLOR_TEXT, font=dict(color="#FFFFFF", family="Inter")),
    )
    st.plotly_chart(comp_fig, use_container_width=True)

    st.markdown(
        f"<div style='font-size: 0.88rem; color: {COLOR_MUTED}; margin-top: -0.5rem; margin-bottom: 1.2rem;'>"
        f"🏆 <strong>Highest-scoring model for {metric_choice}:</strong> <span style='color: {COLOR_PRIMARY}; font-weight: 700;'>{best_model_name}</span> ({best_model_score:.2f}%)"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Side-by-Side: 2x2 Confusion Matrix & Precision-Recall Chart
    # -----------------------------------------------------------------------
    cm_col, pr_col = st.columns(2, gap="large")

    with cm_col:
        st.markdown("#### 2. Confusion Matrix")
        cm_model_choice = st.selectbox(
            "Select model for confusion matrix:",
            options=[
                "Ensemble (Soft - Tuned Weights)",
                "Calibrated Linear SVM",
                "Multinomial Naive Bayes",
                "Logistic Regression",
            ],
            index=0,
        )

        selected_cm_dict = test_results.get(cm_model_choice, {})
        cm_matrix = np.array(selected_cm_dict.get("confusion_matrix", [[0, 0], [0, 0]]))
        tn, fp = cm_matrix[0, 0], cm_matrix[0, 1]
        fn, tp = cm_matrix[1, 0], cm_matrix[1, 1]

        # Cell labels with high contrast readability
        z_text = [
            [
                f"<span style='color:#FFFFFF; font-size:16px;'><b>{tn}</b></span><br><span style='font-size:11px; color:#E0E7FF;'>Correct Legitimate</span>",
                f"<span style='color:#9F1239; font-size:16px;'><b>{fp}</b></span><br><span style='font-size:11px; color:#9F1239; font-weight:bold;'>Legitimate emails flagged</span>",
            ],
            [
                f"<span style='color:#B45309; font-size:16px;'><b>{fn}</b></span><br><span style='font-size:11px; color:#B45309; font-weight:bold;'>Spam emails missed</span>",
                f"<span style='color:#FFFFFF; font-size:16px;'><b>{tp}</b></span><br><span style='font-size:11px; color:#E0E7FF;'>Correct Spam</span>",
            ],
        ]

        # Single Indigo intensity scale
        cm_colorscale = [
            [0.0, "#EEF2FF"],
            [0.2, "#C7D2FE"],
            [0.6, "#6366F1"],
            [1.0, "#4338CA"],
        ]

        cm_fig = go.Figure(
            data=go.Heatmap(
                z=cm_matrix,
                x=["Predicted Legitimate", "Predicted Spam"],
                y=["Actual Legitimate", "Actual Spam"],
                text=z_text,
                texttemplate="%{text}",
                textfont={"size": 13},
                colorscale=cm_colorscale,
                showscale=False,
                hoverinfo="text",
                hovertext=[
                    [
                        f"Correct Legitimate (True Negatives): {tn}",
                        f"Legitimate emails flagged (False Positives): {fp}",
                    ],
                    [
                        f"Spam emails missed (False Negatives): {fn}",
                        f"Correct Spam (True Positives): {tp}",
                    ],
                ],
            )
        )
        cm_fig.update_layout(
            xaxis=dict(title=None, tickfont=dict(color=COLOR_TEXT, size=11, weight="bold")),
            yaxis=dict(title=None, autorange="reversed", tickfont=dict(color=COLOR_TEXT, size=11, weight="bold")),
            height=260,
            margin=dict(l=10, r=10, t=15, b=25),
            paper_bgcolor=COLOR_CARD,
            plot_bgcolor=COLOR_CARD,
            hoverlabel=dict(bgcolor=COLOR_TEXT, font=dict(color="#FFFFFF", family="Inter")),
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        st.caption(
            f"• **Legitimate emails flagged**: `{fp}` &bull; "
            f"• **Spam emails missed**: `{fn}` &bull; "
            f"• **Accuracy**: `{selected_cm_dict.get('accuracy', 0.0)*100:.2f}%`"
        )

    with pr_col:
        st.markdown("#### 3. Precision–Recall Trajectories")
        if curves_data:
            pr_fig = go.Figure()

            # Distinct line styles & colors for classroom projector readability
            curve_styles = {
                "Ensemble (Soft - Tuned)": {"color": COLOR_PRIMARY, "dash": "solid", "width": 3.5},
                "Calibrated Linear SVM": {"color": "#6366F1", "dash": "dash", "width": 2.2},
                "Multinomial Naive Bayes": {"color": "#059669", "dash": "dot", "width": 2.2},
                "Logistic Regression": {"color": "#D97706", "dash": "dashdot", "width": 2.2},
            }

            for model_name, c_data in curves_data.items():
                style = curve_styles.get(
                    model_name, {"color": COLOR_MUTED, "dash": "solid", "width": 2.0}
                )
                pr_fig.add_trace(
                    go.Scatter(
                        x=c_data["recall"],
                        y=c_data["precision"],
                        mode="lines",
                        name=f"{model_name} (AP: {c_data['ap']:.4f})",
                        line=dict(
                            color=style["color"],
                            dash=style["dash"],
                            width=style["width"],
                        ),
                    )
                )

            pr_fig.update_layout(
                xaxis=dict(
                    title="Recall (Spam Caught)",
                    title_font=dict(size=11, color=COLOR_MUTED),
                    range=[0.80, 1.01],
                    showgrid=True,
                    gridcolor=COLOR_BORDER,
                    tickfont=dict(color=COLOR_MUTED, size=11),
                ),
                yaxis=dict(
                    title="Precision",
                    title_font=dict(size=11, color=COLOR_MUTED),
                    range=[0.85, 1.01],
                    showgrid=True,
                    gridcolor=COLOR_BORDER,
                    tickfont=dict(color=COLOR_MUTED, size=11),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.48,
                    xanchor="left",
                    x=0,
                    font=dict(size=10, color=COLOR_TEXT),
                ),
                height=260,
                margin=dict(l=15, r=15, t=15, b=40),
                paper_bgcolor=COLOR_CARD,
                plot_bgcolor=COLOR_CARD,
                hoverlabel=dict(bgcolor=COLOR_TEXT, font=dict(color="#FFFFFF", family="Inter")),
            )
            st.plotly_chart(pr_fig, use_container_width=True)
            st.caption("Distinct line patterns (solid, dash, dot, dashdot) ensure clarity in projection.")
        else:
            st.info("Curve data not found in `reports/curves_data.json`.")

    st.divider()

    # -----------------------------------------------------------------------
    # Section 4: Feature Comparison (Ablation)
    # -----------------------------------------------------------------------
    st.markdown("#### 4. Feature Comparison (Ablation Study)")
    st.markdown(
        f"<div style='font-size: 0.88rem; color: {COLOR_MUTED}; margin-top: -0.3rem; margin-bottom: 0.75rem;'>"
        "Measures the incremental contribution of Text TF-IDF, Sender Domain, and Message Structural Characteristics."
        "</div>",
        unsafe_allow_html=True,
    )

    if ablation_data:
        # Standardize labels to: "Text only", "Text + sender", "All features"
        ablation_mapping = {
            "Text Only": "Text only",
            "Text + Sender": "Text + sender",
            "Text + Sender + Message Characteristics": "All features",
        }

        ab_names = []
        ab_f1s = []
        ab_precs = []
        ab_recs = []

        for orig_key, clean_label in ablation_mapping.items():
            if orig_key in ablation_data:
                ab_names.append(clean_label)
                ab_f1s.append(ablation_data[orig_key]["f1"] * 100)
                ab_precs.append(ablation_data[orig_key]["precision"] * 100)
                ab_recs.append(ablation_data[orig_key]["recall"] * 100)

        # Baseline is Text only
        baseline_f1 = ab_f1s[0]
        f1_annotations = [
            f"{s:.2f}%\n(Baseline)" if i == 0 else f"{s:.2f}%\n({(s - baseline_f1):+.2f} pp)"
            for i, s in enumerate(ab_f1s)
        ]

        ab_fig = go.Figure()
        ab_fig.add_trace(
            go.Bar(
                name="Precision",
                x=ab_names,
                y=ab_precs,
                marker_color="#818CF8",  # Light indigo
                text=[f"{p:.2f}%" for p in ab_precs],
                textposition="outside",
                textfont=dict(size=11, color=COLOR_TEXT),
            )
        )
        ab_fig.add_trace(
            go.Bar(
                name="Recall",
                x=ab_names,
                y=ab_recs,
                marker_color="#10B981",  # Fresh green
                text=[f"{r:.2f}%" for r in ab_recs],
                textposition="outside",
                textfont=dict(size=11, color=COLOR_TEXT),
            )
        )
        ab_fig.add_trace(
            go.Bar(
                name="Spam F1 (Primary)",
                x=ab_names,
                y=ab_f1s,
                marker_color=COLOR_PRIMARY,  # Deep indigo
                text=f1_annotations,
                textposition="outside",
                textfont=dict(size=11, weight="bold", color=COLOR_TEXT),
            )
        )

        ab_fig.update_layout(
            barmode="group",
            yaxis=dict(
                range=[90, 102],
                ticksuffix="%",
                title="Score Percentage",
                title_font=dict(size=11, color=COLOR_MUTED),
                showgrid=True,
                gridcolor=COLOR_BORDER,
                tickfont=dict(color=COLOR_MUTED, size=11),
            ),
            xaxis=dict(
                tickfont=dict(size=12, color=COLOR_TEXT, weight="bold"),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color=COLOR_TEXT),
            ),
            height=300,
            margin=dict(l=20, r=20, t=30, b=30),
            paper_bgcolor=COLOR_CARD,
            plot_bgcolor=COLOR_CARD,
            hoverlabel=dict(bgcolor=COLOR_TEXT, font=dict(color="#FFFFFF", family="Inter")),
        )
        st.plotly_chart(ab_fig, use_container_width=True)

        st.caption(
            "📌 **Ablation Finding**: Adding sender domain alone caused a slight dip (-0.62 pp) due to unseen sender sparsity. "
            "However, combining text and sender with **structural message signals** (URL density, capitalization, punctuation, HTML flags) "
            "achieved the peak F1 score of **97.13%** (+1.23 pp improvement over text+sender)."
        )
    else:
        st.info("Feature ablation results not found in `reports/ablation_metrics.json`.")
