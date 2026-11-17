"""Generate an academic presentation for Batch 15: Spam Email Detection Using Ensemble Learning.
Uses python-pptx to generate a widescreen (16:9) presentation with embedded charts and exact metrics.
"""

from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# Paths
REPORTS_DIR = Path("reports/figures")
PPTX_PATH = Path("Spam_Email_Detection_Batch15_Presentation.pptx")

# 16:9 Dimensions
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

# Color Palette Constants
COLOR_BG_IVORY = RGBColor(247, 245, 240)    # #F7F5F0
COLOR_CARD_WHITE = RGBColor(255, 255, 255)  # #FFFFFF
COLOR_PRIMARY_INDIGO = RGBColor(67, 56, 202)# #4338CA
COLOR_CHARCOAL = RGBColor(31, 36, 48)       # #1F2430
COLOR_SLATE = RGBColor(86, 96, 113)         # #566071
COLOR_BORDER = RGBColor(222, 218, 210)      # #DEDAD2
COLOR_MINT = RGBColor(16, 185, 129)         # #10B981
COLOR_CORAL = RGBColor(225, 29, 72)         # #E11D48
COLOR_LIGHT_BG = RGBColor(238, 242, 255)    # #EEF2FF


def create_deck():
    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT
    blank_layout = prs.slide_layouts[6]  # Blank slide

    def add_header(slide, title_text, category_text="Introduction to Machine Learning [21CSE305P] • Batch 15 • CTECH Department"):
        # Header background banner
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.9))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(9.5)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_PRIMARY_INDIGO
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(20)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_CHARCOAL

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    
    # Background card
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_WIDTH, SLIDE_HEIGHT)
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG_IVORY
    bg.line.color.rgb = COLOR_BORDER

    # Center card
    card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(0.8), Inches(11.333), Inches(5.9))
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD_WHITE
    card.line.color.rgb = COLOR_BORDER

    tb = s1.shapes.add_textbox(Inches(1.5), Inches(1.1), Inches(10.333), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "Introduction to Machine Learning [21CSE305P] • Batch 15 • CTECH Department"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    p2 = tf.add_paragraph()
    p2.text = "Spam Email Detection Using Ensemble Learning"
    p2.font.size = Pt(28)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_CHARCOAL
    
    p3 = tf.add_paragraph()
    p3.text = "A Multi-Feature Soft-Voting Architecture (Naive Bayes + Logistic Regression + Calibrated Linear SVM)"
    p3.font.size = Pt(13)
    p3.font.color.rgb = COLOR_SLATE

    # Team Members Box
    tb_team = s1.shapes.add_textbox(Inches(1.5), Inches(3.6), Inches(10.333), Inches(2.8))
    tf_team = tb_team.text_frame
    tf_team.word_wrap = True
    
    p_th = tf_team.paragraphs[0]
    p_th.text = "PROJECT TEAM MEMBERS"
    p_th.font.size = Pt(11)
    p_th.font.bold = True
    p_th.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    members = [
        ("Aman Kumar Chouhan", "RA2411003011899"),
        ("Karthick P", "RA2411003011910"),
        ("Shri Muhilathithan V U", "RA2411003011920"),
        ("Midhun A", "RA2411003012595"),
    ]
    
    for name, reg in members:
        pm = tf_team.add_paragraph()
        pm.text = f"•  {name:<26}  {reg}"
        pm.font.size = Pt(13)
        pm.font.bold = True
        pm.font.color.rgb = COLOR_CHARCOAL

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & OBJECTIVE
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement & Project Objectives")
    
    # Left Card: Problem Statement
    c_left = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c_left.fill.solid()
    c_left.fill.fore_color.rgb = COLOR_CARD_WHITE
    c_left.line.color.rgb = COLOR_BORDER
    
    tb_l = s2.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "PROBLEM STATEMENT"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    bullets_l = [
        ("Asymmetric Cost Hazard:", "In spam detection, False Positives (dropping legitimate business or personal emails) carry catastrophic cost compared to False Negatives."),
        ("Feature Brittleness:", "Text-only spam filters are vulnerable to keyword obfuscation, synonym swapping, and zero-day phrasing."),
        ("Probability Calibration:", "Individual classifiers (like raw Linear SVM or Naive Bayes) either lack true probabilities or output over-confident extreme distributions.")
    ]
    for title, desc in bullets_l:
        p_t = tf_l.add_paragraph()
        p_t.text = f"• {title}"
        p_t.font.size = Pt(11.5)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_CHARCOAL
        
        p_d = tf_l.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = COLOR_SLATE

    # Right Card: Project Objectives & Approach
    c_right = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c_right.fill.solid()
    c_right.fill.fore_color.rgb = COLOR_CARD_WHITE
    c_right.line.color.rgb = COLOR_BORDER
    
    tb_r = s2.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.1), Inches(4.8))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "PROJECT OBJECTIVE & ARCHITECTURE"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    bullets_r = [
        ("Multi-Modal Signal Fusion:", "Extract text TF-IDF (10,000 unigrams/bigrams), sender domain metadata, and 9 structural message characteristics."),
        ("Soft-Voting Ensemble:", "Combine Multinomial Naive Bayes (w=1.0), Logistic Regression (w=0.25), and Platt-Calibrated Linear SVM (w=1.5)."),
        ("Tuned Thresholding:", "Enforce a strict 50% probability cutoff with cross-validation weight optimization to maximize Spam F1 while minimizing False Positives."),
        ("Interactive Demo Dashboard:", "Deliver a production-ready Streamlit app with single-message scoring and transparent sub-model probability breakdown.")
    ]
    for title, desc in bullets_r:
        p_t = tf_r.add_paragraph()
        p_t.text = f"• {title}"
        p_t.font.size = Pt(11.5)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_CHARCOAL
        
        p_d = tf_r.add_paragraph()
        p_d.text = f"   {desc}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 3: DATASET OVERVIEW & DEDUPLICATION AUDIT
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Dataset Overview, Curation & Deduplication")
    
    # Left specs card
    c_s3_l = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c_s3_l.fill.solid()
    c_s3_l.fill.fore_color.rgb = COLOR_CARD_WHITE
    c_s3_l.line.color.rgb = COLOR_BORDER
    
    tb_s3 = s3.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf_s3 = tb_s3.text_frame
    tf_s3.word_wrap = True
    
    p = tf_s3.paragraphs[0]
    p.text = "DATASET SPECIFICATIONS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    items = [
        ("Data Sources:", "Standardized Enron Email + Ling-Spam + SpamAssassin benchmarks."),
        ("Total Records:", "4,533 verified unique email records."),
        ("Class Distribution:", "2,722 Legitimate (60.1%) | 1,811 Spam (39.9%)."),
        ("Train / Test Split:", "80:20 Stratified Partition (3,632 Train / 901 Test)."),
        ("Test Set Composition:", "569 Legitimate (63.15%) | 332 Spam (36.85%)."),
        ("Deduplication Protocol:", "Exact and near-duplicate text-hash grouping. ZERO duplicate group overlap between train and test splits to guarantee unbiased evaluation.")
    ]
    for k, v in items:
        p_k = tf_s3.add_paragraph()
        p_k.text = f"• {k} {v}"
        p_k.font.size = Pt(11)
        p_k.font.color.rgb = COLOR_CHARCOAL
    
    # Right Image
    img_dist = REPORTS_DIR / "class_distribution.png"
    if img_dist.exists():
        s3.shapes.add_picture(str(img_dist), Inches(6.8), Inches(1.5), width=Inches(5.7))

    # =========================================================================
    # SLIDE 4: FEATURE ENGINEERING PIPELINE
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Multi-Feature Engineering Pipeline")
    
    # 3 Column Cards
    col_w = Inches(3.65)
    gap = Inches(0.35)
    left_start = Inches(0.8)
    
    f_groups = [
        ("1. TEXT TF-IDF", "Message Body + Subject", [
            "Word unigrams + bigrams (1 to 2 n-grams)",
            "Top 10,000 vocabulary features",
            "Sublinear term frequency scaling (1 + log(tf))",
            "Sublinear scaling dampens repetitive spam keywords",
            "Standard English stopword removal"
        ]),
        ("2. SENDER METADATA", "Sender Address / Domain", [
            "Domain isolation (e.g. lottery-promo.biz)",
            "High-frequency sender domain One-Hot Encoding",
            "Missing sender boolean flag (missing_sender=1)",
            "Captures spoofed or omitted From headers common in phishing emails",
            "Unknown domain handling during live inference"
        ]),
        ("3. STRUCTURAL SIGNALS", "9 Numeric Engineered Features", [
            "url_count: Total hyperlinks in body",
            "has_html: Presence of raw HTML tags",
            "uppercase_ratio: Proportion of CAPITALIZED chars",
            "digit_ratio: Proportion of digits (codes, numbers)",
            "exclamation_count: Total '!' characters",
            "question_count: Total '?' characters",
            "MinMaxScaler: Scaled to [0, 1] for Naive Bayes"
        ])
    ]
    
    for i, (head, sub, pts) in enumerate(f_groups):
        cx = left_start + i * (col_w + gap)
        card = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, Inches(1.5), col_w, Inches(5.3))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_WHITE
        card.line.color.rgb = COLOR_BORDER
        
        tb = s4.shapes.add_textbox(cx + Inches(0.2), Inches(1.7), col_w - Inches(0.4), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = head
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY_INDIGO
        
        ps = tf.add_paragraph()
        ps.text = sub
        ps.font.size = Pt(10)
        ps.font.color.rgb = COLOR_SLATE
        
        for pt in pts:
            pp = tf.add_paragraph()
            pp.text = f"• {pt}"
            pp.font.size = Pt(10.5)
            pp.font.color.rgb = COLOR_CHARCOAL

    # =========================================================================
    # SLIDE 5: WHY AN ENSEMBLE MODEL?
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Why Soft-Voting Ensemble? Model Complementarity")
    
    # Left: Explanation table / bullets
    c_s5_l = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c_s5_l.fill.solid()
    c_s5_l.fill.fore_color.rgb = COLOR_CARD_WHITE
    c_s5_l.line.color.rgb = COLOR_BORDER
    
    tb_s5 = s5.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf_s5 = tb_s5.text_frame
    tf_s5.word_wrap = True
    
    p = tf_s5.paragraphs[0]
    p.text = "INDIVIDUAL MODEL STRENGTHS & LIMITATIONS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    submodels = [
        ("Multinomial Naive Bayes (Weight: 1.0)", "Strength: Superb keyword frequency sensitivity, high recall.\nLimitation: Naive independence assumption leads to extreme over-confident probabilities (near 0 or 1)."),
        ("Logistic Regression (Weight: 0.25)", "Strength: Excellent log-odds calibration, robust linear baseline.\nLimitation: Sensitive to dense collinear features; lower standalone recall (94.88%)."),
        ("Calibrated Linear SVM (Weight: 1.5)", "Strength: Highest standalone precision (97.84%), maximum-margin geometric separation in high-dimensional sparse TF-IDF.\nLimitation: Standard SVM outputs margins, not probabilities; resolved via Platt Sigmoid Calibration.")
    ]
    for m_title, m_desc in submodels:
        pt = tf_s5.add_paragraph()
        pt.text = f"• {m_title}"
        pt.font.size = Pt(11)
        pt.font.bold = True
        pt.font.color.rgb = COLOR_CHARCOAL
        
        pd = tf_s5.add_paragraph()
        pd.text = f"   {m_desc}"
        pd.font.size = Pt(10)
        pd.font.color.rgb = COLOR_SLATE
    
    # Right: Formula card
    c_s5_r = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c_s5_r.fill.solid()
    c_s5_r.fill.fore_color.rgb = COLOR_CARD_WHITE
    c_s5_r.line.color.rgb = COLOR_BORDER
    
    tb_form = s5.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.1), Inches(4.8))
    tf_form = tb_form.text_frame
    tf_form.word_wrap = True
    
    pf = tf_form.paragraphs[0]
    pf.text = "SOFT-VOTING MATHEMATICAL FORMULATION"
    pf.font.size = Pt(12)
    pf.font.bold = True
    pf.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    form_bullets = [
        ("Platt Sigmoid Calibration for Linear SVM:", "P(y=1 | f) = 1 / (1 + exp(A·f + B))\nWhere f is the signed decision function value, and A, B are fitted via cross-validation log-loss."),
        ("Weighted Soft-Voting Decision Rule:", "P_Ensemble(Spam | x) = (1.0·P_NB + 0.25·P_LR + 1.5·P_SVM) / 2.75"),
        ("Decision Cutoff Rule:", "Class = SPAM if P_Ensemble >= 0.50 else LEGITIMATE"),
        ("Cross-Validation F1 Gain:", "Tuned weights [1.0, 0.25, 1.5] achieved 0.9765 CV F1 compared to 0.9738 for equal weights [1.0, 1.0, 1.0].")
    ]
    for ft, fd in form_bullets:
        pft = tf_form.add_paragraph()
        pft.text = f"• {ft}"
        pft.font.size = Pt(11)
        pft.font.bold = True
        pft.font.color.rgb = COLOR_CHARCOAL
        
        pfd = tf_form.add_paragraph()
        pfd.text = f"   {fd}"
        pfd.font.size = Pt(10)
        pfd.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 6: FEATURE GROUP ABLATION STUDY
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Feature Ablation Analysis — Structural Signal Gains")
    
    # Left: Explanation of ablation
    c_s6_l = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c_s6_l.fill.solid()
    c_s6_l.fill.fore_color.rgb = COLOR_CARD_WHITE
    c_s6_l.line.color.rgb = COLOR_BORDER
    
    tb_s6 = s6.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf_s6 = tb_s6.text_frame
    tf_s6.word_wrap = True
    
    p = tf_s6.paragraphs[0]
    p.text = "ABLATION TIERS & INSIGHTS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    ab_notes = [
        ("Tier 1: Text Only (Baseline)", "F1 = 0.9652 | Precision = 0.9696 | Recall = 0.9608\n10 False Positives, 13 False Negatives."),
        ("Tier 2: Text + Sender Domain", "F1 = 0.9590 (-0.62 pp)\nSlight dip caused by unseen sender domain sparsity in the test partition."),
        ("Tier 3: All Features (Text + Sender + 9 Structural Signals)", "F1 = 0.9713 (+1.23 pp Gain) | Accuracy = 97.89%\nReduced False Positives to 8 and False Negatives to 11."),
        ("Key Defense Takeaway:", "Structural features (URL counts, HTML markers, capitalization ratios) bridge domain sparsity, propelling the model to peak test performance.")
    ]
    for k, v in ab_notes:
        pk = tf_s6.add_paragraph()
        pk.text = f"• {k}"
        pk.font.size = Pt(11)
        pk.font.bold = True
        pk.font.color.rgb = COLOR_CHARCOAL
        
        pv = tf_s6.add_paragraph()
        pv.text = f"   {v}"
        pv.font.size = Pt(10)
        pv.font.color.rgb = COLOR_SLATE

    # Right: Ablation Graph
    img_ab = REPORTS_DIR / "feature_ablation.png"
    if img_ab.exists():
        s6.shapes.add_picture(str(img_ab), Inches(6.8), Inches(1.5), width=Inches(5.7))

    # =========================================================================
    # SLIDE 7: MODEL BENCHMARK & COMPARATIVE EVALUATION
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Model Performance Comparison on Held-Out Test Set (N=901)")
    
    # Embed model comparison image
    img_comp = REPORTS_DIR / "model_comparison_presentation.png"
    if not img_comp.exists():
        img_comp = REPORTS_DIR / "model_comparison.png"
    if img_comp.exists():
        s7.shapes.add_picture(str(img_comp), Inches(0.8), Inches(1.5), width=Inches(7.2))

    # Right Summary Card
    c_s7_r = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.3), Inches(1.5), Inches(4.2), Inches(5.3))
    c_s7_r.fill.solid()
    c_s7_r.fill.fore_color.rgb = COLOR_CARD_WHITE
    c_s7_r.line.color.rgb = COLOR_BORDER
    
    tb_s7 = s7.shapes.add_textbox(Inches(8.5), Inches(1.7), Inches(3.8), Inches(4.8))
    tf_s7 = tb_s7.text_frame
    tf_s7.word_wrap = True
    
    p = tf_s7.paragraphs[0]
    p.text = "OFFICIAL HELD-OUT RESULTS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    benchmarks = [
        ("Tuned Soft Ensemble (WINNER):", "Spam F1: 0.9713 (97.13%)\nPrecision: 97.57%\nRecall: 96.69%\nAccuracy: 97.89%"),
        ("Calibrated Linear SVM:", "Spam F1: 0.9665\nPrecision: 97.84% (Highest Prec)\nRecall: 95.48%"),
        ("Multinomial Naive Bayes:", "Spam F1: 0.9611\nPrecision: 95.54%\nRecall: 96.69%"),
        ("Logistic Regression:", "Spam F1: 0.9574\nPrecision: 96.63%\nRecall: 94.88%"),
        ("Majority Dummy Baseline:", "Spam F1: 0.0000\nAccuracy: 63.15%")
    ]
    for b_title, b_score in benchmarks:
        pbt = tf_s7.add_paragraph()
        pbt.text = f"• {b_title}"
        pbt.font.size = Pt(11)
        pbt.font.bold = True
        pbt.font.color.rgb = COLOR_CHARCOAL
        
        pbs = tf_s7.add_paragraph()
        pbs.text = f"   {b_score}"
        pbs.font.size = Pt(10)
        pbs.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 8: CONFUSION MATRIX & ERROR TRADEOFF ANALYSIS
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Confusion Matrix & Error Trade-off Analysis")
    
    # Left: Confusion Matrix Graph
    img_cm = REPORTS_DIR / "cm_soft_ensemble.png"
    if not img_cm.exists():
        img_cm = REPORTS_DIR / "confusion_matrix_ensemble.png"
    if img_cm.exists():
        s8.shapes.add_picture(str(img_cm), Inches(0.8), Inches(1.5), width=Inches(5.6))
        
    # Right: Error Pattern Breakdown
    c_s8_r = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c_s8_r.fill.solid()
    c_s8_r.fill.fore_color.rgb = COLOR_CARD_WHITE
    c_s8_r.line.color.rgb = COLOR_BORDER
    
    tb_s8 = s8.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.1), Inches(4.8))
    tf_s8 = tb_s8.text_frame
    tf_s8.word_wrap = True
    
    p = tf_s8.paragraphs[0]
    p.text = "ERROR DISTRIBUTION ANALYSIS (N=901 TEST SAMPLES)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    cm_items = [
        ("True Negatives (TN = 561):", "561 legitimate emails correctly routed to inbox (98.6% specificity)."),
        ("False Positives (FP = 8) — CRITICAL TYPE I ERROR:", "Only 8 legitimate emails incorrectly flagged as spam (1.4% false alarm rate). Keeps critical correspondence safe."),
        ("False Negatives (FN = 11) — TYPE II ERROR:", "11 spam messages slipped past the filter (3.3% miss rate)."),
        ("True Positives (TP = 321):", "321 spam emails caught and blocked (96.7% recall rate)."),
        ("Operational Assessment:", "Ensemble tuning successfully suppressed False Positives to 8, striking the optimal industry trade-off for automated inbox filtering.")
    ]
    for k, v in cm_items:
        pk = tf_s8.add_paragraph()
        pk.text = f"• {k}"
        pk.font.size = Pt(11)
        pk.font.bold = True
        pk.font.color.rgb = COLOR_CHARCOAL
        
        pv = tf_s8.add_paragraph()
        pv.text = f"   {v}"
        pv.font.size = Pt(10)
        pv.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 9: PRECISION-RECALL & ROC TRAJECTORIES
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Precision-Recall & Receiver Operating Characteristic Curves")
    
    # Two images side by side
    img_pr = REPORTS_DIR / "precision_recall_curves.png"
    img_roc = REPORTS_DIR / "roc_curves.png"
    
    if img_pr.exists():
        s9.shapes.add_picture(str(img_pr), Inches(0.8), Inches(1.5), width=Inches(5.7))
    if img_roc.exists():
        s9.shapes.add_picture(str(img_roc), Inches(6.8), Inches(1.5), width=Inches(5.7))

    # =========================================================================
    # SLIDE 10: STREAMLIT DASHBOARD ARCHITECTURE
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Interactive Streamlit Demonstration Dashboard")
    
    # 2 Column Cards
    c10_l = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c10_l.fill.solid()
    c10_l.fill.fore_color.rgb = COLOR_CARD_WHITE
    c10_l.line.color.rgb = COLOR_BORDER
    
    tb10_l = s10.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf10_l = tb10_l.text_frame
    tf10_l.word_wrap = True
    
    p = tf10_l.paragraphs[0]
    p.text = "VIEW 1: ANALYZE EMAIL COMPOSER"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    pts_v1 = [
        ("Interactive Composer:", "From, Subject, and Body fields with instant preset buttons: 'Try spam example' & 'Try legitimate example'."),
        ("Focal Verdict Card:", "Displays high-visibility 'Predicted: Spam' (Soft Rose #FFF1F2) or 'Predicted: Legitimate' (Soft Green #F0FDF4)."),
        ("Cutoff Gauge Bar:", "Full-width probability bar highlighting distance from 50% threshold."),
        ("Sub-Model Contribution Chart:", "Live horizontal bar chart showing individual probabilities for Naive Bayes, Logistic Regression, Linear SVM, and Ensemble."),
        ("Out-of-Vocabulary Detector:", "Warns if unrecognizable gibberish input has insufficient TF-IDF coverage.")
    ]
    for k, v in pts_v1:
        pk = tf10_l.add_paragraph()
        pk.text = f"• {k}"
        pk.font.size = Pt(11)
        pk.font.bold = True
        pk.font.color.rgb = COLOR_CHARCOAL
        
        pv = tf10_l.add_paragraph()
        pv.text = f"   {v}"
        pv.font.size = Pt(10)
        pv.font.color.rgb = COLOR_SLATE

    # Right Card: View 2
    c10_r = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c10_r.fill.solid()
    c10_r.fill.fore_color.rgb = COLOR_CARD_WHITE
    c10_r.line.color.rgb = COLOR_BORDER
    
    tb10_r = s10.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.1), Inches(4.8))
    tf10_r = tb10_r.text_frame
    tf10_r.word_wrap = True
    
    p = tf10_r.paragraphs[0]
    p.text = "VIEW 2: MODEL PERFORMANCE DASHBOARD"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    pts_v2 = [
        ("Top KPI Metric Cards:", "Live display of Spam F1 (0.9713), Precision (97.57%), Recall (96.69%), and Test Set Size (901 emails)."),
        ("Model Benchmark Comparison:", "Interactive bar chart switchable across F1, Precision, and Recall."),
        ("Dynamic Confusion Matrix:", "Interactive model switcher displaying Plain-English error categories ('Legitimate emails flagged' & 'Spam emails missed')."),
        ("PR Trajectories & Ablation Bars:", "Embedded interactive Plotly curves matching the warm ivory/white dashboard theme."),
        ("Zero Data Storage Privacy:", "All inferences execute strictly in-memory without persistent logging.")
    ]
    for k, v in pts_v2:
        pk = tf10_r.add_paragraph()
        pk.text = f"• {k}"
        pk.font.size = Pt(11)
        pk.font.bold = True
        pk.font.color.rgb = COLOR_CHARCOAL
        
        pv = tf10_r.add_paragraph()
        pv.text = f"   {v}"
        pv.font.size = Pt(10)
        pv.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 11: CONCLUSION & FUTURE SCOPE
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Conclusion, Key Takeaways & Future Directions")
    
    # Left Card: Conclusions
    c11_l = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))
    c11_l.fill.solid()
    c11_l.fill.fore_color.rgb = COLOR_CARD_WHITE
    c11_l.line.color.rgb = COLOR_BORDER
    
    tb11_l = s11.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(5.0), Inches(4.8))
    tf11_l = tb11_l.text_frame
    tf11_l.word_wrap = True
    
    p = tf11_l.paragraphs[0]
    p.text = "CORE SUMMARY & KEY TAKEAWAYS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    conc_pts = [
        ("Superiority of Ensembles:", "Combining Naive Bayes, Logistic Regression, and Calibrated Linear SVM outperformed every individual baseline, achieving 0.9713 Spam F1 and 97.89% Accuracy."),
        ("Critical False Positive Suppression:", "Platt sigmoid calibration paired with tuned voting weights restricted False Positives to only 8 out of 569 test emails (98.6% inbox specificity)."),
        ("Structural Features Overcome Sparsity:", "Integrating 9 engineered structural signals with TF-IDF boosted F1 by +1.23 pp, proving that structural indicators (URLs, HTML, capitalization) are vital defenses against text evasion."),
        ("Production Deployment Ready:", "The full pipeline is serialized as a single artifact (`spam_ensemble_pipeline.joblib`) powering a responsive web app.")
    ]
    for k, v in conc_pts:
        pk = tf11_l.add_paragraph()
        pk.text = f"• {k}"
        pk.font.size = Pt(11)
        pk.font.bold = True
        pk.font.color.rgb = COLOR_CHARCOAL
        
        pv = tf11_l.add_paragraph()
        pv.text = f"   {v}"
        pv.font.size = Pt(10)
        pv.font.color.rgb = COLOR_SLATE

    # Right Card: Future Scope
    c11_r = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))
    c11_r.fill.solid()
    c11_r.fill.fore_color.rgb = COLOR_CARD_WHITE
    c11_r.line.color.rgb = COLOR_BORDER
    
    tb11_r = s11.shapes.add_textbox(Inches(7.1), Inches(1.7), Inches(5.1), Inches(4.8))
    tf11_r = tb11_r.text_frame
    tf11_r.word_wrap = True
    
    p = tf11_r.paragraphs[0]
    p.text = "FUTURE RESEARCH & EXTENSIONS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    fut_pts = [
        ("Transformer Language Models:", "Incorporate lightweight contextual embeddings (DistilBERT or RoBERTa) to detect semantic paraphrase spam and modern AI-generated phishing."),
        ("Domain Reputation Feeds:", "Integrate live DNSBL (DNS-based Blackhole List) and WHOIS domain age lookups for real-time sender authenticity verification."),
        ("Adversarial Robustness Testing:", "Stress-test the ensemble against homoglyph attacks, invisible zero-width Unicode characters, and prompt injection."),
        ("Client-Side Email Plugins:", "Package the inference pipeline as an IMAP/Outlook add-in for on-device edge classification.")
    ]
    for k, v in fut_pts:
        pk = tf11_r.add_paragraph()
        pk.text = f"• {k}"
        pk.font.size = Pt(11)
        pk.font.bold = True
        pk.font.color.rgb = COLOR_CHARCOAL
        
        pv = tf11_r.add_paragraph()
        pv.text = f"   {v}"
        pv.font.size = Pt(10)
        pv.font.color.rgb = COLOR_SLATE

    # =========================================================================
    # SLIDE 12: THANK YOU / Q&A
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    bg12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SLIDE_WIDTH, SLIDE_HEIGHT)
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = COLOR_BG_IVORY
    bg12.line.color.rgb = COLOR_BORDER

    card12 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(1.2), Inches(10.333), Inches(5.1))
    card12.fill.solid()
    card12.fill.fore_color.rgb = COLOR_CARD_WHITE
    card12.line.color.rgb = COLOR_BORDER

    tb12 = s12.shapes.add_textbox(Inches(2.0), Inches(1.6), Inches(9.333), Inches(4.3))
    tf12 = tb12.text_frame
    tf12.word_wrap = True
    
    p = tf12.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = "Presentation Concluded"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY_INDIGO
    
    p2 = tf12.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.text = "THANK YOU"
    p2.font.size = Pt(36)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_CHARCOAL
    
    p3 = tf12.add_paragraph()
    p3.alignment = PP_ALIGN.CENTER
    p3.text = "Questions & Discussion"
    p3.font.size = Pt(18)
    p3.font.color.rgb = COLOR_SLATE

    p_line = tf12.add_paragraph()
    p_line.alignment = PP_ALIGN.CENTER
    p_line.text = "──────────────────────────────────────────────"
    p_line.font.size = Pt(12)
    p_line.font.color.rgb = COLOR_BORDER

    p_team_h = tf12.add_paragraph()
    p_team_h.alignment = PP_ALIGN.CENTER
    p_team_h.text = "Batch 15 • Department of Computational Technologies (CTECH)"
    p_team_h.font.size = Pt(12)
    p_team_h.font.bold = True
    p_team_h.font.color.rgb = COLOR_CHARCOAL

    p_names = tf12.add_paragraph()
    p_names.alignment = PP_ALIGN.CENTER
    p_names.text = "Aman Kumar Chouhan (RA2411003011899) • Karthick P (RA2411003011910)\nShri Muhilathithan V U (RA2411003011920) • Midhun A (RA2411003012595)"
    p_names.font.size = Pt(11)
    p_names.font.color.rgb = COLOR_SLATE

    # Save presentation
    prs.save(str(PPTX_PATH))
    print(f"Presentation saved successfully to: {PPTX_PATH.resolve()}")

if __name__ == "__main__":
    create_deck()
