# Spam Email Detection Using Ensemble Learning (Batch 15)

A complete, runnable academic machine learning project implementing a supervised binary email classifier to distinguish spam from legitimate email (ham). Features a soft-voting ensemble combining **Multinomial Naive Bayes**, **Logistic Regression**, and **Calibrated Linear Support Vector Classifier** on text, sender domain metadata, and structural message characteristics.

---

## 📌 Project Overview

- **Corpus**: Apache SpamAssassin Public Corpus (4,533 cleaned emails: 2,722 Ham, 1,811 Spam).
- **Architecture**: Soft-Voting Ensemble ($w = [1.0, 0.25, 1.5]$) over 5,135 non-negative sparse features.
- **Primary Metric**: **Spam $F_1$ Score** ($0.9713$), Accuracy ($97.89\%$), ROC-AUC ($0.9968$).
- **Anti-Leakage Design**: Group-aware stratified splitting preventing near-duplicate leakage (0 group overlap) and strict pipeline encapsulation.

---

## 📂 Project Structure

```
mlproject_mail/
│
├── data/
│   ├── raw/                 # Downloaded raw email corpus (gitignored)
│   ├── processed/           # Cleaned tabular dataset (emails_cleaned.csv)
│   └── fixtures/            # Offline synthetic fixtures (synthetic_emails.csv)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Email parsing, MIME decoding, cleaning, deduplication
│   ├── features.py          # TF-IDF, sender OHE, non-negative structural feature pipelines
│   ├── models.py            # Model definitions, calibrated SVM, soft voting, weight tuning
│   └── evaluate.py          # 5-fold CV, test evaluation, ablation, metrics & plot generation
│
├── models/
│   ├── spam_ensemble_pipeline.joblib   # Trained production pipeline artifact
│   └── metadata.json                   # Pipeline parameters and metadata
│
├── reports/
│   ├── figures/             # 7 presentation-ready PNG evaluation plots
│   ├── metrics.json         # Complete JSON evaluation metrics
│   ├── metrics.csv          # Comparative metrics table across all models
│   ├── ablation_metrics.json# Feature ablation metrics
│   └── error_analysis.json  # Qualitative inspection of false positives & negatives
│
├── tests/
│   └── test_pipeline.py     # 10 automated pytest unit & integration tests
│
├── app.py                   # Streamlit web demonstration application
├── predict.py               # Command-line prediction tool
├── REPORT.md                # Detailed academic project report with equations
├── VIVA_GUIDE.md            # Comprehensive viva voce defense guide & talking points
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quickstart & Windows PowerShell Setup

Open a **Windows PowerShell** terminal inside the project directory:

### 1. Environment Setup
```powershell
# Optional: Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Downloads (if needed) the Apache SpamAssassin corpus, extracts RFC 822/2822 emails, strips headers, deduplicates, and saves the cleaned tabular dataset:
```powershell
python -m src.dataset
```
*Note*: If external network access is unavailable, the pipeline automatically falls back to the included synthetic fixture in `data/fixtures/` or accepts locally placed archives in `data/raw/`.

### 3. Model Training, Tuning, and Evaluation
Executes 5-fold group-aware cross-validation, tunes voting weights, evaluates all models on the held-out test partition, runs feature ablation, generates 7 publication figures in `reports/figures/`, and serializes the production model to `models/`:
```powershell
python -m src.evaluate
```

### 4. Interactive Web Demonstration (Streamlit)
Launch the local viva demonstration dashboard:
```powershell
streamlit run app.py
```
Then navigate to `http://localhost:8501` in your browser. Use the preset dropdown to load sample legitimate emails, urgent lottery spam, or loan offers, or input custom messages.

### 5. Command-Line Prediction (CLI)
Classify individual emails directly via the command line:
```powershell
# Predict Spam email
python predict.py --subject "URGENT: Claim $1,000,000 lottery cash prize now!!!" --sender "winner@claim-prize-free.biz" --body "Congratulations! You won big money. Click http://claim-prize-free.biz/win to collect cash!"

# Predict Legitimate (Ham) email
python predict.py --subject "Quarterly team sync notes and roadmap" --sender "manager@engineering.org" --body "Hi team, attached is the presentation for our roadmap planning meeting tomorrow at 10 AM. Please review the slides."

# Output as JSON
python predict.py --subject "Meeting" --body "Let's sync up tomorrow" --json
```

### 6. Run Test Suite
Execute the 10 automated unit and integration tests:
```powershell
python -m pytest -v tests/test_pipeline.py
```

---

## 📊 Measured Test Evaluation Results

Evaluated on the held-out test partition ($N = 901$, 569 Ham, 332 Spam) with zero near-duplicate group leakage:

| Model | Accuracy | Spam Precision | Spam Recall | Spam $F_1$ Score | ROC-AUC | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|
| **Dummy (Majority)** | 63.15% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0.3685 | 0 | 332 |
| **Multinomial Naive Bayes** | 97.11% | 95.54% | **96.69%** | 0.9611 | **0.9970** | **0.9946** | 15 | **11** |
| **Logistic Regression** | 96.89% | 96.63% | 94.88% | 0.9574 | 0.9946 | 0.9899 | 11 | 17 |
| **Calibrated Linear SVM** | 97.56% | **0.9784%** | 95.48% | 0.9665 | 0.9956 | 0.9924 | **7** | 15 |
| **Ensemble (Soft - Equal)** | 97.45% | 96.40% | **96.69%** | 0.9654 | 0.9966 | 0.9937 | 12 | **11** |
| **Ensemble (Soft - Tuned)** | **97.89%** | 97.57% | **96.69%** | **0.9713** | 0.9968 | 0.9941 | 8 | **11** |

### Feature Ablation Comparison
- **Text Only**: $F_1 = 0.9652$ (FP = 10, FN = 13)
- **Text + Sender**: $F_1 = 0.9590$ (FP = 11, FN = 16)
- **Text + Sender + Message Characteristics**: **$F_1 = 0.9713$** (FP = 8, FN = 11)

---

## 🎓 Viva Voce & Presentation Defense

Review `VIVA_GUIDE.md` for complete talking points, equations, and defense strategies covering:
1. Why each estimator was selected (generative, discriminative, max-margin).
2. The mathematical formulation of Soft Voting and Platt sigmoid calibration.
3. How data leakage was strictly prevented via group-aware splitting and encapsulated pipelines.
4. How to interpret the confusion matrix and why False Positives are disproportionately costly.
