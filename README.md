<div align="center">

# Argus

Intelligent Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/MIT-License-blue?style=for-the-badge)

</div>

# Overview

Argus is an artificial neural network that flags fraudulent credit card
transactions in real time. It's built around the actual difficulty of fraud
detection — fraud makes up roughly **0.17%** of all transactions in the
dataset, so a model that always predicts "legitimate" would score 99.8%+
"accuracy" while catching zero fraud. Argus is trained and evaluated
around metrics that survive that imbalance instead.

The system:

- Trains a class-weighted neural network on anonymized transaction data
- Tunes its decision threshold against validation-set F1 instead of
  defaulting to 0.5
- Evaluates on precision, recall, F1, ROC-AUC, and PR-AUC — not accuracy
- Serves predictions through an interactive Streamlit dashboard
- Scores single transactions or full CSV batches

---

# Features

## Modeling Pipeline

| Stage | Responsibility |
|-------|-----------------|
| Data | Stratified train/validation/test split, `StandardScaler` fit on train only |
| Model | Dense ANN trained with balanced class weights (no resampling) |
| Threshold Search | Scans validation-set F1 across candidate thresholds, picks the optimum |
| Evaluation | Scores the untouched test set once, at the chosen threshold |

---

## Interactive Dashboard

The Streamlit app provides:

- Single-transaction scoring with one-click legitimate/suspicious examples
- Live fraud probability with a threshold-relative verdict card
- Batch CSV upload and scoring, with downloadable results
- A sidebar showing live test-set model metrics

---

## Results

Test-set performance at the F1-optimal threshold:

| Threshold | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-----------|-----------|--------|----------|---------|--------|
| 0.99      | 0.796     | 0.779  | 0.787    | 0.959   | 0.687  |

PR-AUC is the more informative metric here — under this severity of class
imbalance, ROC-AUC can look deceptively strong even when precision at usable
recall levels is much lower.

---

# Architecture

```text
Raw Transactions (CSV)
        │
        ▼
Preprocessing ── stratified split, StandardScaler (train-fit only)
        │
        ▼
ANN Training ── class-weighted Dense network
        │
        ▼
Threshold Search ── validation-set F1 sweep
        │
        ▼
Evaluation ── test-set precision / recall / F1 / ROC-AUC / PR-AUC
        │
        ▼
Streamlit App ── single-transaction + batch CSV scoring
```

**Network architecture:**

```text
Input(30) → Dense(64, relu) → Dropout(0.3)
          → Dense(32, relu) → Dropout(0.3)
          → Dense(16, relu)
          → Dense(1, sigmoid)
```

---

# Project Structure

```text
Argus/
│
├── app/
│   └── app.py                   # Streamlit dashboard
│
├── src/fraud_detection/
│   ├── config.py                 # paths, feature schema, constants
│   ├── data.py                   # load_dataset / split_data
│   ├── model.py                  # network architecture, class weights
│   ├── train.py                  # training entry point (CLI)
│   ├── evaluate.py               # evaluation + threshold search (CLI)
│   └── predict.py                # inference (single + batch)
│
├── notebooks/                    # EDA, preprocessing, training, evaluation walkthroughs
├── tests/                        # pytest unit tests
├── models/                       # saved model + scaler artifacts
├── results/                      # evaluation_metrics.csv
├── images/                       # ROC / PR curve plots
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/<username>/Argus.git

cd Argus
```

## Create Virtual Environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -e ".[dev]"
```

## Dataset

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle / ULB Machine Learning Group) and place it at `data/raw/creditcard.csv`.
This path is gitignored and not committed to the repo.

---

# Running the Project

## Launch the dashboard

```bash
streamlit run app/app.py
```

The application will be available at

```
http://localhost:8501
```

## Reproduce training

```bash
python -m fraud_detection.train      # trains the model, saves models/*.keras + scaler.pkl
python -m fraud_detection.evaluate   # scores the test set, saves results/ + images/
```

## Run tests

```bash
pytest
```

Unit tests cover the stratified data-splitting logic and the prediction/batch
prediction interface, with the model and scaler mocked so they run without a
full TensorFlow install or the real dataset.

---

# Technology Stack

**Modeling**

- TensorFlow / Keras
- scikit-learn

**Data**

- pandas
- NumPy

**App**

- Streamlit

**Testing / Tooling**

- pytest
- ruff

---

# Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(Kaggle / ULB Machine Learning Group) — 284,807 European cardholder
transactions from September 2013, with `Time` and `Amount` plus 28
PCA-transformed features (`V1`–`V28`) to protect confidentiality, and a
binary `Class` label (1 = fraud).

---

# Limitations & Roadmap

**Current limitations**

- The 28 PCA features are anonymized, so the model can't be interpreted in
  terms of real transaction attributes (merchant, location, etc.)
- No temporal validation — a production system would need to account for
  fraud patterns drifting over time, which a single random train/test split
  doesn't capture

**Roadmap**

- Compare against gradient-boosted trees (XGBoost / LightGBM)
- SHAP-based explainability for individual predictions
- Temporal / rolling-window validation
- Dockerized deployment
- REST API alongside the Streamlit dashboard

---

# Contributing

Contributions are welcome.

```bash
git checkout -b feature/new-feature
git commit -m "Add new feature"
git push origin feature/new-feature
```

Then open a Pull Request.

---

# License

This project is distributed under the MIT License.
