# Credit Risk Prediction

A machine learning web app that predicts whether a loan applicant is a **good** or **bad** credit risk, built with scikit-learn and Streamlit.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Banks face significant financial risk when approving loans to applicants who may default. This project builds a classification model trained on historical German credit data to help assess applicant risk — returning either **Good** (likely to repay) or **Bad** (risk of default).

---

## Demo

**Live App:** https://credit-risk-prediction-dg6ud7pszj5ifwxujh9aql.streamlit.app/

Enter applicant details in the sidebar → click **Predict Credit Risk** → view prediction with confidence score.

---

## Dataset

**German Credit Dataset** — 1,000 loan records from a German bank.

| Feature | Description |
|---|---|
| Age | Applicant age (years) |
| Sex | male / female |
| Job | Skill level: 0 = unskilled non-resident, 1 = unskilled resident, 2 = skilled, 3 = highly skilled |
| Housing | own / rent / free |
| Saving accounts | little / moderate / quite rich / rich |
| Checking account | little / moderate / rich |
| Credit amount | Loan amount in Deutsche Marks (DM) |
| Duration | Loan term in months |
| Risk | **Target** — good / bad |

After removing rows with missing values: **522 usable records**.

---

## Approach

### Models Evaluated

All models were tuned with 5-fold `GridSearchCV` and `class_weight="balanced"` to handle class imbalance (~56% good, ~44% bad after cleaning).

| Model | Test Accuracy |
|---|---|
| Decision Tree | 58.1% |
| Random Forest | 61.9% |
| **Extra Trees** | **64.8%** |
| XGBoost | 68.6% |

### Why Extra Trees?

Although XGBoost achieved higher test accuracy (68.6%), **Extra Trees was selected** because:

1. **Small dataset:** Only 522 samples after dropping missing values. Extra Trees' random splitting provides stronger regularization and reduces overfitting risk compared to XGBoost's boosting.
2. **Stability:** More consistent cross-validation scores across folds, indicating better generalization.
3. **Speed:** Significantly faster training with competitive performance.

The ~4% accuracy gap is a worthwhile trade-off for a more robust model on this dataset size.

---

## Project Structure

```
credit-risk-prediction/
├── analysis_model.ipynb          # Full EDA + modeling notebook
├── app.py                        # Streamlit web application
├── german_credit_data.csv        # Raw dataset
├── extra_trees_credit_model.pkl  # Trained Extra Trees model
├── Sex_encoder.pkl               # Label encoders for categorical features
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
├── target_encoder.pkl
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/dewwathanyu-creator/credit-risk-prediction.git
cd credit-risk-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **Note:** The trained model files (`.pkl`) are included in the repo so you can run the app directly. To retrain the model, run all cells in `analysis_model.ipynb`.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.13 | Core language |
| pandas | Data manipulation |
| scikit-learn | Preprocessing, modeling, evaluation |
| XGBoost | Gradient boosting baseline |
| matplotlib / seaborn | Visualization |
| Streamlit | Web application |
| joblib | Model serialization |

---

## License

MIT License — free to use and modify.
