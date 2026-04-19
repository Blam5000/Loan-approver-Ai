# 💳 LoanIQ — AI Loan Approval Predictor

A machine learning web app that predicts whether a loan application should be approved or rejected based on an applicant's financial profile. Built with XGBoost and deployed with Streamlit.

**🔗 Live App:** [Click here to try it](YOUR_STREAMLIT_URL)

---

## What it does

You enter three things — your income, credit score, and how much you want to borrow — and the model gives you an instant decision. It also tells you how confident it is, what factors drove the decision, and if you're rejected, what you can do to improve your chances.

It's not just a prediction. It's an explanation.

---

## Why I built this

I wanted to understand how financial institutions use machine learning to make lending decisions — and more importantly, whether those decisions are explainable. Most people get rejected for loans with no real reason given. This project was my attempt to build something that not only predicts outcomes but shows its work.

---

## How it works

The model is an **XGBoost Gradient Boosting Classifier** — an algorithm that builds hundreds of decision trees sequentially, with each tree correcting the mistakes of the one before it. It was trained on real loan application data with three core features:

- `credit_score` — the applicant's FICO score
- `income` — annual gross income
- `loan_amount` — total amount requested

After training, the model achieved **93% accuracy** on unseen test data. I also ran an overfit diagnostic — comparing training accuracy vs test accuracy — and found a gap of **2.37%**, which is within the healthy range. To get there I tuned hyperparameters including `max_depth`, `subsample`, and L1/L2 regularization.

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| XGBoost | Gradient boosting model |
| scikit-learn | Train/test split, cross validation, metrics |
| Streamlit | Web app frontend |
| joblib | Saving and loading the trained model |
| pandas / numpy | Data handling |

---

## Running it locally

Clone the repo and install dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-ai.git
cd loan-approval-ai
pip install -r requirements.txt
```

Train the model:
```bash
python train_model_fixed.py
```

Launch the app:
```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

---

## What I learned

Going into this I thought getting a high accuracy score was the goal. I got 95% on my first run and thought I was done. Then I learned about overfitting — the model had essentially memorized the training data rather than learning from it. The real work started there: diagnosing the gap, understanding why `max_depth=50` was the main culprit, and systematically fixing it with regularization and subsampling.

The final model trades a small amount of raw accuracy for something more valuable — it actually generalizes to loan applicants it has never seen before.

---

## Project structure

```
loan-approval-ai/
├── app.py                      # Streamlit web app
├── train_model_fixed.py        # Model training script with overfit fixes
├── loan_approval_model_xgb.pkl # Saved trained model
├── Datset_goofy.py             # Dataset loader
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

*Built by [Your Name] · 2026*
