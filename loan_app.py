import streamlit as st
import joblib as jwb
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ · AI Approval Engine",
    page_icon="💳",
    layout="centered",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #060910;
    color: #dde3f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 1rem 5rem; max-width: 800px; }

.hero-wrap { text-align: center; padding: 3rem 1rem 2rem; position: relative; }
.hero-glow {
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(124,58,237,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.35);
    color: #a78bfa; font-size: 11px; font-weight: 600; letter-spacing: 0.13em;
    text-transform: uppercase; padding: 5px 14px; border-radius: 100px; margin-bottom: 1.2rem;
}
.hero-chip::before {
    content: ''; width: 6px; height: 6px; background: #7c3aed;
    border-radius: 50%; display: inline-block;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
    color: #f1f5f9; line-height: 1.1; margin-bottom: 0.75rem; letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #64748b; font-size: 0.95rem; max-width: 420px; margin: 0 auto; line-height: 1.65; }
.div-line { border: none; border-top: 1px solid rgba(255,255,255,0.05); margin: 0.5rem 0 2rem; }
.sec-title {
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #475569; margin-bottom: 1rem;
}
.glass {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 1.75rem; margin-bottom: 1.25rem;
}

div[data-baseweb="input"] input {
    background: #0c1018 !important; border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-size: 15px !important; font-family: 'Inter', sans-serif !important;
}
div[data-baseweb="input"] input:focus {
    border-color: rgba(124,58,237,0.5) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;
}
label { color: #64748b !important; font-size: 12.5px !important; font-weight: 500 !important; }

.cscore-bar-bg {
    background: rgba(255,255,255,0.06); border-radius: 100px; height: 6px;
    margin: 6px 0 4px; overflow: hidden;
}
.cscore-bar-fill { height: 6px; border-radius: 100px; }

div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%);
    color: #fff; border: none; border-radius: 12px; padding: 0.9rem;
    font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700;
    letter-spacing: 0.03em; cursor: pointer; transition: all 0.2s;
    box-shadow: 0 4px 24px rgba(124,58,237,0.3); margin-top: 0.75rem;
}
div.stButton > button:hover { box-shadow: 0 6px 32px rgba(124,58,237,0.45); transform: translateY(-1px); }
div.stButton > button:active { transform: scale(0.985); }

.result-approved {
    background: linear-gradient(145deg, #021a0e, #042d17);
    border: 1px solid rgba(52,211,153,0.3); border-radius: 20px;
    padding: 2.25rem 2rem; text-align: center; margin: 1.5rem 0;
}
.result-rejected {
    background: linear-gradient(145deg, #1a0202, #2d0404);
    border: 1px solid rgba(248,113,113,0.3); border-radius: 20px;
    padding: 2.25rem 2rem; text-align: center; margin: 1.5rem 0;
}
.result-icon { font-size: 2.75rem; margin-bottom: 0.6rem; }
.result-title {
    font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800;
    margin-bottom: 0.4rem; letter-spacing: -0.01em;
}
.result-approved .result-title { color: #34d399; }
.result-rejected .result-title { color: #f87171; }
.result-sub { color: #64748b; font-size: 0.875rem; line-height: 1.6; max-width: 320px; margin: 0 auto 1rem; }
.conf-badge { display: inline-block; padding: 7px 20px; border-radius: 100px; font-size: 0.875rem; font-weight: 600; }
.conf-green { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.conf-red   { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }

.stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 1.25rem 0 0; }
.stat-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 1rem 0.75rem; text-align: center;
}
.stat-val { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700; color: #a78bfa; margin-bottom: 3px; }
.stat-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #475569; }

.feat-row { margin-bottom: 14px; }
.feat-top { display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8; margin-bottom: 5px; font-weight: 500; }
.feat-bg { background: rgba(255,255,255,0.05); border-radius: 100px; height: 5px; overflow: hidden; }
.feat-fill { height: 5px; border-radius: 100px; background: linear-gradient(90deg, #7c3aed, #a78bfa); }

.tip-card { display: flex; gap: 14px; align-items: flex-start; padding: 1.25rem 1.5rem; }
.tip-icon { font-size: 1.4rem; margin-top: 2px; }
.tip-title { font-weight: 600; color: #e2e8f0; font-size: 14px; margin-bottom: 4px; }
.tip-body  { font-size: 13px; color: #64748b; line-height: 1.65; }

.footer { text-align: center; color: #1e293b; font-size: 12px; margin-top: 4rem; padding-bottom: 1rem; }
.footer span { color: #334155; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return jwb.load('loan_approval_model_xgb.pkl')

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-glow"></div>
  <div class="hero-chip">XGBoost Classifier · AI Powered</div>
  <div class="hero-title">Loan<span>IQ</span></div>
  <div class="hero-sub">
    Enter applicant financials for an instant AI-powered loan decision
    backed by your trained XGBoost model.
  </div>
</div>
<hr class="div-line">
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ `loan_approval_model_xgb.pkl` not found — make sure it's in the same folder as `app.py`.")
    st.stop()

# ── Inputs ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="sec-title">Applicant Information</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    income = st.number_input("Annual Income ($)", min_value=0, max_value=1_000_000, value=65_000, step=1_000)
    loan_amount = st.number_input("Loan Amount Requested ($)", min_value=1_000, max_value=2_000_000, value=30_000, step=1_000)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    credit_score = st.number_input("Credit Score (300–850)", min_value=300, max_value=850, value=700, step=1)

    # Credit score bar
    cs_pct = int(((credit_score - 300) / 550) * 100)
    if credit_score >= 750:   cs_label, cs_color = "Excellent", "#34d399"
    elif credit_score >= 700: cs_label, cs_color = "Good",      "#a3e635"
    elif credit_score >= 650: cs_label, cs_color = "Fair",      "#facc15"
    elif credit_score >= 600: cs_label, cs_color = "Poor",      "#fb923c"
    else:                     cs_label, cs_color = "Very Poor", "#f87171"

    st.markdown(f"""
    <div style="margin-top:6px;">
      <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">Credit Rating</div>
      <div class="cscore-bar-bg">
        <div class="cscore-bar-fill" style="width:{cs_pct}%;background:{cs_color};"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:10px;color:#475569;margin-top:3px;">
        <span>300</span><span style="color:{cs_color};font-weight:600;">{cs_label}</span><span>850</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Live debt-to-income
    dti = round((loan_amount / income * 100), 1) if income > 0 else 0
    if dti < 28:   dti_color, dti_hint = "#34d399", "✓ Low risk"
    elif dti < 43: dti_color, dti_hint = "#facc15", "⚠ Moderate risk"
    else:          dti_color, dti_hint = "#f87171", "✗ High risk"

    st.markdown(f"""
    <div style="margin-top:16px;background:#0c1018;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:14px 16px;">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.12em;color:#475569;margin-bottom:4px;">Loan-to-Income Ratio</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:{dti_color};">{dti}%</div>
      <div style="font-size:11px;color:#475569;margin-top:3px;">{dti_hint}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Predict button ─────────────────────────────────────────────────────────────
clicked = st.button("⚡ Analyze Application", use_container_width=True)

if clicked:
    # Exact column order from training: ['loan_amount', 'credit_score', 'income']
    input_arr     = np.array([[loan_amount, credit_score, income]])
    prediction    = model.predict(input_arr)[0]
    probabilities = model.predict_proba(input_arr)[0]
    approve_pct   = round(float(probabilities[1]) * 100, 1)
    reject_pct    = round(float(probabilities[0]) * 100, 1)

    # ── Result card ────────────────────────────────────────────────────────────
    if prediction == 1:
        st.markdown(f"""
        <div class="result-approved">
          <div class="result-icon">✅</div>
          <div class="result-title">Approved</div>
          <div class="result-sub">This application meets the model's criteria based on income, credit score, and loan amount.</div>
          <span class="conf-badge conf-green">Model confidence: {approve_pct}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
          <div class="result-icon">❌</div>
          <div class="result-title">Rejected</div>
          <div class="result-sub">This application does not meet the model's approval threshold. See improvement tips below.</div>
          <span class="conf-badge conf-red">Model confidence: {reject_pct}%</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Stats row ──────────────────────────────────────────────────────────────
    monthly = round((loan_amount * 0.065 / 12) / (1 - (1 + 0.065/12)**-60), 2) if loan_amount > 0 else 0
    st.markdown(f"""
    <div class="stats-grid">
      <div class="stat-card"><div class="stat-val">{approve_pct}%</div><div class="stat-lbl">Approval odds</div></div>
      <div class="stat-card"><div class="stat-val">${monthly:,.0f}/mo</div><div class="stat-lbl">Est. payment</div></div>
      <div class="stat-card"><div class="stat-val">{dti}%</div><div class="stat-lbl">Debt-to-income</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sec-title">What drove this decision?</p>', unsafe_allow_html=True)
    feature_names = ['Loan Amount', 'Credit Score', 'Income']
    importances   = model.feature_importances_
    total         = importances.sum()
    pairs         = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    bars_html     = "".join(f"""
        <div class="feat-row">
          <div class="feat-top"><span>{n}</span><span>{round((i/total)*100,1)}%</span></div>
          <div class="feat-bg"><div class="feat-fill" style="width:{round((i/total)*100,1)}%"></div></div>
        </div>""" for n, i in pairs)
    st.markdown(f'<div class="glass">{bars_html}</div>', unsafe_allow_html=True)

    # ── Improvement tips (rejected only) ───────────────────────────────────────
    if prediction == 0:
        tips = []
        if credit_score < 700:
            tips.append(("📈", "Raise your credit score",
                f"A score of {credit_score} is below the typical 700 threshold. Paying down debt and making on-time payments can improve it in 3–6 months."))
        if dti > 43:
            tips.append(("💰", "Request a smaller loan",
                f"Your {dti}% loan-to-income ratio is high. Try requesting ${int(income * 0.35):,} or less to stay under the 35% safe zone."))
        if income < loan_amount * 0.3:
            tips.append(("💼", "Add a co-applicant",
                "Documenting a co-applicant's income or adding additional income sources could significantly strengthen this application."))

        if tips:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="sec-title">How to improve chances</p>', unsafe_allow_html=True)
            for icon, title, body in tips:
                st.markdown(f"""
                <div class="glass tip-card">
                  <div class="tip-icon">{icon}</div>
                  <div><div class="tip-title">{title}</div><div class="tip-body">{body}</div></div>
                </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built with <span>XGBoost · scikit-learn · Streamlit</span> &nbsp;·&nbsp; Trained on Loan_data_cleaned
</div>
""", unsafe_allow_html=True)