# ================= Loan Default Predictor (Dark Theme, Polished) =================
# Imports
import streamlit as st
import pandas as pd
import joblib, json
from pathlib import Path

# ------- MUST BE FIRST STREAMLIT CALL -------
st.set_page_config(page_title="Loan Default Predictor", page_icon="💳", layout="centered")

# ------- Global Dark Theme (black / grey, consistent) -------
st.markdown("""
<style>
:root{
  --bg:#0B0B0C;         /* app background (near black) */
  --surface:#151618;    /* cards/inputs background (dark grey) */
  --border:#2A2C2F;     /* thin borders */
  --text:#F2F4F7;       /* main text (near white) */
  --text-2:#C9CDD2;     /* secondary text (soft grey) */
  --muted:#8A8F98;      /* helper text */
  --good:#19A974;       /* green accent */
  --risk:#D64545;       /* red accent */
  --btn:#2F3136;        /* button bg */
  --btn-h:#3A3D43;      /* button hover */

  /* Footer variables (match theme) */
  --footer-bg: var(--bg);
  --footer-text: var(--text);
  --footer-border: var(--border);
  --footer-link: var(--text-2);
  --footer-link-hover: var(--text);
}
* { letter-spacing: .1px; }
.stApp{ background:var(--bg); color:var(--text); font-family: 'Segoe UI',system-ui,-apple-system,Roboto,Arial,sans-serif; }
.block-container{ max-width:1100px; padding-top:.75rem; }

/* Headings */
.app-title{ font-weight:800; font-size:40px; margin:.75rem 0 .25rem; color:var(--text); }
.app-caption{ color:var(--text-2); font-size:15px; margin-bottom:.25rem; }
.section-title{ font-weight:700; color:var(--text-2); margin:.5rem 0 .35rem; }

/* Sidebar */
[data-testid="stSidebar"]{ background:var(--surface); color:var(--text); border-right:1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span{ color:var(--text-2)!important; }
[data-testid="stSidebar"] .stSlider > div{ color:var(--text); }

/* Inputs */
.stNumberInput input, .stSelectbox div[data-baseweb="select"]>div{
  background:var(--bg); color:var(--text);
  border:1px solid var(--border); border-radius:10px;
}
.stNumberInput label, .stSelectbox label{ color:var(--text-2); font-weight:600; }

/* Buttons */
.stButton>button{
  background:var(--btn); color:var(--text);
  border:1px solid var(--border); border-radius:10px;
  font-weight:700; padding:.6rem 1rem;
}
.stButton>button:hover{ background:var(--btn-h); }

/* Cards & metrics */
.card{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:16px 18px; }
[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
[data-testid="stMetricValue"]{ color:var(--text); font-weight:800; }

/* Result grid (big, readable) */
.result-grid{ display:grid; grid-template-columns:1.2fr 1fr 1fr; gap:16px; align-items:center; }
.result-title{ font-weight:600; color:var(--text-2); }
.result-value{ margin:.25rem 0 0; font-size:32px; font-weight:800; }
@media (max-width: 900px){
  .result-grid{ grid-template-columns:1fr; }
}

/* Status tags */
.tag{ display:inline-flex; align-items:center; gap:10px; padding:8px 12px; border-radius:12px; font-weight:800; letter-spacing:.3px; }
.tag.good{ background:rgba(25,169,116,.12); color:#CFF6E9; border:1px solid rgba(25,169,116,.35); }
.tag.risk{ background:rgba(214,69,69,.12);  color:#FFD5D5; border:1px solid rgba(214,69,69,.35); }

/* Tables & code */
.stCode, .stDataFrame{ border:1px solid var(--border); border-radius:10px; }

/* Footer */
hr.theme-hr{ border:none; border-top:1px solid var(--border); margin-top:2rem; margin-bottom:.5rem; }
.footer{
  display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px;
  background: var(--footer-bg)!important;
  border: 1px solid var(--footer-border)!important;
  border-radius:12px; padding:14px 18px;
  color: var(--footer-text)!important; text-align:center;
}
.footer .footer-grid{
  display:grid; grid-template-columns: repeat(2, auto);
  column-gap:28px; row-gap:6px; justify-content:center; align-items:center;
}
.footer .item{ white-space:nowrap; }
@media (max-width:520px){
  .footer .footer-grid{ grid-template-columns:1fr; }
  .footer .item{ white-space:normal; }
}
.footer a{ color: var(--footer-link)!important; text-decoration:none; }
.footer a:hover{ color: var(--footer-link-hover)!important; text-decoration:underline; }
.footer .brand{ font-size:.95rem; font-weight:700; }
.footer .copy{ font-size:.9rem; color:var(--muted); }
</style>
""", unsafe_allow_html=True)

# ------- Load artifacts -------
@st.cache_resource
def load_artifacts():
    # Use relative paths so it works on any system
    pipe_path = Path("artifacts/loan_default_lr_pipeline.pkl")
    meta_path = Path("artifacts/loan_default_lr_metadata.json")

    pipeline = joblib.load(pipe_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = load_artifacts()
DEFAULT_THR = float(meta.get("threshold", 0.5))
NUM_COLS = list(map(str, meta.get("numeric_features", [])))
CAT_COLS = list(map(str, meta.get("categorical_features", [])))

# ------- Header -------
st.markdown('<div class="app-title">💳 Loan Default Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-caption">Predicting loan default risk using behavioral, demographic, and financial features. '
    'This app deploys the production-ready pipeline (preprocessing + SMOTE-balanced Logistic Regression) '
    'to support data-driven credit decisions and reduce portfolio losses.</div>',
    unsafe_allow_html=True
)

# ------- Sidebar (Threshold) -------
with st.sidebar:
    st.markdown("### ⚙️ Decision Threshold")
    thr = st.slider("Classify as default if probability ≥ threshold", 0.05, 0.95, DEFAULT_THR, 0.01)
    st.caption(f"Saved threshold: **{DEFAULT_THR:.2f}**")

# ------- Prediction helper -------
def predict_df(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    probs = pipeline.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    out = df.copy()
    out["proba_default"] = probs
    out["pred_default"] = preds
    return out

# ------- Tabs -------
tab_single, tab_batch = st.tabs(["🔹 Single Applicant", "📄 Batch via CSV"])

# ===== TAB 1: Single Applicant =====
with tab_single:
    st.markdown('<div class="section-title">Enter Applicant Details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        loanamount = st.number_input("Loan Amount", 0.0, 1e7, 5000.0, 500.0)
        termdays = st.number_input("Term Days", 1, 365, 30, 1)
        loannumber = st.number_input("Loan Number", 1, 20, 1, 1)
        approved_hour = st.number_input("Approved Hour (0–23)", 0, 23, 12, 1)
        avg_interest_amount = st.number_input("Average Interest Amount", 0.0, 1e6, 200.0, 10.0)
        avg_daily_repayment_amount = st.number_input("Avg Daily Repayment Amount", 0.0, 1e5, 150.0, 5.0)
        loan_to_term_ratio = st.number_input("Loan-to-Term Ratio", 0.0, 1e5, 166.7, 1.0)
    with c2:
        estimated_income = st.number_input("Estimated Income", 0.0, 1e7, 60000.0, 1000.0)
        debt_to_income = st.number_input("Debt-to-Income (0–1)", 0.0, 5.0, 0.35, 0.01)
        loan_to_income_ratio = st.number_input("Loan-to-Income Ratio", 0.0, 5.0, 0.08, 0.01)
        avg_credit_score = st.number_input("Average Credit Score", 0.0, 1000.0, 650.0, 5.0)
        age = st.number_input("Age", 18, 100, 32, 1)
        bank_account_type = st.selectbox("Bank Account Type", ["Savings", "Current", "Other", "Unknown"])
        employment_status_clients = st.selectbox("Employment Status", ["Permanent","Self-employed","Student","Unemployed","Retired","Contract","Unknown"])
        approved_weekday = st.selectbox("Approved Weekday", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        bank_name_clients = st.selectbox("Bank Name", ["GT Bank","First Bank","Access Bank","UBA","Zenith Bank","Diamond Bank","Stanbic IBTC","Skye Bank","Sterling Bank","Union Bank","Heritage Bank","Keystone Bank","Unity Bank","Unknown"])

    single_row = pd.DataFrame([{
        "loanamount": loanamount, "termdays": termdays, "loannumber": loannumber, "approved_hour": approved_hour,
        "avg_interest_amount": avg_interest_amount, "avg_daily_repayment_amount": avg_daily_repayment_amount,
        "loan_to_term_ratio": loan_to_term_ratio, "estimated_income": estimated_income, "debt_to_income": debt_to_income,
        "loan_to_income_ratio": loan_to_income_ratio, "avg_credit_score": avg_credit_score, "age": int(age),
        "bank_account_type": bank_account_type, "employment_status_clients": employment_status_clients,
        "approved_weekday": approved_weekday, "bank_name_clients": bank_name_clients
    }])

    # Remember last prediction in session
    if "score" not in st.session_state:
        st.session_state.score = None

    # Predict button
    if st.button("Predict Default"):
        res = predict_df(single_row, thr).iloc[0]
        st.session_state.score = {
            "prob": float(res["proba_default"]),
            "pred": int(res["pred_default"]),
            "thr_used": float(thr),
        }

    # Render result card
    s = st.session_state.score
    prob_txt = f"{s['prob']:.3f}" if s else "—"
    thr_txt = f"{thr:.2f}"
    if s:
        tag_cls = "risk" if s["pred"] == 1 else "good"
        tag_txt = "DEFAULT (1)" if s["pred"] == 1 else "NO DEFAULT (0)"
        prediction_html = f"<div class='tag {tag_cls}'>{tag_txt}</div>"
    else:
        prediction_html = "<div class='tag' style='background:var(--surface);border:1px solid var(--border);color:var(--text-2)'>No prediction yet</div>"

    st.markdown(f"""
    <div class="card">
      <div class="result-grid">
        <div>
          <div class="result-title">Probability of Default</div>
          <div class="result-value">{prob_txt}</div>
        </div>
        <div>
          <div class="result-title">Threshold</div>
          <div class="result-value">{thr_txt}</div>
        </div>
        <div>
          <div class="result-title">Prediction</div>
          <div style="margin-top:.35rem">{prediction_html}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ===== TAB 2: Batch CSV =====
with tab_batch:
    st.markdown('<div class="section-title">Upload CSV</div>', unsafe_allow_html=True)
    st.caption("CSV must contain the same feature names used during training.")

    sample = pd.DataFrame([{
        "loanamount": 5000, "termdays": 30, "loannumber": 1, "approved_hour": 12,
        "avg_interest_amount": 200, "avg_daily_repayment_amount": 150, "loan_to_term_ratio": 166.7,
        "estimated_income": 60000, "debt_to_income": 0.35, "loan_to_income_ratio": 0.08,
        "avg_credit_score": 650, "age": 32,
        "bank_account_type": "Savings", "employment_status_clients": "Permanent",
        "approved_weekday": "Wednesday", "bank_name_clients": "GT Bank"
    }])
    st.download_button("Download sample CSV", data=sample.to_csv(index=False), file_name="sample_input.csv")

    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df_in = pd.read_csv(file)
        st.write("Preview:", df_in.head())
        scored = predict_df(df_in, thr)
        st.success("Scoring complete.")
        st.write(scored.head())
        st.download_button("Download results CSV", data=scored.to_csv(index=False), file_name="predictions.csv")

# ------- Footer -------
st.markdown("""
<hr class="theme-hr">
<div class="footer">
  <div class="footer-grid">
    <div class="item"><strong>Built by:</strong> Euba Morenikeji Ibilola</div>
    <div class="item"><strong>Email:</strong> <a href="mailto:Morenikejieuba@gmail.com">Morenikejieuba@gmail.com</a></div>
    <div class="item"><strong>Phone:</strong> <a href="tel:+393513493155">+39 351 349 3155</a></div>
    <div class="item"><strong>LinkedIn:</strong> <a href="https://www.linkedin.com/in/morenikeji-euba-92a125190/" target="_blank" rel="noopener noreferrer">morenikeji-euba-92a125190</a></div>
  </div>
  <div class="brand">Loan Default Predictor</div>
  <div class="copy">© 2025 • Risk Analytics • All rights reserved</div>
</div>
""", unsafe_allow_html=True)
