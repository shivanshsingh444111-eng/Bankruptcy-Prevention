# WHAT: Import OS module.
# WHY: To get the exact folder path of this script on Windows safely.
import os

# WHAT: Import Streamlit library.
# WHY: To build the web application dashboard without writing HTML/JS from scratch.
import streamlit as st

# WHAT: Import NumPy library.
# WHY: Needed for numerical operations and array handling.
import numpy as np

# WHAT: Import Pandas library.
# WHY: To convert the user's input values into a structured table (DataFrame) that the model expects.
import pandas as pd

# WHAT: Import Joblib library.
# WHY: To load our pre-trained machine learning model from the .pkl file into Python memory.
import joblib

# ----------------- PAGE CONFIGURATION -----------------
# WHAT: Configure browser title, icon, and wide desktop layout.
# WHY: Gives the web application a professional look and full screen width for charts.
st.set_page_config(
    page_title="Corporate Solvency & Risk Analyzer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- MODERN CUSTOM CSS -----------------
# WHAT: Custom CSS code injected via Markdown.
# WHY: To add clean fonts, glowing cards, and visual color signals (Red for danger, Green for safe).
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        border: 1px solid #334155;
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 6px;
    }
    
    .result-danger {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 10px;
        color: #991b1b;
        margin-bottom: 20px;
    }
    
    .result-safe {
        background-color: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 20px;
        border-radius: 10px;
        color: #166534;
        margin-bottom: 20px;
    }
    
    .reason-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px;
        margin-top: 15px;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #0284c7;
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0369a1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD SAVED MODEL -----------------
# WHAT: Locate the directory and the model file name.
# WHY: Avoids hardcoding C:\ paths so the app runs smoothly anywhere.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bankruptcy_model.pkl")

# WHAT: Cache resource decorator.
# WHY: Prevents reloading the model from disk on every button click, keeping the app fast.
@st.cache_resource
def load_saved_model(path):
    return joblib.load(path)

model = None

# WHAT: Check if the model file is available on disk.
# WHY: If found, load it immediately; if missing, provide an upload box so the app never crashes.
if os.path.exists(MODEL_PATH):
    try:
        model = load_saved_model(MODEL_PATH)
    except Exception as error_msg:
        st.error(f"Error loading model: {error_msg}")
        st.stop()
else:
    st.warning("⚠️ `bankruptcy_model.pkl` not found in folder. Upload it below:")
    uploaded_file = st.file_uploader("Upload model file:", type=["pkl"])
    if uploaded_file is not None:
        try:
            model = joblib.load(uploaded_file)
            st.success("Model loaded successfully!")
        except Exception as error_msg:
            st.error(f"Invalid model file: {error_msg}")
            st.stop()
    else:
        st.stop()

# ----------------- SIDEBAR EXPLANATION -----------------
# WHAT: Left sidebar content.
# WHY: Explains what 0.0, 0.5, and 1.0 mean so anyone reviewing the project understands the scoring.
with st.sidebar:
    st.markdown("### 📌 Simple Rating Scale")
    st.write("🟢 **0.0 = Strong / Low Risk**")
    st.caption("Safe condition, zero stress.")
    
    st.write("🟡 **0.5 = Average / Medium Risk**")
    st.caption("Normal market balance.")
    
    st.write("🔴 **1.0 = Weak / High Risk**")
    st.caption("Critical stress, high risk.")
    
    st.divider()
    st.markdown("### 🎯 Key Drivers")
    st.write("1. **Financial Flexibility** (Emergency Cash)")
    st.write("2. **Competitiveness** (Market Standing)")
    st.write("3. **Credibility** (Credit Trust)")

# ----------------- HERO BANNER -----------------
# WHAT: Top visual banner header.
# WHY: Introduces the project title and purpose clearly to the user.
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🏢 Corporate Bankruptcy Risk & Solvency Engine</div>
    <div style="color: #94a3b8; font-size: 14px;">
        Adjust company risk values below to evaluate solvency and understand why risk changes.
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- USER INPUT CONTROLS -----------------
# WHAT: Section title for input parameters.
# WHY: Signals to the user to choose their risk values.
st.subheader("📋 Select Evaluation Values for the Company")

# WHAT: Mapping dictionary for dropdowns.
# WHY: Lets the user see human-readable labels with color emojis, while passing clean float numbers (0.0, 0.5, 1.0) to the model.
risk_options = {
    "0.0 - Strong / Low Risk 🟢": 0.0,
    "0.5 - Average / Medium Risk 🟡": 0.5,
    "1.0 - Weak / High Risk 🔴": 1.0
}

# WHAT: Divide inputs into 3 columns.
# WHY: Makes the layout compact and clean instead of one long scrolling page.
col1, col2, col3 = st.columns(3)

with col1:
    ind_risk_label = st.selectbox(
        "1. Industrial Risk (Industry stability)",
        list(risk_options.keys()), index=0
    )
    cred_label = st.selectbox(
        "4. Credibility (Trust in paying debt)",
        list(risk_options.keys()), index=0
    )

with col2:
    mgmt_risk_label = st.selectbox(
        "2. Management Risk (Leadership quality)",
        list(risk_options.keys()), index=0
    )
    comp_label = st.selectbox(
        "5. Competitiveness (Market demand & edge)",
        list(risk_options.keys()), index=0
    )

with col3:
    fin_flex_label = st.selectbox(
        "3. Financial Flexibility (Cash reserves)",
        list(risk_options.keys()), index=0
    )
    op_risk_label = st.selectbox(
        "6. Operating Risk (Daily business stability)",
        list(risk_options.keys()), index=0
    )

# WHAT: Collect the selected numeric values in exact order.
# WHY: The machine learning model strictly requires numbers in the exact feature sequence it was trained on.
user_numeric_values = [
    risk_options[ind_risk_label],
    risk_options[mgmt_risk_label],
    risk_options[fin_flex_label],
    risk_options[cred_label],
    risk_options[comp_label],
    risk_options[op_risk_label]
]

# WHAT: Feature names matching the original dataset columns.
# WHY: Scikit-learn models trained with DataFrames warn or fail if column names are missing.
feature_names = [
    'industrial_risk',
    'management_risk',
    'financial_flexibility',
    'credibility',
    'competitiveness',
    'operating_risk'
]

# WHAT: Build DataFrame with 1 row of user inputs.
# WHY: Formats input as a 2D table ready for model.predict().
features_df = pd.DataFrame([user_numeric_values], columns=feature_names)

# WHAT: Action button to trigger prediction.
# WHY: Prevents the model from running repeatedly while the user is still selecting inputs.
st.write("")
predict_button = st.button("🔍 Check Solvency Status & Explain Reasons", use_container_width=True)

# ----------------- PREDICTION & EXPLANATION -----------------
# WHAT: Execute prediction block when button is clicked.
# WHY: Runs inference only when the user requests an evaluation.
if predict_button:
    # WHAT: Generate prediction output class.
    # WHY: Identifies whether the company falls into Bankruptcy or Non-Bankruptcy.
    prediction_result = model.predict(features_df)[0]
    is_bankrupt = str(prediction_result).lower() in ['bankruptcy', '0', 'yes', 'true']
    
    st.markdown("---")
    st.subheader("📊 Assessment Outcome")
    
    res_col1, res_col2 = st.columns([1.4, 1])
    
    with res_col1:
        # WHAT: Render outcome card based on boolean flag.
        # WHY: Gives immediate, unmistakable visual feedback (Red alert for danger, Green card for safe).
        if is_bankrupt:
            st.markdown("""
            <div class="result-danger">
                <h3 style="margin: 0; color: #b91c1c;">⚠️ High Bankruptcy Risk Detected</h3>
                <p style="margin-top: 8px; font-size: 15px; color: #7f1d1d;">
                    The company is in critical financial danger. Cash reserves and solvency safeguards are weak.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-safe">
                <h3 style="margin: 0; color: #15803d;">✅ Safe / Financially Solvent (Non-Bankruptcy)</h3>
                <p style="margin-top: 8px; font-size: 15px; color: #14532d;">
                    The company is financially healthy. Cash flows, operations, and debt coverage are stable.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
    with res_col2:
        # WHAT: Calculate prediction confidence score if model supports predict_proba.
        # WHY: Provides transparent mathematical confidence percentage to the user.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_df)[0]
            confidence_score = max(probabilities) * 100
            st.metric(label="Model Confidence Score", value=f"{confidence_score:.1f}%")
            st.progress(int(confidence_score))
        else:
            st.metric(label="Predicted Class", value="Bankruptcy" if is_bankrupt else "Non-Bankruptcy")

    # ----------------- WHY THIS RESULT OCCURRED -----------------
    # WHAT: Explainability breakdown header.
    # WHY: Bridges the gap between raw predictions and understandable business insights.
    st.subheader("💡 Why Did the Model Give This Result?")
    
    simple_labels = {
        'financial_flexibility': 'Financial Flexibility (Cash reserves)',
        'competitiveness': 'Competitiveness (Market standing)',
        'credibility': 'Credibility (Credit trust)',
        'management_risk': 'Management Risk (Leadership quality)',
        'operating_risk': 'Operating Risk (Daily running costs)',
        'industrial_risk': 'Industrial Risk (Sector conditions)'
    }
    
    danger_factors = []
    protective_factors = []
    
    # WHAT: Iterate through inputs to separate positive vs negative drivers.
    # WHY: Directly answers the question: "Which change caused the risk to go up or down?"
    for col_key, val in zip(feature_names, user_numeric_values):
        label = simple_labels[col_key]
        if val == 1.0:
            danger_factors.append(label)
        elif val == 0.0:
            protective_factors.append(label)
            
    col_reasons_1, col_reasons_2 = st.columns(2)
    
    with col_reasons_1:
        st.markdown("#### 🔴 Factors Increasing Risk:")
        if danger_factors:
            for item in danger_factors:
                st.write(f"- ❌ **{item}** is High (1.0). This damages company solvency.")
        else:
            st.write("✔️ None of the factors are at High Risk (1.0).")
            
    with col_reasons_2:
        st.markdown("#### 🟢 Factors Protecting the Company:")
        if protective_factors:
            for item in protective_factors:
                st.write(f"- ✅ **{item}** is Strong (0.0). This provides financial safety.")
        else:
            st.write("⚠️ None of the factors are set to Strong (0.0).")
            
    # WHAT: Summary cards explaining core business impact rules.
    # WHY: Clear takeaway lines that make presentation and explanation effortless.
    st.markdown("""
    <div class="reason-card">
        <h4 style="margin-top: 0; color: #0f172a;">🧠 How Factor Changes Affect the Result:</h4>
        <ul style="margin-bottom: 0; color: #334155; font-size: 14px; line-height: 1.6;">
            <li><b>Why Risk Becomes High:</b> When <b>Financial Flexibility</b> and <b>Competitiveness</b> are set to <b>1.0</b>, the company loses cash and sales power, triggering an immediate <b>Bankruptcy</b> prediction.</li>
            <li><b>Why Risk Becomes Low:</b> Changing <b>Financial Flexibility</b> and <b>Competitiveness</b> from 1.0 down to <b>0.0</b> drops risk immediately because the company has steady liquidity and revenue.</li>
            <li><b>Why Credibility Matters:</b> High <b>Credibility (0.0)</b> ensures that banks continue to support the company with credit even during operational stress.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)