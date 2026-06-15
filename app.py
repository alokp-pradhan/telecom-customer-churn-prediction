# ============================================================
# app.py  —  Telecom Customer Churn Prediction
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ─────────────────────────────────────────────
# Load Pickle Files
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    lr_model        = pickle.load(open("lr_model.pkl",          "rb"))
    rf_model        = pickle.load(open("rf_model.pkl",          "rb"))
    xgb_model       = pickle.load(open("xgb_model.pkl",         "rb"))
    lgb_model       = pickle.load(open("lgb_model.pkl",         "rb"))
    cat_model       = pickle.load(open("cat_model.pkl",         "rb"))
    scaler          = pickle.load(open("scaler.pkl",             "rb"))
    feature_columns = pickle.load(open("feature_columns.pkl",   "rb"))
    return lr_model, rf_model, xgb_model, lgb_model, cat_model, scaler, feature_columns

lr_model, rf_model, xgb_model, lgb_model, cat_model, scaler, feature_columns = load_artifacts()

models = {
    'logistic_regression': lr_model,
    'random_forest':       rf_model,
    'xgboost':             xgb_model,
    'lightgbm':            lgb_model,
    'catboost':            cat_model,
}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📡 Telecom Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they are likely to **churn**.")
st.markdown("---")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

model_display_names = {
    'catboost':            '🥇 CatBoost',
    'xgboost':             '🥈 XGBoost',
    'lightgbm':            '🥉 LightGBM',
    'random_forest':       '🌲 Random Forest',
    'logistic_regression': '📈 Logistic Regression',
}

selected_model_key = st.sidebar.selectbox(
    "Select Model",
    options=list(model_display_names.keys()),
    format_func=lambda x: model_display_names[x]
)

show_probability        = st.sidebar.checkbox("Show Churn Probability",  value=True)
show_feature_importance = st.sidebar.checkbox("Show Feature Importance", value=False)

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender         = st.selectbox("Gender",         ['Male', 'Female'])
    senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    partner        = st.selectbox("Partner",        ['Yes', 'No'])
    dependents     = st.selectbox("Dependents",     ['No', 'Yes'])

with col2:
    st.markdown("**Account Info**")
    tenure            = st.slider("Tenure (months)", 0, 72, 12)
    contract          = st.selectbox("Contract Type",     ['Month-to-month', 'One year', 'Two year'])
    payment_method    = st.selectbox("Payment Method",    ['Electronic check', 'Mailed check',
                                                           'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])

with col3:
    st.markdown("**Charges**")
    monthly_charges = st.number_input("Monthly Charges ($)",  min_value=0.0,   max_value=200.0,   value=65.0,  step=0.5)
    total_charges   = st.number_input("Total Charges ($)",    min_value=0.0,   max_value=10000.0, value=float(monthly_charges * tenure), step=1.0)

st.markdown("---")
st.subheader("📦 Services Subscribed")

col4, col5, col6, col7 = st.columns(4)

with col4:
    phone_service  = st.selectbox("Phone Service",  ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])

with col5:
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security  = st.selectbox("Online Security",  ['No', 'Yes', 'No internet service'])

with col6:
    online_backup     = st.selectbox("Online Backup",     ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])

with col7:
    tech_support     = st.selectbox("Tech Support",      ['No', 'Yes', 'No internet service'])
    streaming_tv     = st.selectbox("Streaming TV",      ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies",  ['No', 'Yes', 'No internet service'])

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
def build_input_df(raw):
    df = pd.DataFrame([raw])

    df['SeniorCitizen']  = 1 if raw['SeniorCitizen'] == 'Yes' else 0
    df['tenure']         = int(raw['tenure'])
    df['MonthlyCharges'] = float(raw['MonthlyCharges'])
    df['TotalCharges']   = float(raw['TotalCharges'])

    df['TotalSpend']      = df['MonthlyCharges'] * df['tenure']
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ContractRisk']    = (df['Contract'] == 'Month-to-month').astype(int)

    service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['ServiceCount']   = sum((df[col] == 'Yes').astype(int) for col in service_cols)
    df['CostPerService'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)

    df['TenureGroup'] = pd.cut(
        df['tenure'], bins=[0, 12, 24, 48, 72],
        labels=['0-1 Year', '1-2 Years', '2-4 Years', '4-6 Years']
    )
    df['MonthlyChargesGroup'] = pd.cut(
        df['MonthlyCharges'], bins=[0, 35, 65, 90, 200],
        labels=['Low', 'Medium', 'High', 'Very High']
    )

    df['SeniorHighCharge']      = ((df['SeniorCitizen'] == 1) & (df['MonthlyCharges'] > 64.76)).astype(int)
    df['Tenure_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['LoyaltyScore']          = df['tenure'] * df['ServiceCount']

    cat_cols   = [c for c in df.columns if df[c].dtype in ['object', 'category']]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Add any missing columns as 0
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    return df_encoded[feature_columns]

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Predict Churn", use_container_width=True, type="primary")

if predict_btn:
    raw_input = {
        'gender':           gender,
        'SeniorCitizen':    senior_citizen,
        'Partner':          partner,
        'Dependents':       dependents,
        'tenure':           tenure,
        'PhoneService':     phone_service,
        'MultipleLines':    multiple_lines,
        'InternetService':  internet_service,
        'OnlineSecurity':   online_security,
        'OnlineBackup':     online_backup,
        'DeviceProtection': device_protection,
        'TechSupport':      tech_support,
        'StreamingTV':      streaming_tv,
        'StreamingMovies':  streaming_movies,
        'Contract':         contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod':    payment_method,
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
    }

    input_df    = build_input_df(raw_input)
    model       = models[selected_model_key]
    input_array = scaler.transform(input_df) if selected_model_key == 'logistic_regression' else input_df.values

    prediction  = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]

    # ── Result ──
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    r1, r2, r3 = st.columns(3)
    with r1:
        if prediction == 1:
            st.error("⚠️ **CHURN PREDICTED**\nThis customer is likely to leave.")
        else:
            st.success("✅ **NO CHURN**\nThis customer is likely to stay.")
    with r2:
        if show_probability:
            st.metric("Churn Probability", f"{probability * 100:.1f}%",
                      delta="High Risk" if probability > 0.5 else "Low Risk")
    with r3:
        st.metric("Model Used", model_display_names[selected_model_key])

    # ── Gauge ──
    if show_probability:
        st.markdown("#### Churn Probability Gauge")
        gauge_color = "🔴" if probability > 0.7 else ("🟡" if probability > 0.4 else "🟢")
        filled      = int(probability * 20)
        bar_str     = "█" * filled + "░" * (20 - filled)
        st.markdown(f"`{gauge_color} [{bar_str}]  {probability * 100:.1f}%`")

    # ── Feature Importance ──
    if show_feature_importance and hasattr(model, 'feature_importances_'):
        st.markdown("---")
        st.subheader("📌 Top 10 Feature Importances")
        import matplotlib.pyplot as plt
        fi_df = pd.DataFrame({
            'Feature':    feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue')
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance — {model_display_names[selected_model_key]}')
        plt.tight_layout()
        st.pyplot(fig)

    # ── Input Summary ──
    with st.expander("📋 View Input Summary"):
        st.dataframe(pd.DataFrame([raw_input]).T.rename(columns={0: 'Value'}))

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("📡 Telecom Churn Predictor | Built with Streamlit | Models: LR · RF · XGBoost · LightGBM · CatBoost")
