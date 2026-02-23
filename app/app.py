import streamlit as st
import joblib
import pandas as pd
import os

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Churn Prediction System",
    page_icon="📊",
    layout="centered"
)

# ---------------- LOAD MODEL ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "churn_model.pkl")

model = joblib.load(model_path)
FINAL_THRESHOLD = 0.45

# ---------------- HEADER ---------------- #
st.title("📊 AI-Powered Customer Churn Prediction")
st.markdown("Predict customer churn risk and get actionable business insights.")

st.divider()

# ---------------- INPUT SECTION ---------------- #
st.subheader("📝 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check",
         "Mailed check",
         "Bank transfer (automatic)",
         "Credit card (automatic)"]
    )
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])

st.divider()

# ---------------- PREDICTION BUTTON ---------------- #
if st.button("🔍 Predict Churn Risk", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Tenure Months": [tenure],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges],
        "Contract": [contract],
        "Payment Method": [payment_method],
        "Dependents": [dependents],
        "Tech Support": [tech_support],
        "Online Security": [online_security]
    })

    # Add missing columns used during training
    for col in model.named_steps["preprocessing"].feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = "No"

    # Reorder columns correctly
    input_data = input_data[model.named_steps["preprocessing"].feature_names_in_]

    # Get probability
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("📊 Prediction Results")

    # Show probability
    st.markdown(f"### Churn Probability: **{probability:.2%}**")

    # Progress bar
    st.progress(float(probability))

    # Risk categorization
    if probability < 0.30:
        st.success("🟢 Low Risk of Churn")
        st.info("Recommended Action: No immediate intervention required.")

    elif probability < 0.60:
        st.warning("🟡 Medium Risk of Churn")
        st.info("Recommended Action: Engage customer with personalized retention offer.")

    else:
        st.error("🔴 High Risk of Churn")
        st.info("Recommended Action: Offer retention discount and encourage long-term contract upgrade.")

    st.divider()

    # Additional business interpretation
st.subheader("📈 Business Insight")

insights = []

if tenure < 6:
    insights.append("• Low tenure indicates early churn vulnerability.")

if contract == "Month-to-month":
    insights.append("• Month-to-month contracts show higher churn probability.")

if payment_method == "Electronic check":
    insights.append("• Electronic check users historically show higher churn.")

if monthly_charges > 90:
    insights.append("• High monthly charges increase churn risk.")

if dependents == "Yes":
    insights.append("• Customers with dependents are generally more stable.")

# If no specific insights triggered
if not insights:
    insights.append("• Customer profile does not show strong historical churn risk indicators.")

# Display insights
for insight in insights:
    st.write(insight)

# ---------------- FOOTER ---------------- #
st.divider()
st.caption("Built with Machine Learning | Logistic Regression + SHAP | ROC-AUC: 0.84")