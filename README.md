# 🔁 End-to-End Customer Churn Prediction System

---

## 📌 Project Overview

This project presents a complete **end-to-end machine learning solution** for predicting customer churn in a telecom business setting. The objective is to identify customers who are likely to leave the service so that proactive retention strategies can be applied.

The system covers the entire ML lifecycle, including:
- Data preprocessing
- Model development
- Class imbalance handling
- Performance evaluation
- Explainability using **SHAP**
- Deployment through a **Streamlit** web application

> 🚀 A live version of the application is deployed on **Streamlit Cloud**.

---

## ❓ Problem Statement

Customer churn is a critical business problem in subscription-based industries such as telecommunications. Acquiring new customers is significantly more expensive than retaining existing ones.

Predicting which customers are at high risk of churn enables companies to:

- 💰 Reduce revenue loss
- 🤝 Improve customer retention
- 📊 Optimize marketing and retention budgets
- 🎯 Personalize engagement strategies

This project builds a predictive system that classifies customers as **likely to churn** or **not churn** based on historical behavioral and service usage data.

---

## 📂 Dataset Description

The dataset contains customer-level information including:

| Category | Features |
|---|---|
| **Demographics** | Dependents, senior citizen status |
| **Account Info** | Tenure, contract type, payment method |
| **Service Usage** | Internet service, tech support, online security |
| **Financial Metrics** | Monthly charges, total charges |

**Target Variable:**
- `Churn Value` → `0` = No Churn, `1` = Churn

---

## 🧪 Methodology

### 1. Data Preprocessing
- Removed data leakage columns (e.g., churn score, churn reason)
- Converted incorrect data types (e.g., `Total Charges`)
- Handled missing values
- Separated numerical and categorical features
- Applied **one-hot encoding** using a `ColumnTransformer` within a pipeline

### 2. Handling Class Imbalance
The dataset was moderately imbalanced (~26% churn rate). Instead of resampling initially, a **class-weighted Logistic Regression** model was used to improve recall for the minority class.

### 3. Model Development

**Models Evaluated:**
- Logistic Regression *(baseline)*
- Random Forest
- Logistic Regression with `class_weight="balanced"`

**✅ Final Selected Model:**
- Logistic Regression with class weighting

### 4. Performance Metrics

Because churn detection prioritizes capturing high-risk customers, evaluation focused on:

| Metric | Value |
|---|---|
| **ROC-AUC** | 0.84 |
| **Churn Recall** | ~0.79 *(at default threshold)* |
| Precision | — |
| F1-Score | — |

> Threshold tuning was applied for further business optimization.

---

## ⚖️ Threshold Optimization

Instead of relying on the default `0.5` probability threshold, multiple thresholds were evaluated:

| Threshold | Effect |
|---|---|
| `0.4` | Higher Recall → Aggressive retention strategy |
| `0.5` | Balanced default |
| `0.6` | Higher Precision → Controlled marketing budget |

This allows business stakeholders to choose strategies based on **cost sensitivity**.

---

## 🔍 Model Explainability (SHAP)

SHAP was used to:
- Identify **global feature importance**
- Interpret how features push predictions toward churn or retention
- Improve **transparency and trust** in the model

**Key Churn Drivers Identified:**

| Factor | Effect on Churn |
|---|---|
| Short tenure | ⬆️ Increases churn risk |
| Month-to-month contracts | ⬆️ Increases churn probability |
| Electronic check payment | ⬆️ Increases churn |
| High monthly charges | ⬆️ Increases churn |
| Long-term contracts | ⬇️ Reduces churn |
| Having dependents | ⬇️ Reduces churn |

---

## 🚀 Deployment

The model was deployed using:

- **Streamlit** — Interactive web interface
- **Joblib** — Model serialization
- **Streamlit Community Cloud** — Hosting

**The application allows users to:**
- Input customer attributes
- Receive churn probability
- View risk categorization: `Low` / `Medium` / `High`
- See business-oriented recommendations

**Environment compatibility issues were resolved by:**
- Pinning exact library versions
- Matching Python versions between training and deployment
- Using `runtime.txt` for controlled environment setup

---

## 🛠️ Tech Stack

| Tool | Version |
|---|---|
| Python | 3.10 |
| scikit-learn | 1.6.1 |
| NumPy | 1.26.4 |
| pandas | 2.2.2 |
| joblib | 1.5.3 |
| SHAP | latest |
| Streamlit | latest |

---

## 📚 Key Learnings

This project provided hands-on experience in:

- ✅ End-to-end ML pipeline design
- ✅ Handling imbalanced datasets
- ✅ Model evaluation beyond accuracy
- ✅ Threshold tuning for business alignment
- ✅ Explainable AI using SHAP
- ✅ Cloud deployment and environment debugging
- ✅ Version control and reproducible ML workflows

---

## 🔮 Future Improvements

- [ ] Experiment with XGBoost and gradient boosting models
- [ ] Add probability calibration
- [ ] Integrate SHAP explanations directly into the web interface
- [ ] Add customer segmentation analysis
- [ ] Implement CI/CD pipeline for automated deployment

---

> Made with ❤️ using Python & Streamlit
