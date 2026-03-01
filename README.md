End-to-End Customer Churn Prediction System
Project Overview
This project presents a complete end-to-end machine learning solution for predicting customer churn in a telecom business setting. The objective is to identify customers who are likely to leave the service so that proactive retention strategies can be applied.
The system covers the entire ML lifecycle, including data preprocessing, model development, class imbalance handling, performance evaluation, explainability using SHAP, and deployment through a Streamlit web application.
A live version of the application is deployed on Streamlit Cloud.
Problem Statement
Customer churn is a critical business problem in subscription-based industries such as telecommunications. Acquiring new customers is significantly more expensive than retaining existing ones. Therefore, predicting which customers are at high risk of churn enables companies to:
•	Reduce revenue loss
•	Improve customer retention
•	Optimize marketing and retention budgets
•	Personalize engagement strategies
This project builds a predictive system that classifies customers as likely to churn or not churn based on historical behavioral and service usage data.
Dataset Description
The dataset contains customer-level information including:
•	Demographics (e.g., dependents, senior citizen status)
•	Account information (e.g., tenure, contract type, payment method)
•	Service usage (e.g., internet service, tech support, online security)
•	Financial metrics (monthly charges, total charges)
Target variable:
•	Churn Value (0 = No churn, 1 = Churn)
Methodology
1. Data Preprocessing
•	Removed data leakage columns such as churn score and churn reason.
•	Converted incorrect data types (e.g., Total Charges).
•	Handled missing values.
•	Separated numerical and categorical features.
•	Applied one-hot encoding using a ColumnTransformer within a pipeline.
2. Handling Class Imbalance
The dataset was moderately imbalanced (~26% churn rate). Instead of resampling initially, a class-weighted Logistic Regression model was used to improve recall for the minority class.
3. Model Development
Models evaluated:
•	Logistic Regression (baseline)
•	Random Forest
•	Logistic Regression with class_weight="balanced"
Final selected model:
•	Logistic Regression with class weighting
4. Performance Metrics
Because churn detection prioritizes capturing high-risk customers, evaluation focused on:
•	Recall (Churn class)
•	Precision
•	F1-score
•	ROC-AUC
Final model performance:
•	ROC-AUC: 0.84
•	Churn Recall: ~0.79 (at default threshold)
•	Threshold tuning applied for business optimization
Threshold Optimization
Instead of relying on the default 0.5 probability threshold, multiple thresholds (0.4, 0.5, 0.6) were evaluated to balance precision and recall.
This allows business stakeholders to choose strategies based on cost sensitivity:
•	Lower threshold → higher recall (aggressive retention strategy)
•	Higher threshold → higher precision (controlled marketing budget)
Model Explainability (SHAP)
SHAP was used to:
•	Identify global feature importance
•	Interpret how features push predictions toward churn or retention
•	Improve transparency and trust in the model
Key churn drivers identified:
•	Short tenure increases churn risk
•	Month-to-month contracts increase churn probability
•	Electronic check payment method increases churn
•	High monthly charges increase churn
•	Long-term contracts and dependents reduce churn
Deployment
The model was deployed using:
•	Streamlit for interactive web interface
•	Joblib for model serialization
•	Streamlit Community Cloud for hosting
The application allows users to:
•	Input customer attributes
•	Receive churn probability
•	View risk categorization (Low, Medium, High)
•	See business-oriented recommendations
Environment compatibility issues were resolved by:
•	Pinning exact library versions
•	Matching Python versions between training and deployment
•	Using runtime.txt for controlled environment setup
Tech Stack
•	Python 3.10
•	scikit-learn 1.6.1
•	NumPy 1.26.4
•	pandas 2.2.2
•	joblib 1.5.3
•	SHAP
•	Streamlit
Key Learnings
This project provided hands-on experience in:
•	End-to-end ML pipeline design
•	Handling imbalanced datasets
•	Model evaluation beyond accuracy
•	Threshold tuning for business alignment
•	Explainable AI using SHAP
•	Cloud deployment and environment debugging
•	Version control and reproducible ML workflows
Future Improvements
•	Experiment with XGBoost and gradient boosting models
•	Add probability calibration
•	Integrate SHAP explanations directly into the web interface
•	Add customer segmentation analysis
•	Implement CI/CD pipeline for automated deployment

