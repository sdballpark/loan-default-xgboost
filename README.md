# 🏦 Loan Default Prediction with XGBoost and Explainable AI

## 📚 Problem Statement
Predict which customers are likely to default on a loan using historical loan application data.

## 📂 Dataset
- LendingClub Loan Data (Kaggle)

## 🛠️ Methodology
- Data cleaning and feature engineering
- Train models using **XGBoost** and **LightGBM**
- Evaluate using AUC-ROC, F1 Score
- Model interpretability with **SHAP values**

## 🚀 Deliverables
- Prediction model for loan defaults
- SHAP dashboard to explain predictions
- Streamlit app for end users

## 📈 Key Metrics
- Target: ROC-AUC > 0.85
- Interpretability requirement (business audit)

## 💡 Future Enhancements
- Add ensemble methods
- Monitor model drift with production data

## 🧠 How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/1_EDA.ipynb
python src/model_training.py
streamlit run app/streamlit_app.py
```
