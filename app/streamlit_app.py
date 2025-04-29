import streamlit as st
import pandas as pd
from src.model_inference import load_model, predict

st.title('🏦 Loan Default Prediction App')

uploaded_file = st.file_uploader("Upload applicant loan data (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    model = load_model('models/loan_default_model.pkl')
    
    predictions = predict(model, data)
    
    data['Default Prediction'] = predictions
    st.write(data)
    st.success('Predictions completed!')
