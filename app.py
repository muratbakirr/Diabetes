import streamlit as st
import pandas as pd
import numpy as np
import joblib

def load_model(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def main():
    model, scaler = load_model('./models/diabetes_rf_best_model.pkl', './models/scaler.pkl')

    st.title('Diabetes Prediction')

    st.sidebar.header('User Input Features')

    Pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    Glucose = st.sidebar.slider('Glucose', min_value=0, max_value=200, value=120)
    BloodPressure = st.sidebar.slider('BloodPressure', min_value=0, max_value=150, value=70)
    SkinThickness = st.sidebar.slider('SkinThickness', min_value=0, max_value=100, value=20)
    Insulin = st.sidebar.slider('Insulin', min_value=0, max_value=1000, value=100)
    BMI = st.sidebar.slider('BMI', min_value=0.0, max_value=60.0, value=25.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, value=0.5)
    Age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)

    X = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    X = scaler.transform(X)
    y_pred = model.predict(X)

    if st.button('Predict'):
        if y_pred[0] == 0:
            st.write('The person is not diabetic')
        else:
            st.write('The person is diabetic')


if __name__ == '__main__':
  main()
