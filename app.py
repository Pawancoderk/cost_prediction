import numpy as np
import streamlit as st
import pickle

with open("D:\python\Regresion-model Project\App\models\scaler.pkl","rb") as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open("D:\python\Regresion-model Project\App\models\model.pkl","rb") as model_file:
    loaded_model = pickle.load(model_file)

st.title("E-Commerce Predictor ")

avg_session_length = st.number_input("Average session length")
time_on_app = st.number_input("Time on app")
lenght_of_membership = st.number_input("Length of membership")

if st.button("Predict"):
    data = np.array([avg_session_length,time_on_app,lenght_of_membership]).reshape(1,-1)
    data_new = loaded_scaler.transform(data)
    prediction = loaded_model.predict(data_new)

    st.success(f"Yearly Amount Spent is : $ {prediction[0]}")