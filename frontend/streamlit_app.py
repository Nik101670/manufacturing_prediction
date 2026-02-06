import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Manufacturing App", layout="centered")

st.title("🏭 Manufacturing Prediction App")

# Get features from API
try:
    features = requests.get(f"{API_URL}/features").json()["feature_columns"]
except:
    st.error("Backend API not running. Start FastAPI first.")
    st.stop()

st.write("Fill feature values below:")

user_data = {}

for feat in features:
    user_data[feat] = st.number_input(feat, value=0.0)

if st.button("Predict"):
    res = requests.post(f"{API_URL}/predict", json={"data": user_data})

    if res.status_code == 200:
        pred = res.json()["prediction"]
        st.success(f"Prediction: {pred}")
    else:
        st.error("Prediction failed")
        st.write(res.text)



