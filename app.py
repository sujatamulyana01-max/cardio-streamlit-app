import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cardio Health Prediction",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_assets():
    classifier = joblib.load("cardio_classifier.pkl")
    regressor = joblib.load("systolic_regressor.pkl")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return classifier, regressor, metadata

classifier, regressor, metadata = load_assets()

st.title("❤️ Cardio Health Prediction App")
st.write(
    "Aplikasi ini dibuat dari project notebook Cardiovascular Data Modelling. "
    "Masukkan data pasien untuk memprediksi risiko penyakit kardiovaskular dan tekanan darah sistolik."
)

with st.expander("Info performa model"):
    st.write(f"**Classification Accuracy:** {metadata['classification_accuracy']:.4f}")
    st.write(f"**Classification F1 Score:** {metadata['classification_f1']:.4f}")
    st.write(f"**Regression MAE:** {metadata['regression_mae']:.2f}")
    st.write(f"**Regression RMSE:** {metadata['regression_rmse']:.2f}")
    st.write(f"**Regression R²:** {metadata['regression_r2']:.4f}")

tab1, tab2 = st.tabs(["Prediksi Cardio", "Prediksi Tekanan Darah Sistolik"])

with tab1:
    st.subheader("Prediksi Risiko Penyakit Kardiovaskular")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia (tahun)", min_value=30, max_value=100, value=50)
        height = st.number_input("Tinggi badan (cm)", min_value=130, max_value=210, value=165)
        weight = st.number_input("Berat badan (kg)", min_value=40.0, max_value=200.0, value=70.0)
        ap_hi = st.number_input("Tekanan sistolik / ap_hi", min_value=80, max_value=220, value=120)
        ap_lo = st.number_input("Tekanan diastolik / ap_lo", min_value=50, max_value=140, value=80)

    with col2:
        cholesterol = st.selectbox("Kolesterol", [0, 1, 2], format_func=lambda x: ["Normal", "Above normal", "Well above normal"][x])
        gluc = st.selectbox("Glukosa", [0, 1, 2], format_func=lambda x: ["Normal", "Above normal", "Well above normal"][x])
        smoke = st.selectbox("Merokok", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        alco = st.selectbox("Konsumsi alkohol", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        active = st.selectbox("Aktif secara fisik", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

    bmi = round(weight / ((height / 100) ** 2), 2)
    pulse_pressure = ap_hi - ap_lo

    input_cls = pd.DataFrame([{
        "age": age,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "bmi": bmi,
        "pulse_pressure": pulse_pressure,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }])

    st.write(f"**BMI:** {bmi}")
    st.write(f"**Pulse Pressure:** {pulse_pressure}")

    if st.button("Prediksi Cardio"):
        prediction = classifier.predict(input_cls)[0]
        probability = classifier.predict_proba(input_cls)[0][1]

        if prediction == 1:
            st.error(f"Prediksi: Berisiko penyakit kardiovaskular. Probabilitas: {probability:.2%}")
        else:
            st.success(f"Prediksi: Tidak terindikasi berisiko tinggi. Probabilitas risiko: {probability:.2%}")

with tab2:
    st.subheader("Prediksi Tekanan Darah Sistolik / ap_hi")

    col1, col2 = st.columns(2)
    with col1:
        age_r = st.number_input("Usia untuk regresi", min_value=30, max_value=100, value=50)
        gender_r = st.selectbox("Gender", [0, 1], format_func=lambda x: "Kategori 1" if x == 1 else "Kategori 0")
        height_r = st.number_input("Tinggi badan untuk regresi (cm)", min_value=130, max_value=210, value=165)
        weight_r = st.number_input("Berat badan untuk regresi (kg)", min_value=40.0, max_value=200.0, value=70.0)
        ap_lo_r = st.number_input("Tekanan diastolik / ap_lo untuk regresi", min_value=50, max_value=140, value=80)

    with col2:
        cholesterol_r = st.selectbox("Kolesterol untuk regresi", [0, 1, 2], key="chol_r")
        gluc_r = st.selectbox("Glukosa untuk regresi", [0, 1, 2], key="gluc_r")
        smoke_r = st.selectbox("Merokok untuk regresi", [0, 1], key="smoke_r")
        alco_r = st.selectbox("Alkohol untuk regresi", [0, 1], key="alco_r")
        active_r = st.selectbox("Aktif fisik untuk regresi", [0, 1], key="active_r")
        cardio_r = st.selectbox("Status cardio", [0, 1], format_func=lambda x: "Ada cardio" if x == 1 else "Tidak ada cardio")

    bmi_r = round(weight_r / ((height_r / 100) ** 2), 2)

    input_reg = pd.DataFrame([{
        "age": age_r,
        "gender": gender_r,
        "height": height_r,
        "weight": weight_r,
        "ap_lo": ap_lo_r,
        "cholesterol": cholesterol_r,
        "gluc": gluc_r,
        "smoke": smoke_r,
        "alco": alco_r,
        "active": active_r,
        "cardio": cardio_r,
        "bmi": bmi_r
    }])

    st.write(f"**BMI:** {bmi_r}")

    if st.button("Prediksi Tekanan Sistolik"):
        predicted_ap_hi = regressor.predict(input_reg)[0]
        st.info(f"Prediksi tekanan darah sistolik: **{predicted_ap_hi:.2f} mmHg**")

st.caption("Catatan: Aplikasi ini untuk tugas/edukasi, bukan pengganti diagnosis dokter.")