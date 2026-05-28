import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cardio Health Prediction",
    page_icon="❤️",
    layout="wide"
)

NORMAL_SYSTOLIC_MAX = 120
NORMAL_DIASTOLIC_MAX = 80

CHOLESTEROL_LABELS = {
    0: "Normal",
    1: "Above normal",
    2: "Well above normal"
}

GLUCOSE_LABELS = {
    0: "Normal",
    1: "Above normal",
    2: "Well above normal"
}

YES_NO_LABELS = {
    0: "Tidak",
    1: "Ya"
}

GENDER_LABELS = {
    0: "Kategori 0",
    1: "Kategori 1"
}


@st.cache_resource
def load_assets():
    classifier = joblib.load("cardio_classifier.pkl")
    regressor = joblib.load("systolic_regressor.pkl")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return classifier, regressor, metadata


classifier, regressor, metadata = load_assets()


def calculate_bmi(weight, height):
    return round(weight / ((height / 100) ** 2), 2)


def calculate_pulse_pressure(ap_hi, ap_lo):
    return ap_hi - ap_lo


def confidence_label(confidence_value):
    if confidence_value >= 0.75:
        return "Tinggi"
    if confidence_value >= 0.60:
        return "Sedang"
    return "Rendah"


def bp_normal_label(predicted_ap_hi, ap_lo):
    """Binary label for the regression output: Normal or Tidak Normal."""
    if predicted_ap_hi < NORMAL_SYSTOLIC_MAX and ap_lo < NORMAL_DIASTOLIC_MAX:
        return "Normal"
    return "Tidak Normal"


def regression_confidence(predicted_value):
    """
    Regression does not produce probability, so this is an estimation score
    based on average model error (MAE) relative to the predicted systolic value.
    """
    mae = float(metadata.get("regression_mae", 0))
    if predicted_value <= 0:
        return 0.0
    score = 1 - (mae / predicted_value)
    return max(0.0, min(1.0, score))


def make_classification_input(age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    bmi = calculate_bmi(weight, height)
    pulse_pressure = calculate_pulse_pressure(ap_hi, ap_lo)
    row = {
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
    }
    return pd.DataFrame([row])[metadata["classification_features"]], bmi, pulse_pressure


def make_regression_input(age, gender, height, weight, ap_lo, cholesterol, gluc, smoke, alco, active, cardio):
    bmi = calculate_bmi(weight, height)
    row = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "cardio": cardio,
        "bmi": bmi
    }
    return pd.DataFrame([row])[metadata["regression_features"]], bmi


def predict_cardio(input_cls):
    prediction = int(classifier.predict(input_cls)[0])
    probabilities = classifier.predict_proba(input_cls)[0]
    risk_probability = float(probabilities[1])
    confidence = float(max(probabilities))
    return prediction, risk_probability, confidence


def predict_systolic(input_reg):
    return float(regressor.predict(input_reg)[0])


def show_metric_cards(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(item["label"], item["value"], item.get("delta"))


def render_common_inputs(prefix, default_ap_hi=120, default_ap_lo=80):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Usia (tahun)", min_value=30, max_value=100, value=50, key=f"{prefix}_age")
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: GENDER_LABELS[x], key=f"{prefix}_gender")
        height = st.number_input("Tinggi badan (cm)", min_value=130, max_value=210, value=165, key=f"{prefix}_height")
        weight = st.number_input("Berat badan (kg)", min_value=40.0, max_value=200.0, value=70.0, key=f"{prefix}_weight")
    with col2:
        ap_hi = st.number_input("Tekanan sistolik / ap_hi", min_value=80, max_value=220, value=default_ap_hi, key=f"{prefix}_ap_hi")
        ap_lo = st.number_input("Tekanan diastolik / ap_lo", min_value=50, max_value=140, value=default_ap_lo, key=f"{prefix}_ap_lo")
        cholesterol = st.selectbox("Kolesterol", [0, 1, 2], format_func=lambda x: CHOLESTEROL_LABELS[x], key=f"{prefix}_cholesterol")
        gluc = st.selectbox("Glukosa", [0, 1, 2], format_func=lambda x: GLUCOSE_LABELS[x], key=f"{prefix}_gluc")
    with col3:
        smoke = st.selectbox("Merokok", [0, 1], format_func=lambda x: YES_NO_LABELS[x], key=f"{prefix}_smoke")
        alco = st.selectbox("Konsumsi alkohol", [0, 1], format_func=lambda x: YES_NO_LABELS[x], key=f"{prefix}_alco")
        active = st.selectbox("Aktif secara fisik", [0, 1], format_func=lambda x: YES_NO_LABELS[x], key=f"{prefix}_active")

    return {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }


def predict_full_profile(profile):
    input_cls, bmi, pulse_pressure = make_classification_input(
        profile["age"], profile["height"], profile["weight"],
        profile["ap_hi"], profile["ap_lo"], profile["cholesterol"],
        profile["gluc"], profile["smoke"], profile["alco"], profile["active"]
    )
    cls_pred, risk_prob, cls_confidence = predict_cardio(input_cls)

    input_reg, bmi_reg = make_regression_input(
        profile["age"], profile["gender"], profile["height"], profile["weight"],
        profile["ap_lo"], profile["cholesterol"], profile["gluc"],
        profile["smoke"], profile["alco"], profile["active"], cls_pred
    )
    predicted_ap_hi = predict_systolic(input_reg)
    reg_confidence = regression_confidence(predicted_ap_hi)
    reg_label = bp_normal_label(predicted_ap_hi, profile["ap_lo"])

    return {
        "BMI": bmi,
        "Pulse Pressure": pulse_pressure,
        "Prediksi Cardio": "Berisiko" if cls_pred == 1 else "Tidak Berisiko Tinggi",
        "Risk Probability": risk_prob,
        "Classification Confidence": cls_confidence,
        "Confidence Level": confidence_label(cls_confidence),
        "Prediksi Sistolik": predicted_ap_hi,
        "Regression Label": reg_label,
        "Regression Confidence": reg_confidence,
        "Regression Confidence Level": confidence_label(reg_confidence)
    }


st.title("❤️ Cardio Health Prediction App")
st.write(
    "Aplikasi ini dibuat dari project notebook Cardiovascular Data Modelling. "
    "Masukkan data pasien untuk memprediksi risiko penyakit kardiovaskular, "
    "tekanan darah sistolik, confidence level, serta membandingkan dua skenario data."
)

with st.expander("Info performa model dan aturan deployment"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Classification Accuracy", f"{metadata['classification_accuracy']:.2%}")
        st.metric("Classification F1 Score", f"{metadata['classification_f1']:.2%}")
    with col_b:
        st.metric("Regression MAE", f"{metadata['regression_mae']:.2f} mmHg")
        st.metric("Regression RMSE", f"{metadata['regression_rmse']:.2f} mmHg")
    with col_c:
        st.metric("Regression R²", f"{metadata['regression_r2']:.2%}")
        st.metric("Normal BP Rule", "<120 / <80")

    st.info(
        "Confidence level klasifikasi dihitung dari probabilitas tertinggi model. "
        "Untuk regresi, confidence adalah estimasi berbasis MAE, bukan probabilitas klinis. "
        "Label regresi memakai aturan sederhana: Normal jika prediksi sistolik <120 mmHg dan diastolik <80 mmHg; selain itu Tidak Normal."
    )


tab1, tab2, tab3 = st.tabs([
    "Prediksi Cardio",
    "Prediksi Tekanan Darah Sistolik",
    "Comparison Feature"
])

with tab1:
    st.subheader("Prediksi Risiko Penyakit Kardiovaskular")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia (tahun)", min_value=30, max_value=100, value=50, key="cls_age")
        height = st.number_input("Tinggi badan (cm)", min_value=130, max_value=210, value=165, key="cls_height")
        weight = st.number_input("Berat badan (kg)", min_value=40.0, max_value=200.0, value=70.0, key="cls_weight")
        ap_hi = st.number_input("Tekanan sistolik / ap_hi", min_value=80, max_value=220, value=120, key="cls_ap_hi")
        ap_lo = st.number_input("Tekanan diastolik / ap_lo", min_value=50, max_value=140, value=80, key="cls_ap_lo")

    with col2:
        cholesterol = st.selectbox("Kolesterol", [0, 1, 2], format_func=lambda x: CHOLESTEROL_LABELS[x], key="cls_cholesterol")
        gluc = st.selectbox("Glukosa", [0, 1, 2], format_func=lambda x: GLUCOSE_LABELS[x], key="cls_gluc")
        smoke = st.selectbox("Merokok", [0, 1], format_func=lambda x: YES_NO_LABELS[x], key="cls_smoke")
        alco = st.selectbox("Konsumsi alkohol", [0, 1], format_func=lambda x: YES_NO_LABELS[x], key="cls_alco")
        active = st.selectbox("Aktif secara fisik", [0, 1], format_func=lambda x: YES_NO_LABELS[x], key="cls_active")

    input_cls, bmi, pulse_pressure = make_classification_input(
        age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    )

    show_metric_cards([
        {"label": "BMI", "value": f"{bmi:.2f}"},
        {"label": "Pulse Pressure", "value": f"{pulse_pressure} mmHg"},
        {"label": "Akurasi Model", "value": f"{metadata['classification_accuracy']:.2%}"}
    ])

    if st.button("Prediksi Cardio", type="primary"):
        prediction, probability, confidence = predict_cardio(input_cls)
        st.divider()
        show_metric_cards([
            {"label": "Risk Probability", "value": f"{probability:.2%}"},
            {"label": "Confidence", "value": f"{confidence:.2%}", "delta": confidence_label(confidence)},
            {"label": "F1 Score Model", "value": f"{metadata['classification_f1']:.2%}"}
        ])

        if prediction == 1:
            st.error("Prediksi: Berisiko penyakit kardiovaskular.")
        else:
            st.success("Prediksi: Tidak terindikasi berisiko tinggi.")

        st.caption(
            "Confidence menunjukkan seberapa yakin model terhadap kelas hasil prediksi. "
            "Nilai ini berbeda dari akurasi model keseluruhan."
        )

with tab2:
    st.subheader("Prediksi Tekanan Darah Sistolik / ap_hi")

    col1, col2 = st.columns(2)
    with col1:
        age_r = st.number_input("Usia untuk regresi", min_value=30, max_value=100, value=50, key="reg_age")
        gender_r = st.selectbox("Gender", [0, 1], format_func=lambda x: GENDER_LABELS[x], key="reg_gender")
        height_r = st.number_input("Tinggi badan untuk regresi (cm)", min_value=130, max_value=210, value=165, key="reg_height")
        weight_r = st.number_input("Berat badan untuk regresi (kg)", min_value=40.0, max_value=200.0, value=70.0, key="reg_weight")
        ap_lo_r = st.number_input("Tekanan diastolik / ap_lo untuk regresi", min_value=50, max_value=140, value=80, key="reg_ap_lo")

    with col2:
        cholesterol_r = st.selectbox("Kolesterol untuk regresi", [0, 1, 2], format_func=lambda x: CHOLESTEROL_LABELS[x], key="reg_cholesterol")
        gluc_r = st.selectbox("Glukosa untuk regresi", [0, 1, 2], format_func=lambda x: GLUCOSE_LABELS[x], key="reg_gluc")
        smoke_r = st.selectbox("Merokok untuk regresi", [0, 1], format_func=lambda x: YES_NO_LABELS[x], key="reg_smoke")
        alco_r = st.selectbox("Alkohol untuk regresi", [0, 1], format_func=lambda x: YES_NO_LABELS[x], key="reg_alco")
        active_r = st.selectbox("Aktif fisik untuk regresi", [0, 1], format_func=lambda x: YES_NO_LABELS[x], key="reg_active")
        cardio_r = st.selectbox("Status cardio", [0, 1], format_func=lambda x: "Ada cardio" if x == 1 else "Tidak ada cardio", key="reg_cardio")

    input_reg, bmi_r = make_regression_input(
        age_r, gender_r, height_r, weight_r, ap_lo_r,
        cholesterol_r, gluc_r, smoke_r, alco_r, active_r, cardio_r
    )

    st.metric("BMI", f"{bmi_r:.2f}")

    if st.button("Prediksi Tekanan Sistolik", type="primary"):
        predicted_ap_hi = predict_systolic(input_reg)
        label = bp_normal_label(predicted_ap_hi, ap_lo_r)
        confidence = regression_confidence(predicted_ap_hi)

        st.divider()
        show_metric_cards([
            {"label": "Prediksi Sistolik", "value": f"{predicted_ap_hi:.2f} mmHg"},
            {"label": "Label Regresi", "value": label},
            {"label": "Confidence Estimasi", "value": f"{confidence:.2%}", "delta": confidence_label(confidence)}
        ])

        if label == "Normal":
            st.success("Hasil regresi termasuk label: Normal.")
        else:
            st.warning("Hasil regresi termasuk label: Tidak Normal.")

        st.caption(
            f"Estimasi error rata-rata model adalah MAE ±{metadata['regression_mae']:.2f} mmHg. "
            "Label ini hanya output aplikasi untuk kebutuhan deployment/tugas."
        )

with tab3:
    st.subheader("Comparison Feature")
    st.write(
        "Fitur ini digunakan untuk membandingkan dua skenario input. "
        "Output yang dibandingkan meliputi risk probability, confidence level, prediksi sistolik, "
        "dan label hasil regresi Normal/Tidak Normal."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("### Skenario A")
        profile_a = render_common_inputs("comp_a", default_ap_hi=120, default_ap_lo=80)
    with right:
        st.markdown("### Skenario B")
        profile_b = render_common_inputs("comp_b", default_ap_hi=135, default_ap_lo=85)

    if st.button("Bandingkan Skenario", type="primary"):
        result_a = predict_full_profile(profile_a)
        result_b = predict_full_profile(profile_b)

        comparison_df = pd.DataFrame([
            {
                "Skenario": "A",
                "BMI": result_a["BMI"],
                "Pulse Pressure": result_a["Pulse Pressure"],
                "Prediksi Cardio": result_a["Prediksi Cardio"],
                "Risk Probability": f"{result_a['Risk Probability']:.2%}",
                "Classification Confidence": f"{result_a['Classification Confidence']:.2%}",
                "Confidence Level": result_a["Confidence Level"],
                "Prediksi Sistolik": f"{result_a['Prediksi Sistolik']:.2f} mmHg",
                "Regression Label": result_a["Regression Label"],
                "Regression Confidence": f"{result_a['Regression Confidence']:.2%}",
                "Regression Confidence Level": result_a["Regression Confidence Level"]
            },
            {
                "Skenario": "B",
                "BMI": result_b["BMI"],
                "Pulse Pressure": result_b["Pulse Pressure"],
                "Prediksi Cardio": result_b["Prediksi Cardio"],
                "Risk Probability": f"{result_b['Risk Probability']:.2%}",
                "Classification Confidence": f"{result_b['Classification Confidence']:.2%}",
                "Confidence Level": result_b["Confidence Level"],
                "Prediksi Sistolik": f"{result_b['Prediksi Sistolik']:.2f} mmHg",
                "Regression Label": result_b["Regression Label"],
                "Regression Confidence": f"{result_b['Regression Confidence']:.2%}",
                "Regression Confidence Level": result_b["Regression Confidence Level"]
            }
        ])

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        delta_risk = result_b["Risk Probability"] - result_a["Risk Probability"]
        delta_systolic = result_b["Prediksi Sistolik"] - result_a["Prediksi Sistolik"]

        show_metric_cards([
            {"label": "Selisih Risk Probability B - A", "value": f"{delta_risk:.2%}"},
            {"label": "Selisih Prediksi Sistolik B - A", "value": f"{delta_systolic:.2f} mmHg"},
            {"label": "Kesimpulan Cepat", "value": "B lebih tinggi" if delta_risk > 0 else "A lebih tinggi / sama"}
        ])

st.caption("Catatan: Aplikasi ini untuk tugas/edukasi, bukan pengganti diagnosis dokter atau tenaga kesehatan.")
