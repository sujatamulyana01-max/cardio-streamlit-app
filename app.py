import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cardio Health Prediction",
    page_icon="❤️",
    layout="wide"
)

CHOLESTEROL_LABELS = {
    0: "Normal",
    1: "Di atas normal",
    2: "Jauh di atas normal"
}

GLUCOSE_LABELS = {
    0: "Normal",
    1: "Di atas normal",
    2: "Jauh di atas normal"
}

YES_NO_LABELS = {
    0: "Tidak",
    1: "Ya"
}

GENDER_LABELS = {
    0: "Perempuan",
    1: "Laki-laki"
}

FEATURE_LABELS = {
    "age": "Usia",
    "gender": "Gender",
    "height": "Tinggi badan",
    "weight": "Berat badan",
    "ap_hi": "Tekanan sistolik",
    "ap_lo": "Tekanan diastolik",
    "bmi": "BMI",
    "pulse_pressure": "Pulse pressure",
    "cholesterol": "Kolesterol",
    "gluc": "Glukosa",
    "smoke": "Merokok",
    "alco": "Konsumsi alkohol",
    "active": "Aktivitas fisik",
    "cardio": "Status cardio"
}

FEATURE_NOTES = {
    "age": "Model melihat usia sebagai salah satu pola penting dari data training.",
    "height": "Tinggi badan ikut dipakai karena berkaitan dengan perhitungan BMI.",
    "weight": "Berat badan berpengaruh pada BMI dan pola kesehatan pada data.",
    "ap_hi": "Tekanan sistolik menjadi input penting untuk prediksi penyakit kardiovaskular.",
    "ap_lo": "Tekanan diastolik membantu model membaca kondisi tekanan darah.",
    "bmi": "BMI adalah hasil perhitungan dari berat dan tinggi badan.",
    "pulse_pressure": "Pulse pressure adalah selisih sistolik dan diastolik.",
    "cholesterol": "Kategori kolesterol dapat memberi sinyal tambahan pada prediksi.",
    "gluc": "Kategori glukosa digunakan sebagai informasi metabolik tambahan.",
    "smoke": "Status merokok menjadi salah satu faktor gaya hidup pada input model.",
    "alco": "Konsumsi alkohol menjadi faktor gaya hidup tambahan pada input model.",
    "active": "Aktivitas fisik digunakan untuk membaca pola gaya hidup.",
    "gender": "Gender digunakan oleh model regresi sebagai salah satu variabel input.",
    "cardio": "Status cardio dipakai sebagai input tambahan pada model regresi sistolik."
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


def bp_normal_label(age_years, gender, predicted_ap_hi, ap_lo):
    if age_years <= 1 / 12:
        sys_min, sys_max = 60, 90
        dia_min, dia_max = 20, 60
    elif age_years < 1:
        sys_min, sys_max = 87, 105
        dia_min, dia_max = 53, 66
    elif age_years <= 3:
        sys_min, sys_max = 95, 105
        dia_min, dia_max = 52, 66
    elif age_years <= 5:
        sys_min, sys_max = 95, 110
        dia_min, dia_max = 56, 70
    elif age_years <= 10:
        sys_min, sys_max = 97, 112
        dia_min, dia_max = 57, 71
    elif age_years < 18:
        sys_min, sys_max = 112, 128
        dia_min, dia_max = 66, 80
    else:
        tolerance_sys = 10
        tolerance_dia = 10
        if gender == 0:
            if age_years <= 39:
                normal_sys, normal_dia = 110, 68
            elif age_years <= 59:
                normal_sys, normal_dia = 122, 74
            else:
                normal_sys, normal_dia = 139, 68
        else:
            if age_years <= 39:
                normal_sys, normal_dia = 119, 70
            elif age_years <= 59:
                normal_sys, normal_dia = 124, 77
            else:
                normal_sys, normal_dia = 133, 69
        sys_min = normal_sys - tolerance_sys
        sys_max = normal_sys + tolerance_sys
        dia_min = normal_dia - tolerance_dia
        dia_max = normal_dia + tolerance_dia

    if sys_min <= predicted_ap_hi <= sys_max and dia_min <= ap_lo <= dia_max:
        return "Normal"
    return "Tidak Normal"


def regression_confidence(predicted_value):
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
        age = st.number_input("Usia (tahun)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, key=f"{prefix}_age")
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


def get_feature_importance_df(model, feature_names, input_df=None, top_n=6):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_).ravel()
    else:
        importances = [0] * len(feature_names)

    rows = []
    total_importance = float(sum(importances)) if sum(importances) != 0 else 1.0

    for feature, importance in zip(feature_names, importances):
        row = {
            "Feature": feature,
            "Nama Fitur": FEATURE_LABELS.get(feature, feature),
            "Importance": float(importance),
            "Kontribusi (%)": (float(importance) / total_importance) * 100,
            "Keterangan": FEATURE_NOTES.get(feature, "Fitur ini digunakan model dalam proses prediksi.")
        }
        if input_df is not None and feature in input_df.columns:
            row["Nilai Input"] = input_df.iloc[0][feature]
        rows.append(row)

    explanation_df = pd.DataFrame(rows).sort_values("Importance", ascending=False).head(top_n)
    return explanation_df.reset_index(drop=True)


def render_xai_section(title, model, feature_names, input_df, top_n=6):
    st.markdown(f"### {title}")
    st.write(
        "Bagian ini menunjukkan fitur yang paling diperhatikan model saat membuat prediksi. "
        "Nilainya berasal dari feature importance model."
    )

    explanation_df = get_feature_importance_df(model, feature_names, input_df, top_n)
    display_columns = ["Nama Fitur", "Nilai Input", "Kontribusi (%)", "Keterangan"]
    explanation_df["Kontribusi (%)"] = explanation_df["Kontribusi (%)"].round(2)

    st.dataframe(explanation_df[display_columns], use_container_width=True, hide_index=True)

    chart_df = explanation_df[["Nama Fitur", "Kontribusi (%)"]].set_index("Nama Fitur")
    st.bar_chart(chart_df)


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
    reg_status = bp_normal_label(profile["age"], profile["gender"], predicted_ap_hi, profile["ap_lo"])

    return {
        "BMI": bmi,
        "Pulse Pressure": pulse_pressure,
        "Prediksi Cardio": "Terdeteksi" if cls_pred == 1 else "Tidak Terdeteksi",
        "Risk Probability": risk_prob,
        "Classification Confidence": cls_confidence,
        "Confidence Level": confidence_label(cls_confidence),
        "Prediksi Sistolik": predicted_ap_hi,
        "Regression Status": reg_status,
        "Regression Confidence": reg_confidence,
        "Regression Confidence Level": confidence_label(reg_confidence)
    }


st.title("❤️ Prediksi Penyakit Kardiovaskular")
st.write(
    "Masukkan data kesehatan pasien, lalu sistem akan menampilkan hasil prediksi penyakit kardiovaskular, "
    "prediksi tekanan darah sistolik, confidence level, dan feature yang paling berpengaruh menurut model."
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
        st.metric("Aturan Normal", "Usia + Gender")

    st.info(
        "Confidence level klasifikasi dihitung dari probabilitas tertinggi model. "
        "Untuk regresi, confidence adalah estimasi berbasis MAE. "
        "Status tekanan darah disesuaikan dengan usia dan gender. "
        "Explainable AI memakai feature importance untuk menunjukkan feature yang paling berpengaruh pada prediksi."
    )


tab1, tab2, tab3 = st.tabs([
    "Prediksi Penyakit",
    "Prediksi Tekanan Darah Sistolik",
    "Comparison Feature"
])

with tab1:
    st.subheader("Prediksi Penyakit Kardiovaskular")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia (tahun)", min_value=0.0, max_value=100.0, value=50.0, step=0.1, key="cls_age")
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

    if st.button("Prediksi Penyakit", type="primary"):
        prediction, probability, confidence = predict_cardio(input_cls)
        st.divider()
        show_metric_cards([
            {"label": "Probabilitas Penyakit", "value": f"{probability:.2%}"},
            {"label": "Confidence", "value": f"{confidence:.2%}", "delta": confidence_label(confidence)},
            {"label": "F1 Score Model", "value": f"{metadata['classification_f1']:.2%}"}
        ])

        if prediction == 1:
            st.error("Prediksi: Terdeteksi penyakit kardiovaskular.")
        else:
            st.success("Prediksi: Tidak terdeteksi penyakit kardiovaskular.")

        st.caption(
            "Confidence menunjukkan seberapa yakin model terhadap kelas hasil prediksi. "
            "Nilai ini berbeda dari akurasi model keseluruhan."
        )

        with st.expander("Lihat Explainable AI"):
            render_xai_section(
                "Feature yang paling berpengaruh pada prediksi penyakit",
                classifier,
                metadata["classification_features"],
                input_cls
            )

with tab2:
    st.subheader("Prediksi Tekanan Darah Sistolik / ap_hi")

    col1, col2 = st.columns(2)
    with col1:
        age_r = st.number_input("Usia untuk regresi", min_value=0.0, max_value=100.0, value=50.0, step=0.1, key="reg_age")
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
        status = bp_normal_label(age_r, gender_r, predicted_ap_hi, ap_lo_r)
        confidence = regression_confidence(predicted_ap_hi)

        st.divider()
        show_metric_cards([
            {"label": "Prediksi Sistolik", "value": f"{predicted_ap_hi:.2f} mmHg"},
            {"label": "Status", "value": status},
            {"label": "Confidence Estimasi", "value": f"{confidence:.2%}", "delta": confidence_label(confidence)}
        ])

        if status == "Normal":
            st.success("Normal")
        else:
            st.warning("Tidak Normal")

        st.caption(f"Estimasi error rata-rata model adalah MAE ±{metadata['regression_mae']:.2f} mmHg.")

        with st.expander("Lihat Explainable AI"):
            render_xai_section(
                "Feature yang paling berpengaruh pada prediksi sistolik",
                regressor,
                metadata["regression_features"],
                input_reg
            )

with tab3:
    st.subheader("Comparison Feature")
    st.write(
        "Di bagian ini, comparison berarti membandingkan feature yang paling berpengaruh menurut model. "
        "Jadi bukan membandingkan dua pasien, tetapi melihat feature mana yang paling besar kontribusinya pada hasil prediksi."
    )

    profile = render_common_inputs("feature_comp", default_ap_hi=120, default_ap_lo=80)

    if st.button("Tampilkan Feature Paling Berpengaruh", type="primary"):
        input_cls, bmi_comp, pulse_pressure_comp = make_classification_input(
            profile["age"], profile["height"], profile["weight"],
            profile["ap_hi"], profile["ap_lo"], profile["cholesterol"],
            profile["gluc"], profile["smoke"], profile["alco"], profile["active"]
        )
        cls_pred, disease_prob, cls_confidence = predict_cardio(input_cls)

        input_reg, _ = make_regression_input(
            profile["age"], profile["gender"], profile["height"], profile["weight"],
            profile["ap_lo"], profile["cholesterol"], profile["gluc"],
            profile["smoke"], profile["alco"], profile["active"], cls_pred
        )
        predicted_ap_hi = predict_systolic(input_reg)
        status = bp_normal_label(profile["age"], profile["gender"], predicted_ap_hi, profile["ap_lo"])

        st.divider()
        show_metric_cards([
            {"label": "BMI", "value": f"{bmi_comp:.2f}"},
            {"label": "Pulse Pressure", "value": f"{pulse_pressure_comp} mmHg"},
            {"label": "Probabilitas Penyakit", "value": f"{disease_prob:.2%}"},
            {"label": "Confidence", "value": f"{cls_confidence:.2%}", "delta": confidence_label(cls_confidence)},
            {"label": "Prediksi Sistolik", "value": f"{predicted_ap_hi:.2f} mmHg"},
            {"label": "Status", "value": status}
        ])

        classifier_xai = get_feature_importance_df(
            classifier,
            metadata["classification_features"],
            input_cls,
            top_n=6
        )
        regressor_xai = get_feature_importance_df(
            regressor,
            metadata["regression_features"],
            input_reg,
            top_n=6
        )

        classifier_xai["Model"] = "Prediksi Penyakit"
        regressor_xai["Model"] = "Prediksi Sistolik"
        combined_xai = pd.concat([classifier_xai, regressor_xai], ignore_index=True)
        combined_xai["Kontribusi (%)"] = combined_xai["Kontribusi (%)"].round(2)

        st.markdown("### Perbandingan feature paling berpengaruh")
        st.dataframe(
            combined_xai[["Model", "Nama Fitur", "Nilai Input", "Kontribusi (%)", "Keterangan"]],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Grafik comparison feature")
        chart_df = combined_xai[["Model", "Nama Fitur", "Kontribusi (%)"]].copy()
        chart_df["Feature"] = chart_df["Model"] + " - " + chart_df["Nama Fitur"]
        st.bar_chart(chart_df.set_index("Feature")[["Kontribusi (%)"]])

        st.caption("Nilai feature importance menjelaskan pola yang dipakai model saat membuat prediksi.")
