import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("health_data.csv")

df = df.drop(columns=[c for c in ["id", "Unnamed: 0"] if c in df.columns])
df = df.drop_duplicates()
df["age"] = (df["age"] / 365).round().astype(int)

df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 220)]
df = df[(df["ap_lo"] >= 50) & (df["ap_lo"] <= 140)]
df = df[df["ap_hi"] > df["ap_lo"]]
df = df[(df["height"] >= 130) & (df["height"] <= 210)]
df = df[(df["weight"] >= 40) & (df["weight"] <= 200)]

df["bmi"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]

classification_features = [
    "age", "height", "weight", "ap_hi", "ap_lo",
    "bmi", "pulse_pressure", "cholesterol", "gluc",
    "smoke", "alco", "active"
]

X = df[classification_features]
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classifier = GradientBoostingClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

regression_features = [
    "age", "gender", "height", "weight",
    "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active", "cardio", "bmi"
]

X_reg = df[regression_features]
y_reg = df["ap_hi"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

regressor = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
regressor.fit(X_train_reg, y_train_reg)

y_pred_reg = regressor.predict(X_test_reg)

metadata = {
    "classification_features": classification_features,
    "regression_features": regression_features,
    "classification_accuracy": float(accuracy_score(y_test, y_pred)),
    "classification_f1": float(f1_score(y_test, y_pred)),
    "regression_mae": float(mean_absolute_error(y_test_reg, y_pred_reg)),
    "regression_rmse": float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))),
    "regression_r2": float(r2_score(y_test_reg, y_pred_reg)),
    "deployment_features": {
        "comparison_feature": True,
        "classification_confidence_rule": "max predicted class probability from classifier.predict_proba",
        "regression_confidence_rule": "1 - MAE / predicted_systolic, clipped to 0-1; for educational deployment only",
        "regression_label_rule": "Normal if predicted systolic < 120 and input diastolic < 80; otherwise Tidak Normal"
    }
}

joblib.dump(classifier, "cardio_classifier.pkl")
joblib.dump(regressor, "systolic_regressor.pkl")

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model berhasil dibuat.")
print(metadata)