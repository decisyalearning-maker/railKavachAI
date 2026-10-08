from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "structural_sensor_data.csv"
MODEL_PATH = BASE / "models" / "structural_model.joblib"
METRICS_PATH = BASE / "models" / "structural_metrics.json"

FEATURES = [
    "rms_vibration_mm_s",
    "peak_acceleration_g",
    "dominant_freq_1_hz",
    "dominant_freq_2_hz",
    "rail_temperature_c",
    "gauge_deviation_mm",
    "ballast_settlement_mm",
]

def train_model(force=False):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and METRICS_PATH.exists() and not force:
        return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["fault_type"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return model, metrics

def predict_structural(input_df):
    model, _ = train_model(False)
    result = input_df.copy()
    result["predicted_fault"] = model.predict(input_df[FEATURES])
    probs = model.predict_proba(input_df[FEATURES])
    result["confidence_pct"] = (probs.max(axis=1) * 100).round(1)
    return result
