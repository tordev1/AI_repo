"""
Machine Learning Model — Random Forest Classifier
Predicts student risk level (low/medium/high) from features.
Includes training, evaluation, and prediction functions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "risk_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "gender_encoder.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_students.csv")

FEATURE_COLS = ["gender_encoded", "workload", "study_hours", "confidence", "stress", "days_to_deadline"]


def load_and_prepare_data(path=DATA_PATH):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df["gender_encoded"] = le.fit_transform(df["gender"])
    X = df[FEATURE_COLS]
    y = df["risk"]
    return X, y, le


def train_model(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=seed, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=["low", "medium", "high"])
    report = classification_report(y_test, y_pred, labels=["low", "medium", "high"])

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }
    return metrics


def save_model(clf, le):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return clf, le


def predict_risk(clf, le, gender, workload, study_hours, confidence, stress, days_to_deadline):
    gender_enc = le.transform([gender])[0]
    feature_names = ["gender_encoded", "workload", "study_hours", "confidence", "stress", "days_to_deadline"]
    features = pd.DataFrame(
        [[gender_enc, workload, study_hours, confidence, stress, days_to_deadline]],
        columns=feature_names,
    )
    prediction = clf.predict(features)[0]
    probabilities = clf.predict_proba(features)[0]
    class_labels = clf.classes_
    prob_dict = {label: round(float(prob), 3) for label, prob in zip(class_labels, probabilities)}
    return prediction, prob_dict


def get_feature_importance(clf):
    names = ["gender", "workload", "study_hours", "confidence", "stress", "days_to_deadline"]
    importances = clf.feature_importances_
    return sorted(zip(names, importances), key=lambda x: x[1], reverse=True)


def main():
    print("Loading data...")
    X, y, le = load_and_prepare_data()

    print("Training Random Forest model...")
    clf, X_train, X_test, y_train, y_test = train_model(X, y)

    print("\n--- Evaluation Metrics ---")
    metrics = evaluate_model(clf, X_test, y_test)
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print(f"\nConfusion Matrix (low / medium / high):\n{metrics['confusion_matrix']}")
    print(f"\nClassification Report:\n{metrics['report']}")

    print("\n--- Feature Importance ---")
    for name, imp in get_feature_importance(clf):
        print(f"  {name:20s}: {imp:.4f}")

    save_model(clf, le)

    # Quick test prediction
    pred, probs = predict_risk(clf, le, "female", 4, 10, 4.0, 7.5, 5)
    print(f"\nTest prediction: risk={pred}, probabilities={probs}")


if __name__ == "__main__":
    main()
