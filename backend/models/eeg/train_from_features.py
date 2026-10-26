from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("default")

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "bonn_preprocessed_features_rebuilt.csv"
MODEL_PATH = BASE / "bonn_eeg_retrained_pipeline.joblib"
METRICS_PATH = BASE / "bonn_eeg_retrained_metrics.json"

LABEL_NAMES = {
    0: "Normal EEG",
    1: "Seizure-free",
    2: "Active seizure",
}
METADATA = {
    "file", "path", "code", "set", "medical_meaning", "condition",
    "label", "epoch_id", "subject", "subject_id", "patient",
    "patient_id", "sample_count", "duration_seconds",
}

def main():
    data = pd.read_csv(CSV_PATH)

    if "label" not in data:
        raise ValueError("CSV must contain a 'label' column.")

    data = data.dropna(subset=["label"]).copy()
    data["label"] = data["label"].astype(int)

    features = [
        c for c in data.columns
        if c not in METADATA
        and pd.api.types.is_numeric_dtype(data[c])
        and data[c].notna().any()
    ]

    if not features:
        raise ValueError("No numeric feature columns found.")

    X = data[features].replace([np.inf, -np.inf], np.nan)
    y = data["label"]

    if y.nunique() < 2:
        raise ValueError("At least two classes are required.")

    print(f"Rows: {len(data)} | Features: {len(features)}")
    print("Class counts:", y.value_counts().sort_index().to_dict())
    print("Feature columns:", features)

    # The available CSV has one row per recording. No verified patient IDs
    # are available, so this is a stratified recording-level holdout.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]),
        "RBF SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=42,
                n_jobs=-1
            )),
        ]),
        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced", random_state=42,
                n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_models = {}

    for name, pipeline in models.items():
        print(f"\nTraining: {name}", flush=True)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            candidate = clone(pipeline)
            candidate.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = candidate.predict(X_train.iloc[val_idx])
            scores.append(f1_score(
                y_train.iloc[val_idx], pred, average="macro", zero_division=0
            ))

        fitted = clone(pipeline)
        fitted.fit(X_train, y_train)
        test_pred = fitted.predict(X_test)

        result = {
            "model": name,
            "cv_macro_f1_mean": float(np.mean(scores)),
            "cv_macro_f1_std": float(np.std(scores)),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
            "test_macro_f1": float(f1_score(
                y_test, test_pred, average="macro", zero_division=0
            )),
            "classification_report": classification_report(
                y_test, test_pred,
                labels=sorted(LABEL_NAMES),
                target_names=[LABEL_NAMES[k] for k in sorted(LABEL_NAMES)],
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_test, test_pred, labels=sorted(LABEL_NAMES)
            ).tolist(),
        }
        results.append(result)
        fitted_models[name] = fitted
        print(
            f"CV macro-F1={result['cv_macro_f1_mean']:.4f} "
            f"(±{result['cv_macro_f1_std']:.4f}); "
            f"test accuracy={result['test_accuracy']:.4f}; "
            f"test macro-F1={result['test_macro_f1']:.4f}",
            flush=True,
        )

    results.sort(key=lambda r: (r["cv_macro_f1_mean"], r["test_macro_f1"]), reverse=True)
    best_name = results[0]["model"]
    best_pipeline = fitted_models[best_name]

    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "feature_columns": features,
        "label_names": LABEL_NAMES,
        "sampling_frequency_hz": 173.61,
        "dataset_mode": "precomputed_features",
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "python_sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "limitations": [
            "The CSV contains precomputed features; raw-signal preprocessing cannot be verified from this file alone.",
            "Patient identifiers are not verified, so patient-wise generalization is not established.",
            "This is an academic dataset experiment, not a clinically validated seizure detector.",
            "Predicted probabilities, if added later, require calibration assessment."
        ],
    }

    joblib.dump(bundle, MODEL_PATH)

    summary = {
        "dataset": CSV_PATH.name,
        "rows": int(len(data)),
        "feature_count": len(features),
        "class_counts": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
        "test_size": 0.20,
        "cv": "5-fold stratified CV on training split",
        "selection_metric": "mean CV macro-F1; test macro-F1 tie-breaker",
        "selected_model": best_name,
        "models": results,
        "limitations": bundle["limitations"],
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))

    print("\n=== MODEL COMPARISON (sorted by CV macro-F1) ===")
    for r in results:
        print(
            f"{r['model']}: CV macro-F1={r['cv_macro_f1_mean']:.4f}, "
            f"test accuracy={r['test_accuracy']:.4f}, "
            f"test macro-F1={r['test_macro_f1']:.4f}"
        )

    print("\n=== BEST MODEL TEST REPORT ===")
    print(classification_report(
        y_test,
        best_pipeline.predict(X_test),
        labels=sorted(LABEL_NAMES),
        target_names=[LABEL_NAMES[k] for k in sorted(LABEL_NAMES)],
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(
        y_test, best_pipeline.predict(X_test), labels=sorted(LABEL_NAMES)
    ))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    main()
