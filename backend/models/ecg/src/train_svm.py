
"""Standalone SVM experiments for ECG beat classification."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.common import (
    load_dataset,
    split_records,
    save_metrics,
    CLASSES,
)
from src.features import extract_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ECG SVM classifiers."
    )
    parser.add_argument(
        "--data",
        default="data/processed",
        help="Directory containing processed ECG arrays.",
    )
    parser.add_argument(
        "--out",
        default="results/svm_experiment",
        help="Directory for SVM experiment outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2020,
        help="Random seed.",
    )
    return parser.parse_args()


def load_features(data_dir):
    X, y, groups = load_dataset(data_dir)

    print("Extracting ECG features...")
    features = extract_features(X)

    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    y_encoded = np.array(
        [class_to_id[label] for label in y],
        dtype=int,
    )

    print("Feature matrix shape:", features.shape)
    print("Number of records:", len(np.unique(groups)))

    return features, y_encoded, groups


def create_record_split(groups, y, seed):
    train_records, val_records, test_records = split_records(
        groups, seed=seed
    )

    train_mask = np.isin(groups, train_records)
    val_mask = np.isin(groups, val_records)
    test_mask = np.isin(groups, test_records)

    # Ensure records do not overlap between partitions.
    assert set(train_records).isdisjoint(set(val_records))
    assert set(train_records).isdisjoint(set(test_records))
    assert set(val_records).isdisjoint(set(test_records))

    print("\nSplit sizes:")

    for name, mask in (
        ("Train", train_mask),
        ("Validation", val_mask),
        ("Test", test_mask),
    ):
        print(
            f"{name}: {mask.sum():,} beats, "
            f"{len(np.unique(groups[mask]))} records"
        )
        print(
            "Class counts:",
            dict(zip(*np.unique(y[mask], return_counts=True))),
        )

    return (
        train_mask,
        val_mask,
        test_mask,
        train_records,
        val_records,
        test_records,
    )


def build_svm_model(
    variant="class_weight",
    C=2.0,
    gamma="scale",
    seed=2020,
):
    """Build an SVM pipeline with the selected imbalance strategy."""

    # Original RBF SVM baseline with balanced class weights.
    if variant == "class_weight":
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight="balanced",
                    probability=False,
                    random_state=seed,
                    cache_size=1000,
                ),
            ),
        ])

    if variant not in (
        "random_oversampling",
        "smote",
    ):
        raise ValueError(
            f"Unknown imbalance strategy: {variant}"
        )

    if variant == "random_oversampling":
        sampler = RandomOverSampler(random_state=seed)
    else:
        sampler = SMOTE(random_state=seed)

    # Linear SVM keeps oversampling experiments computationally practical.
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("sampler", sampler),
        (
            "svm",
            LinearSVC(
                C=C,
                class_weight=None,
                max_iter=10000,
                random_state=seed,
            ),
        ),
    ])


def evaluate_model(model, X, y, split_name):
    predictions = model.predict(X)

    metrics = {
        "split": split_name,
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
    }

    print(
        f"{split_name}: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )

    return metrics


def select_best_model(
    X_train,
    y_train,
    X_val,
    y_val,
    seed,
):
    """Select the best configuration using validation macro-F1."""

    candidates = []

    # RBF SVM with class weighting.
    for C in (0.5, 2.0):
        candidates.append({
            "variant": "class_weight",
            "kernel": "rbf",
            "C": C,
            "gamma": "scale",
        })

    # Linear SVM with random oversampling.
    for C in (0.5, 2.0):
        candidates.append({
            "variant": "random_oversampling",
            "kernel": "linear",
            "C": C,
            "gamma": None,
        })

    # Linear SVM with SMOTE.
    for C in (0.5, 2.0):
        candidates.append({
            "variant": "smote",
            "kernel": "linear",
            "C": C,
            "gamma": None,
        })

    best_model = None
    best_config = None
    best_score = -1.0
    results = []

    for config in candidates:
        print("\nTraining configuration:", config)

        model = build_svm_model(
            variant=config["variant"],
            C=config["C"],
            gamma=config["gamma"] or "scale",
            seed=seed,
        )

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_val,
            y_val,
            "Validation",
        )

        result = {
            **config,
            **metrics,
        }
        results.append(result)

        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_model = model
            best_config = config.copy()

    return best_model, best_config, results


def main():
    args = parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data and extract features.
    features, y, groups = load_features(args.data)

    (
        train_mask,
        val_mask,
        test_mask,
        train_records,
        val_records,
        test_records,
    ) = create_record_split(
        groups,
        y,
        args.seed,
    )

    X_train = features[train_mask]
    y_train = y[train_mask]

    X_val = features[val_mask]
    y_val = y[val_mask]

    X_test = features[test_mask]
    y_test = y[test_mask]

    # Select model using validation data only.
    best_model, best_config, validation_results = (
        select_best_model(
            X_train,
            y_train,
            X_val,
            y_val,
            args.seed,
        )
    )

    print("\nSelected configuration:")
    print(best_config)

    # Evaluate the selected model on the test set once.
    print("\nEvaluating selected model on held-out test set...")

    test_metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
        "Test",
    )

    test_predictions = best_model.predict(X_test)

    # Save metrics and confusion matrix.
    save_metrics(
        y_test,
        test_predictions,
        output_dir,
        prefix="svm",
    )

    # Save trained model.
    joblib.dump(
        best_model,
        output_dir / "svm.joblib",
    )

    # Save validation comparison.
    pd.DataFrame(validation_results).to_csv(
        output_dir / "validation_results.csv",
        index=False,
    )

    # Save experiment summary.
    summary = {
        "selected_config": best_config,
        "validation_results": validation_results,
        "test_metrics": test_metrics,
        "seed": args.seed,
        "class_order": CLASSES,
        "selection_metric": "validation_macro_f1",
        "split_method": "record_level",
    }

    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Save record split information.
    split_info = {
        "train_records": train_records.tolist(),
        "validation_records": val_records.tolist(),
        "test_records": test_records.tolist(),
    }

    (output_dir / "split_records.json").write_text(
        json.dumps(split_info, indent=2)
    )

    print("\nSVM experiment completed.")
    print("Outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
