
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.common import load_dataset, split_records, save_metrics, CLASSES
from src.features import extract_features


def make_models(seed):
    """Create comparable model configurations."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.3,
            reg_alpha=3,
            reg_lambda=5,
            colsample_bytree=0.9,
            colsample_bylevel=0.9,
            objective="multi:softprob",
            num_class=len(CLASSES),
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="results/resampling_experiment")
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load beats, labels and record IDs.
    X, y, groups = load_dataset(args.data)
    F = extract_features(X)

    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    unknown = sorted(set(y) - set(class_to_id))
    if unknown:
        raise ValueError(f"Unknown classes in dataset: {unknown}")

    y_id = np.array([class_to_id[label] for label in y], dtype=int)

    # Split complete records, not individual beats.
    train_records, val_records, test_records = split_records(
        groups, args.seed
    )

    train_mask = np.isin(groups, train_records)
    val_mask = np.isin(groups, val_records)
    test_mask = np.isin(groups, test_records)

    if not (
        train_mask.any() and val_mask.any() and test_mask.any()
    ):
        raise ValueError("Train, validation or test split is empty.")

    # Confirm no record appears in multiple splits.
    assert not (set(train_records) & set(val_records))
    assert not (set(train_records) & set(test_records))
    assert not (set(val_records) & set(test_records))

    X_train, y_train = F[train_mask], y_id[train_mask]
    X_val, y_val = F[val_mask], y_id[val_mask]
    X_test, y_test = F[test_mask], y_id[test_mask]

    print(f"Training records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Test records: {len(test_records)}")
    print(f"Training beats: {len(y_train)}")
    print(f"Validation beats: {len(y_val)}")
    print(f"Test beats: {len(y_test)}")

    print("\nOriginal training class counts:")
    for class_name, class_id in class_to_id.items():
        print(f"{class_name}: {np.sum(y_train == class_id)}")

    # Save the split so future experiments can reuse it.
    (out / "split_records.json").write_text(
        json.dumps(
            {
                "train_records": [str(v) for v in train_records],
                "validation_records": [str(v) for v in val_records],
                "test_records": [str(v) for v in test_records],
                "classes": list(CLASSES),
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Resampling is applied only to training data by the pipeline.
    # Validation and test data remain unchanged.
    experiments = [
        ("baseline", None),
        ("random_oversampling", "random"),
        ("smote", "smote"),
    ]

    results = []

    for experiment_name, sampler_name in experiments:
        print(f"\n{'=' * 60}")
        print(f"Experiment: {experiment_name}")
        print(f"{'=' * 60}")

        experiment_dir = out / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        for model_name, base_model in make_models(args.seed).items():
            print(f"\nTraining {model_name}...")

            # Create a fresh model for every experiment.
            model = base_model

            if sampler_name == "random":
                sampler = RandomOverSampler(random_state=args.seed)
                model = ImbPipeline(
                    [
                        ("sampler", sampler),
                        ("model", model),
                    ]
                )

            elif sampler_name == "smote":
                # SMOTE needs enough minority-class training examples.
                # Adapt k_neighbors if the smallest class is very small.
                counts = np.bincount(y_train, minlength=len(CLASSES))
                min_count = int(counts[counts > 0].min())

                if min_count < 2:
                    print(
                        "Skipping SMOTE: a training class has fewer "
                        "than two examples."
                    )
                    continue

                k_neighbors = min(5, min_count - 1)
                sampler = SMOTE(
                    random_state=args.seed,
                    k_neighbors=k_neighbors,
                )
                model = ImbPipeline(
                    [
                        ("sampler", sampler),
                        ("model", model),
                    ]
                )

            # Fit using training records only.
            model.fit(X_train, y_train)

            # Validation predictions are for analysis only.
            val_pred = model.predict(X_val).astype(int)
            val_balanced_acc = balanced_accuracy_score(
                y_val, val_pred
            )

            # Final evaluation on the unchanged test set.
            test_pred = model.predict(X_test).astype(int)

            model_out = experiment_dir / model_name
            summary = save_metrics(
                y_test,
                test_pred,
                model_out,
                prefix=model_name,
            )

            # Save trained model and per-class metrics.
            joblib.dump(model, experiment_dir / f"{model_name}.joblib")

            report = summary["classification_report"]
            for class_name in CLASSES:
                class_metrics = report.get(class_name, {})
                results.append(
                    {
                        "experiment": experiment_name,
                        "model": model_name,
                        "class": class_name,
                        "precision": class_metrics.get("precision", 0),
                        "recall": class_metrics.get("recall", 0),
                        "f1": class_metrics.get("f1-score", 0),
                        "support": class_metrics.get("support", 0),
                        "accuracy": summary["accuracy"],
                        "macro_f1": summary["macro_f1"],
                        "validation_balanced_accuracy": val_balanced_acc,
                        "test_balanced_accuracy": balanced_accuracy_score(
                            y_test, test_pred
                        ),
                    }
                )

            print(
                f"{model_name}: "
                f"accuracy={summary['accuracy']:.4f}, "
                f"macro-F1={summary['macro_f1']:.4f}, "
                f"validation balanced accuracy={val_balanced_acc:.4f}"
            )

    # Save a compact comparison table.
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        out / "resampling_class_metrics.csv", index=False
    )

    # One row per model/experiment, including all key metrics.
    if not results_df.empty:
        comparison = (
            results_df.groupby(["experiment", "model"], as_index=False)
            .agg(
                accuracy=("accuracy", "first"),
                macro_f1=("macro_f1", "first"),
                validation_balanced_accuracy=(
                    "validation_balanced_accuracy", "first"
                ),
                test_balanced_accuracy=("test_balanced_accuracy", "first"),
            )
        )
        comparison.to_csv(
            out / "resampling_model_comparison.csv", index=False
        )
        print("\nModel comparison:")
        print(comparison.to_string(index=False))

    print(f"\nResults saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
