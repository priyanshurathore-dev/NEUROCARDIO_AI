"""Shared labels, metrics, loading, and record-level split utilities."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# AAMI-style grouping used in the paper's later four-class experiment.
# Q is excluded, as in the paper.
CLASS_MAP = {
    **{s: "N" for s in ("N", "L", "R", "e", "j")},
    **{s: "S" for s in ("A", "a", "J", "S")},
    **{s: "V" for s in ("V", "E")},
    "F": "F",
}
CLASSES = ["N", "S", "V", "F"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

def save_metrics(y_true, y_pred, out_dir, prefix="metrics"):
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASSES))),
        target_names=CLASSES, output_dict=True, zero_division=0
    )
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classes": CLASSES,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist(),
    }
    (out_dir / f"{prefix}.json").write_text(json.dumps(summary, indent=2))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=list(range(len(CLASSES))),
        display_labels=CLASSES, cmap="Blues", xticks_rotation=0
    )
    disp.figure_.tight_layout()
    disp.figure_.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close(disp.figure_)
    return summary

def load_dataset(data_dir):
    data_dir = Path(data_dir)
    X = np.load(data_dir / "X.npy")
    y = np.load(data_dir / "y.npy")
    groups = np.load(data_dir / "record_ids.npy")
    return X.astype(np.float32), y.astype(str), groups.astype(str)

def split_records(groups, seed=2020):
    """Deterministic record-level split, approximately 70/15/15."""
    from sklearn.model_selection import train_test_split
    unique = np.unique(groups)
    train, temp = train_test_split(unique, test_size=0.30, random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, random_state=seed)
    return train, val, test
