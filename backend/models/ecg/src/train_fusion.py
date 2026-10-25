"""Stacking demonstration using out-of-fold predictions from traditional models.

For a scientifically valid full fusion experiment, generate fold predictions for every
base learner (including CNN/LSTM) from training folds only. This script implements
the traditional-model part end-to-end and intentionally does not use held-out test
predictions to train the meta-model.
"""
import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.common import load_dataset, split_records, save_metrics, CLASSES
from src.features import extract_features

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed")
    p.add_argument("--out", default="results/fusion")
    p.add_argument("--seed", type=int, default=2020)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    X, y, groups = load_dataset(args.data)
    train_records, val_records, test_records = split_records(groups, args.seed)
    m_train, m_test = np.isin(groups, train_records), np.isin(groups, test_records)
    F = extract_features(X)
    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    yi = np.array([class_to_id[v] for v in y], dtype=int)
    Xtr, ytr = F[m_train], yi[m_train]
    Xte, yte = F[m_test], yi[m_test]
    models = [
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("svm", make_pipeline(StandardScaler(), SVC(C=2, probability=True, class_weight="balanced", random_state=args.seed))),
        ("xgb", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.3,
          reg_alpha=3, reg_lambda=5, colsample_bytree=0.9, colsample_bylevel=0.9,
          objective="multi:softprob", num_class=4, eval_metric="mlogloss",
          random_state=args.seed, n_jobs=-1))
    ]
    # Stratified folds operate on training beats; for stricter group-wise OOF,
    # use StratifiedGroupKFold and verify class coverage in every fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof_blocks, test_blocks = [], []
    for name, model in models:
        print("OOF predictions:", name)
        oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba", n_jobs=None)
        model.fit(Xtr, ytr)
        test_prob = model.predict_proba(Xte)
        oof_blocks.append(oof); test_blocks.append(test_prob)
        joblib.dump(model, out / f"{name}.joblib")
    meta_X = np.concatenate(oof_blocks, axis=1)
    meta_test_X = np.concatenate(test_blocks, axis=1)
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    meta.fit(meta_X, ytr)
    pred = meta.predict(meta_test_X)
    summary = save_metrics(yte, pred, out, prefix="stacking_traditional")
    joblib.dump(meta, out / "meta_model.joblib")
    np.save(out / "meta_oof_features.npy", meta_X)
    np.save(out / "meta_test_features.npy", meta_test_X)
    print(f"Stacking accuracy={summary['accuracy']:.4f}, macro-F1={summary['macro_f1']:.4f}")
    print("NOTE: this is traditional-only stacking. To reproduce heterogeneous fusion, add CNN/LSTM OOF probabilities.")

if __name__ == "__main__":
    main()
