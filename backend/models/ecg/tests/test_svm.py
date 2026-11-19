import numpy as np
from sklearn.pipeline import Pipeline

from src.train_svm import build_svm_model


def test_class_weight_svm_pipeline():
    model = build_svm_model(variant="class_weight")

    assert isinstance(model, Pipeline)
    assert model.named_steps["scaler"] is not None
    assert model.named_steps["svm"].kernel == "rbf"
    assert model.named_steps["svm"].class_weight == "balanced"


def test_svm_can_fit_and_predict():
    X = np.random.default_rng(2020).normal(size=(100, 10))
    y = np.array([0] * 50 + [1] * 30 + [2] * 15 + [3] * 5)

    model = build_svm_model(variant="class_weight")
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert set(np.unique(predictions)).issubset(set(np.unique(y)))


def test_invalid_variant_raises_error():
    try:
        build_svm_model(variant="invalid")
    except ValueError:
        return

    raise AssertionError("Expected ValueError for invalid variant")