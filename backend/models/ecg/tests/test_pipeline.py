import numpy as np
from src.prepare_data import extract_beats, wavelet_denoise
from src.features import extract_features

def test_wavelet_length():
    x = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    y = wavelet_denoise(x)
    assert len(y) == len(x)

def test_segment_length_and_mapping():
    signal = np.arange(1000, dtype=np.float32)
    beats, labels, records = extract_beats(signal, [200, 500], ["N", "V"], "100")
    assert len(beats) == 2
    assert beats[0].shape == (250,)
    assert labels == ["N", "V"]
    assert records == ["100", "100"]

def test_features_finite():
    X = np.random.randn(5, 250).astype(np.float32)
    F = extract_features(X)
    assert F.shape[0] == 5
    assert np.isfinite(F).all()
