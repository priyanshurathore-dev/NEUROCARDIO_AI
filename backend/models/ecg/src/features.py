"""Morphological/statistical and wavelet-coefficient features."""
import numpy as np
import pywt
from scipy.stats import skew, kurtosis

def extract_features(X, wavelet="bior2.6", level=3):
    """Feature set is a documented approximation; paper doesn't list every feature."""
    rows = []
    for beat in np.asarray(X):
        x = np.asarray(beat, dtype=np.float64)
        dx = np.diff(x)
        # Morphological/statistical features
        feats = [
            np.mean(x), np.std(x), np.min(x), np.max(x),
            np.ptp(x), np.sqrt(np.mean(x*x)), np.median(x),
            skew(x), kurtosis(x),
            np.mean(np.abs(dx)), np.std(dx), np.max(np.abs(dx)),
            np.sum(dx > 0), np.sum(dx < 0),
        ]
        # Wavelet coefficient summaries at multiple scales
        coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")
        for c in coeffs:
            feats.extend([
                float(np.mean(c)), float(np.std(c)),
                float(np.sum(c*c)), float(np.mean(np.abs(c))),
                float(np.max(np.abs(c))),
            ])
        rows.append(feats)
    return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
