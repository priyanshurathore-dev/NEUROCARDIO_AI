import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

FS = 173.61
LOWCUT = 0.5
HIGHCUT = 40.0
FILTER_ORDER = 4

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


def load_raw_bonn_file(path):
    try:
        x = np.loadtxt(path, dtype=float, delimiter=None)
    except ValueError:
        x = np.loadtxt(path, dtype=float, delimiter=",")
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x[:, 0]
        elif x.shape[1] == 2:
            x = x[:, 1]
        else:
            raise ValueError("Only single-channel EEG or time+signal files are supported.")
    if x.ndim != 1 or len(x) < 1000 or not np.isfinite(x).all():
        raise ValueError("EEG must contain at least 1000 finite samples.")
    return x


def bandpass_filter(signal, fs=FS):
    x = np.asarray(signal, dtype=float)
    if not np.isfinite(x).all() or not 0 < LOWCUT < HIGHCUT < fs / 2:
        raise ValueError("Invalid EEG signal or sampling frequency.")
    sos = butter(
        FILTER_ORDER, [LOWCUT, HIGHCUT],
        btype="bandpass", fs=fs, output="sos"
    )
    return sosfiltfilt(sos, x)


def hjorth_features(x):
    var0 = float(np.var(x))
    dx = np.diff(x)
    ddx = np.diff(dx)
    var1 = float(np.var(dx))
    var2 = float(np.var(ddx))
    eps = 1e-12
    mobility = np.sqrt(max(var1, 0.0) / max(var0, eps))
    complexity = (
        np.sqrt(max(var2, 0.0) / max(var1, eps)) / max(mobility, eps)
    )
    return {
        "hjorth_activity": var0,
        "hjorth_mobility": float(mobility),
        "hjorth_complexity": float(complexity),
    }


def spectral_features(x, fs=FS):
    f, pxx = welch(x, fs=fs, nperseg=min(256, len(x)))
    if len(f) < 2 or not np.isfinite(pxx).all():
        raise ValueError("PSD calculation failed.")

    df = float(f[1] - f[0])
    mask_total = (f >= 0.5) & (f <= min(40.0, fs / 2))
    total = (
        float(np.trapezoid(pxx[mask_total], f[mask_total]))
        if mask_total.sum() >= 2 else 0.0
    )
    total = max(total, 1e-12)
    out = {}

    for name, (lo, hi) in BANDS.items():
        mask = (f >= lo) & (f <= min(hi, fs / 2))
        power = (
            float(np.trapezoid(pxx[mask], f[mask]))
            if mask.sum() >= 2
            else float(pxx[mask][0] * df) if mask.sum() == 1
            else 0.0
        )
        out[f"{name}_power"] = power
        out[f"{name}_relative_power"] = power / total

    ps = pxx / max(float(np.sum(pxx)), 1e-12)
    ps = ps[ps > 0]
    out["spectral_entropy"] = float(-np.sum(ps * np.log2(ps)))
    out["dominant_frequency"] = float(f[np.argmax(pxx)])
    centroid = float(np.sum(f * pxx) / max(np.sum(pxx), 1e-12))
    out["spectral_centroid"] = centroid
    out["spectral_bandwidth"] = float(
        np.sqrt(np.sum(((f - centroid) ** 2) * pxx) /
                max(np.sum(pxx), 1e-12))
    )
    out["peak_psd"] = float(np.max(pxx))
    return out


def wavelet_haar_features(x, max_levels=5):
    cur = np.asarray(x, dtype=float).copy()
    coeffs = []

    for level in range(1, max_levels + 1):
        if len(cur) < 2:
            break
        even, odd = cur[::2], cur[1::2]
        n = min(len(even), len(odd))
        approx = (even[:n] + odd[:n]) / np.sqrt(2)
        detail = (even[:n] - odd[:n]) / np.sqrt(2)
        coeffs.append((level, detail))
        cur = approx

    coeffs.append((len(coeffs) + 1, cur))
    energies = [float(np.sum(c * c)) for _, c in coeffs]
    total = max(sum(energies), 1e-12)
    out = {}

    for (level, _), energy in zip(coeffs, energies):
        out[f"wavelet_L{level}_energy"] = energy
        out[f"wavelet_L{level}_relative_energy"] = energy / total

    return out


def permutation_entropy(x, order=3, delay=1):
    n = len(x) - delay * (order - 1)
    if n <= 0:
        return 0.0

    counts = {}
    for i in range(n):
        pattern = tuple(
            np.argsort(x[i:i + delay * order:delay], kind="stable")
        )
        counts[pattern] = counts.get(pattern, 0) + 1

    probs = np.asarray(list(counts.values()), dtype=float)
    probs /= probs.sum()
    return float(
        -np.sum(probs * np.log(probs + 1e-12)) /
        np.log(math.factorial(order))
    )


def approximate_entropy(x, m=2, r=None):
    if len(x) <= m + 1:
        return 0.0
    if r is None:
        r = 0.2 * float(np.std(x))
    if r <= 0:
        return 0.0

    def phi(mm):
        emb = np.asarray([
            x[i:i + mm] for i in range(len(x) - mm + 1)
        ])
        counts = []
        for row in emb:
            distances = np.max(np.abs(emb - row), axis=1)
            counts.append(np.mean(distances <= r))
        return float(np.mean(np.log(np.asarray(counts) + 1e-12)))

    return float(phi(m) - phi(m + 1))


def extract_epoch_features(x, fs=FS):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 16 or not np.isfinite(x).all():
        raise ValueError("Invalid EEG signal.")

    std = float(np.std(x))
    q25, q75 = np.percentile(x, [25, 75])
    signs = np.signbit(x)

    out = {
        "mean": float(np.mean(x)),
        "std": std,
        "variance": float(np.var(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.ptp(x)),
        "median": float(np.median(x)),
        "iqr": float(q75 - q25),
        "skewness": float(skew(x, bias=False)) if std > 1e-12 else 0.0,
        "kurtosis": float(kurtosis(x, bias=False)) if std > 1e-12 else 0.0,
        "zero_crossing_rate": float(np.mean(signs[1:] != signs[:-1])),
        "line_length": float(np.mean(np.abs(np.diff(x)))),
        "approximate_entropy": approximate_entropy(x),
        "permutation_entropy": permutation_entropy(x),
    }

    out.update(hjorth_features(x))
    out.update(spectral_features(x, fs))
    out.update(wavelet_haar_features(x))
    return {k: float(v) for k, v in out.items()}


def extract_raw_eeg_features(path, fs=FS):
    signal = load_raw_bonn_file(path)
    filtered = bandpass_filter(signal, fs=fs)
    return extract_epoch_features(filtered, fs=fs)
