"""Download MIT-BIH records, wavelet-denoise, segment annotated beats, save arrays."""
import argparse
from pathlib import Path
import json
import numpy as np
import time
import wfdb
import pywt
import matplotlib.pyplot as plt
from collections import Counter
from src.common import CLASS_MAP
from pathlib import Path


def load_record_with_retry(record_id, max_attempts=5):
    """Retry downloading a record when the network fails."""
    for attempt in range(1, max_attempts + 1):
        try:
            return wfdb.rdrecord(record_id, pn_dir="mitdb")
        except Exception as exc:
            print(
                f"Attempt {attempt}/{max_attempts} failed "
                f"for record {record_id}: {exc}"
            )
            if attempt == max_attempts:
                raise
            time.sleep(5 * attempt)


DEFAULT_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234"
]

def wavelet_denoise(signal, wavelet="bior2.6", level=None):
    """Wavelet threshold denoising; threshold details are an implementation choice."""
    signal = np.asarray(signal, dtype=np.float64)
    w = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(signal), w.dec_len)
    if max_level < 1:
        return signal.astype(np.float32)
    level = min(level or max_level, max_level)
    coeffs = pywt.wavedec(signal, wavelet, mode="symmetric", level=level)
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(max(len(signal), 2)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet, mode="symmetric")[:len(signal)]
    return denoised.astype(np.float32)

def extract_beats(signal, samples, symbols, record_id, before=100, after=150):
    beats, labels, records = [], [], []
    for sample, symbol in zip(samples, symbols):
        if symbol not in CLASS_MAP:
            continue
        start, end = int(sample) - before, int(sample) + after
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        if len(beat) != before + after:
            continue
        beats.append(beat)
        labels.append(CLASS_MAP[symbol])
        records.append(record_id)
    return beats, labels, records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--records", nargs="+", default=DEFAULT_RECORDS)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--wavelet", default="bior2.6", choices=["bior2.6", "db8", "sym8"])
    parser.add_argument("--compare-wavelets", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_dir = Path(
        r"E:\minor\data\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"
    )

    all_beats, all_labels, all_records = [], [], []
    wavelet_comparison = {}
    for rid in args.records:
        print(f"Loading record {rid}...")
        try:
            record_path = str(data_dir / rid)

            rec = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, "atr")
        except Exception as exc:
            print(f"WARNING: skipped {rid}: {exc}")
            continue
        signal = rec.p_signal[:, args.channel].astype(np.float32)
        if args.compare_wavelets:
            for basis in ("bior2.6", "db8", "sym8"):
                den = wavelet_denoise(signal, basis)
                # Without a clean reference signal, these are descriptive proxies,
                # not ground-truth denoising scores.
                wavelet_comparison.setdefault(basis, []).append({
                    "record": rid,
                    "mean_abs_change": float(np.mean(np.abs(signal - den))),
                    "rmse_change": float(np.sqrt(np.mean((signal - den) ** 2))),
                })
        clean = wavelet_denoise(signal, args.wavelet)
        beats, labels, records = extract_beats(clean, ann.sample, ann.symbol, rid)
        all_beats.extend(beats)
        all_labels.extend(labels)
        all_records.extend(records)
        print(f"  fs={rec.fs} Hz, beats retained={len(beats)}")

    if not all_beats:
        raise RuntimeError("No beats extracted. Check PhysioNet access and record IDs.")
    X = np.asarray(all_beats, dtype=np.float32)
    y = np.asarray(all_labels, dtype=str)
    groups = np.asarray(all_records, dtype=str)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    np.save(out / "record_ids.npy", groups)
    metadata = {
        "records_requested": args.records,
        "records_used": sorted(np.unique(groups).tolist()),
        "channel_index": args.channel,
        "sampling_frequency_hz": 360,
        "wavelet": args.wavelet,
        "beat_samples": 250,
        "samples_before": 100,
        "samples_after": 150,
        "class_counts": dict(Counter(y.tolist())),
        "label_mapping": CLASS_MAP,
        "note": "Segmentation uses reference annotations, not the paper's custom dual-slope detector."
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.compare_wavelets:
        (out / "wavelet_comparison.json").write_text(json.dumps(wavelet_comparison, indent=2))
    # Save an example plot for report/debugging.
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(X[0])
    ax.set_title(f"Example denoised beat — label {y[0]}")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(out / "example_beat.png", dpi=160)
    plt.close(fig)
    print(f"Saved {X.shape} beats to {out}; classes={dict(Counter(y.tolist()))}")

if __name__ == "__main__":
    main()
