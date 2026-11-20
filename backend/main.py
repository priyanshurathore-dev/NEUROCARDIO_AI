from pathlib import Path

import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.models.eeg.raw_features import extract_raw_eeg_features

BASE_DIR = Path(__file__).resolve().parent
ECG_DIR = BASE_DIR / "models" / "ecg"
ECG_MODEL_PATH = ECG_DIR / "results" / "traditional" / "svm.joblib"

if str(ECG_DIR) not in sys.path:
    sys.path.insert(0, str(ECG_DIR))

from src.features import extract_features
from src.prepare_data import wavelet_denoise
EEG_MODEL_PATH = BASE_DIR / "models" / "eeg" / "bonn_eeg_retrained_pipeline.joblib"

app = FastAPI(
    title="NeuroCardio AI API",
    description="Academic EEG and ECG signal analysis backend.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

eeg_bundle = None
eeg_load_error = None

try:
    eeg_bundle = joblib.load(EEG_MODEL_PATH)
except Exception as exc:
    eeg_load_error = str(exc)

ecg_model = None
ecg_load_error = None

try:
    ecg_model = joblib.load(ECG_MODEL_PATH)
except Exception as exc:
    ecg_load_error = str(exc)


class EEGFeatureInput(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="The 45 numeric features expected by the trained EEG model.",
    )


@app.get("/")
def root():
    return {"message": "Welcome to NeuroCardio AI API", "status": "running"}


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "NeuroCardio AI API",
        "eeg_model": "loaded" if eeg_bundle is not None else "not_loaded",
        "ecg_model": "loaded" if ecg_model is not None else "not_loaded",
    }


@app.get("/api/eeg/model-info")
def eeg_model_info():
    if eeg_bundle is None:
        raise HTTPException(status_code=503, detail="EEG model could not be loaded.")
    return {
        "model_name": eeg_bundle["model_name"],
        "feature_count": len(eeg_bundle["feature_columns"]),
        "feature_columns": eeg_bundle["feature_columns"],
        "classes": eeg_bundle["label_names"],
        "input_type": "precomputed_features",
    }


@app.post("/api/eeg/predict")
def predict_eeg(payload: EEGFeatureInput):
    if eeg_bundle is None:
        raise HTTPException(status_code=503, detail="EEG model is unavailable.")

    expected = eeg_bundle["feature_columns"]
    received = payload.features

    missing = [name for name in expected if name not in received]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Missing required EEG features", "missing": missing},
        )

    try:
        values = [float(received[name]) for name in expected]
        if not np.isfinite(values).all():
            raise ValueError("All feature values must be finite numbers.")

        X = pd.DataFrame([values], columns=expected)
        model = eeg_bundle["pipeline"]
        predicted = int(model.predict(X)[0])
        label_names = eeg_bundle["label_names"]

        result = {
            "predicted_class": predicted,
            "predicted_label": label_names[predicted],
            "model_name": eeg_bundle["model_name"],
            "input_type": "precomputed_features",
            "clinical_notice": (
                "Research demonstration only; not a medical diagnosis or "
                "a substitute for professional assessment."
            ),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            result["class_scores"] = {
                str(int(cls)): float(score)
                for cls, score in zip(model.classes_, probabilities)
            }

        return result

    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="EEG prediction failed.",
        ) from exc

@app.post("/api/eeg/analyze")
async def analyze_eeg_upload(file: UploadFile = File(...)):
    """Analyze a single-channel Bonn-style EEG TXT/CSV recording."""
    if eeg_bundle is None:
        raise HTTPException(status_code=503, detail="EEG model is unavailable.")

    filename = (file.filename or "").lower()
    if not filename.endswith((".txt", ".csv")):
        raise HTTPException(
            status_code=415,
            detail="Supported formats: single-channel .txt or .csv EEG recordings.",
        )

    import tempfile
    import os

    suffix = Path(filename).suffix
    temp_path = None
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(contents)
            temp_path = temp.name

        features = extract_raw_eeg_features(temp_path)
        expected = eeg_bundle["feature_columns"]
        missing = sorted(set(expected) - set(features))
        extra = sorted(set(features) - set(expected))
        if missing or extra:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Extracted features do not match the trained model.",
                    "missing": missing,
                    "unexpected": extra,
                },
            )

        X = pd.DataFrame([[features[name] for name in expected]], columns=expected)
        model = eeg_bundle["pipeline"]
        predicted = int(model.predict(X)[0])
        label_names = eeg_bundle["label_names"]

        result = {
            "predicted_class": predicted,
            "predicted_label": label_names[predicted],
            "model_name": eeg_bundle["model_name"],
            "input_type": "raw_eeg_file",
            "features_extracted": len(features),
            "sampling_frequency_hz": 173.61,
            "clinical_notice": (
                "Research demonstration only; not a medical diagnosis or "
                "a substitute for professional assessment."
            ),
        }
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[0]
            result["class_scores"] = {
                str(int(cls)): float(score)
                for cls, score in zip(model.classes_, scores)
            }
        return result

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="EEG analysis failed."
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        await file.close()

@app.post("/api/ecg/analyze")
async def analyze_ecg_upload(file: UploadFile = File(...)):
    """Analyze a raw ECG CSV/TXT recording using the trained SVM."""

    if ecg_model is None:
        raise HTTPException(status_code=503, detail="ECG model is unavailable.")

    filename = (file.filename or "").lower()

    if not filename.endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=415,
            detail="Supported formats: ECG .csv and .txt files.",
        )

    import io
    import wfdb.processing as processing

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")

        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_csv(
                io.BytesIO(contents),
                sep=None,
                engine="python",
                header=None,
            )

        if df.empty:
            raise HTTPException(status_code=422, detail="No ECG samples found.")

        # Select the ECG signal column.
        signal_column = None

        for column in df.columns:
            if str(column).strip().lower() in ("mlii", "ecg", "lead_ii"):
                signal_column = column
                break

        if signal_column is None:
            numeric_columns = [
                column for column in df.columns
                if pd.api.types.is_numeric_dtype(df[column])
                and str(column).strip().lower() not in ("time", "time_s", "timestamp")
            ]

            if not numeric_columns:
                raise HTTPException(
                    status_code=422,
                    detail="No numeric ECG signal column found.",
                )

            signal_column = numeric_columns[0]

        signal = pd.to_numeric(
            df[signal_column], errors="coerce"
        ).dropna().to_numpy(dtype=np.float32)

        if len(signal) < 1000:
            raise HTTPException(
                status_code=422,
                detail="ECG recording is too short for analysis.",
            )

        # Determine sampling frequency from the time column when available.
        fs = 360.0

        time_column = next(
            (
                column for column in df.columns
                if str(column).strip().lower() in ("time_s", "time", "timestamp")
            ),
            None,
        )

        if time_column is not None:
            timestamps = pd.to_numeric(
                df[time_column], errors="coerce"
            ).dropna().to_numpy()

            if len(timestamps) > 1:
                interval = np.median(np.diff(timestamps))
                if interval > 0:
                    fs = 1.0 / interval

        if not 100 <= fs <= 1000:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported sampling frequency: {fs:.2f} Hz.",
            )

        # Detect heartbeat locations.
        peaks = processing.xqrs_detect(
            sig=signal,
            fs=fs,
            verbose=False,
        )

        if len(peaks) == 0:
            raise HTTPException(
                status_code=422,
                detail="No heartbeats detected in the uploaded recording.",
            )

        # Denoise and extract 250-sample heartbeat segments.
        clean_signal = wavelet_denoise(signal)

        beats = []

        for peak in peaks:
            start = int(peak) - 100
            end = int(peak) + 150

            if start >= 0 and end <= len(clean_signal):
                beats.append(clean_signal[start:end])

        if not beats:
            raise HTTPException(
                status_code=422,
                detail="No valid heartbeat segments could be extracted.",
            )

        # Extract the same 34 features used during SVM training.
        features = extract_features(np.asarray(beats, dtype=np.float32))

        predictions = ecg_model.predict(features)

        class_names = {
            0: "Normal",
            1: "Supraventricular ectopic beat",
            2: "Ventricular ectopic beat",
            3: "Fusion beat",
        }

        unique_classes, counts = np.unique(
            predictions, return_counts=True
        )

        class_counts = {
            class_names[int(cls)]: int(count)
            for cls, count in zip(unique_classes, counts)
        }

        average_scores = {}

        if hasattr(ecg_model, "predict_proba"):
            probabilities = ecg_model.predict_proba(features)
            average_scores = {
                class_names[int(cls)]: round(float(score), 4)
                for cls, score in zip(
                    ecg_model.classes_,
                    probabilities.mean(axis=0),
                )
            }

        dominant_class = int(
            unique_classes[np.argmax(counts)]
        )

        return {
            "status": "success",
            "model_name": "SVM",
            "input_type": "raw_ecg_file",
            "sampling_frequency_hz": round(float(fs), 2),
            "detected_heartbeats": int(len(peaks)),
            "analyzed_heartbeats": len(beats),
            "predominant_heartbeat_class": class_names[dominant_class],
            "heartbeat_class_counts": class_counts,
            "average_model_scores": average_scores,
            "clinical_notice": (
                "Research demonstration only. These heartbeat classifications "
                "are not a medical diagnosis or a substitute for professional assessment."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ECG analysis failed: {str(exc)}",
        ) from exc
    finally:
        await file.close()