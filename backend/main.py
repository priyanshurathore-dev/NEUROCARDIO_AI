from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.models.eeg.raw_features import extract_raw_eeg_features

BASE_DIR = Path(__file__).resolve().parent
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
        "ecg_model": "not_loaded",
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

