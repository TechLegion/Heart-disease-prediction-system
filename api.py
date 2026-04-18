"""
FastAPI REST backend for the Heart Disease Prediction System.

Exposes every capability currently available in the Streamlit app (`app.py`)
as JSON endpoints, so a React (or any) frontend can consume it.

Run with:
    uvicorn api:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs       (Swagger UI)
    http://localhost:8000/redoc      (ReDoc)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paths / constants (kept consistent with app.py)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "comparison_results.csv")
CV_CSV = os.path.join(RESULTS_DIR, "cv_results.csv")
ROC_FILE = os.path.join(RESULTS_DIR, "roc_data.joblib")

MODEL_FILES: dict[str, str] = {
    "Baseline ANN": "baseline_ann.joblib",
    "GA-Optimized ANN": "ga_ann.joblib",
    "PSO-Optimized ANN": "pso_ann.joblib",
    "Hybrid GA-PSO-ANN": "hybrid_ann.joblib",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "Baseline ANN": "Standard neural network — no optimization applied.",
    "GA-Optimized ANN": "Architecture and learning rate evolved by Genetic Algorithm.",
    "PSO-Optimized ANN": "Learning rate and regularization tuned by Particle Swarm Optimization.",
    "Hybrid GA-PSO-ANN": "GA selects architecture, then PSO fine-tunes training parameters.",
}

SAMPLE_PATIENTS: dict[str, dict[str, Any]] = {
    "Healthy Patient (Low Risk)": {
        "age": 35, "sex": "Female", "cp": "0 - Typical Angina",
        "trestbps": 120, "chol": 180, "fbs": "No",
        "restecg": "0 - Normal", "thalach": 170, "exang": "No",
        "oldpeak": 0.5, "slope": "0 - Upsloping", "ca": "0",
        "thal": "3 - Normal",
    },
    "At-Risk Patient (High Risk)": {
        "age": 62, "sex": "Male", "cp": "3 - Asymptomatic",
        "trestbps": 160, "chol": 290, "fbs": "Yes",
        "restecg": "1 - ST-T Wave Abnormality", "thalach": 108, "exang": "Yes",
        "oldpeak": 3.5, "slope": "2 - Downsloping", "ca": "2",
        "thal": "7 - Reversible Defect",
    },
}

FEATURE_HELP: dict[str, str] = {
    "age": "Patient age in years.",
    "sex": "Biological sex of the patient.",
    "cp": "Type of chest pain. Asymptomatic (type 3) is most associated with heart disease.",
    "trestbps": "Blood pressure (mm Hg) at rest. Normal range: 90–140.",
    "chol": "Serum cholesterol in mg/dl. Desirable: below 200.",
    "fbs": "Is fasting blood sugar above 120 mg/dl? May indicate diabetes.",
    "restecg": "Results of the resting electrocardiogram.",
    "thalach": "Highest heart rate during exercise. Lower values may indicate risk.",
    "exang": "Does exercise cause chest pain (angina)?",
    "oldpeak": "ST depression induced by exercise vs rest. Higher = possible ischaemia. Normal: 0–2.",
    "slope": "Slope of the peak exercise ST segment on the ECG.",
    "ca": "Number of major coronary vessels (0–3) visible on fluoroscopy.",
    "thal": "Thalassemia status. Reversible defect is a strong risk indicator.",
}

NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "age": (20, 65),
    "trestbps": (90, 140),
    "chol": (125, 200),
    "thalach": (100, 190),
    "oldpeak": (0.0, 2.0),
}

RISK_LABELS: dict[str, str] = {
    "age": "Age",
    "trestbps": "Blood Pressure",
    "chol": "Cholesterol",
    "thalach": "Max Heart Rate",
    "oldpeak": "ST Depression",
}

# Categorical option sets (parallel to the Streamlit selectboxes)
CP_OPTIONS = [
    "0 - Typical Angina", "1 - Atypical Angina",
    "2 - Non-anginal Pain", "3 - Asymptomatic",
]
RESTECG_OPTIONS = [
    "0 - Normal", "1 - ST-T Wave Abnormality",
    "2 - Left Ventricular Hypertrophy",
]
SLOPE_OPTIONS = ["0 - Upsloping", "1 - Flat", "2 - Downsloping"]
THAL_OPTIONS = ["3 - Normal", "6 - Fixed Defect", "7 - Reversible Defect"]
CA_OPTIONS = ["0", "1", "2", "3"]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load_model(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_scaler():
    path = os.path.join(MODELS_DIR, "scaler.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_label_encoders():
    path = os.path.join(MODELS_DIR, "label_encoders.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_feature_names():
    path = os.path.join(MODELS_DIR, "feature_names.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _available_models() -> list[str]:
    found: list[str] = []
    for name, fname in MODEL_FILES.items():
        if os.path.exists(os.path.join(MODELS_DIR, fname)):
            found.append(name)
    if not found:
        legacy = os.path.join(MODELS_DIR, "ann_model.joblib")
        if os.path.exists(legacy):
            found.append("Baseline ANN")
    return found


def _resolve_model(name: str):
    """Load the model by display name, falling back to legacy ann_model.joblib."""
    fname = MODEL_FILES.get(name, "ann_model.joblib")
    model = _load_model(fname)
    if model is None:
        model = _load_model("ann_model.joblib")
    return model


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PatientInput(BaseModel):
    """Raw clinical input from the UI (matches the Streamlit form fields)."""

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: str = Field(..., description="'Male' or 'Female'")
    cp: str | int = Field(..., description="Chest pain type 0–3 (int or labelled string)")
    trestbps: float = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=0, description="Serum cholesterol (mg/dl)")
    fbs: str | int = Field(..., description="'Yes' / 'No' (fasting blood sugar > 120)")
    restecg: str | int = Field(..., description="Resting ECG 0–2")
    thalach: float = Field(..., ge=0, description="Max heart rate achieved")
    exang: str | int = Field(..., description="'Yes' / 'No' (exercise-induced angina)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: str | int = Field(..., description="Slope of peak exercise ST 0–2")
    ca: str | int = Field(..., description="Number of major vessels 0–3")
    thal: str | int = Field(..., description="Thalassemia 3/6/7")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 62, "sex": "Male", "cp": "3 - Asymptomatic",
                "trestbps": 160, "chol": 290, "fbs": "Yes",
                "restecg": "1 - ST-T Wave Abnormality", "thalach": 108,
                "exang": "Yes", "oldpeak": 3.5, "slope": "2 - Downsloping",
                "ca": "2", "thal": "7 - Reversible Defect",
            }
        }
    }


class PredictionRequest(BaseModel):
    model_name: str = Field(
        ..., description="One of the available model names (see /api/models)"
    )
    patient: PatientInput


class RiskFactor(BaseModel):
    feature: str
    label: str
    value: float
    low: float
    high: float
    direction: str  # "above" | "below"


class PredictionResponse(BaseModel):
    model_name: str
    prediction: int  # 0 = no disease, 1 = disease
    label: str  # human-readable
    probability_disease: float
    probability_healthy: float
    risk_factors: list[RiskFactor]
    features_used: dict[str, float]  # the processed numeric features fed to the model


# ---------------------------------------------------------------------------
# Feature pipeline (mirrors app.py → page_predict submission block)
# ---------------------------------------------------------------------------
def _first_char_int(value: str | int) -> int:
    """Accepts either an int or a labelled string like '3 - Asymptomatic' → 3."""
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if not s:
        raise ValueError("empty categorical value")
    # Try full int cast first
    try:
        return int(s)
    except ValueError:
        pass
    # Otherwise take the leading int token (e.g. "3 - Asymptomatic")
    head = s.split()[0].split("-")[0].strip()
    return int(head)


def _yes_no(value: str | int) -> float:
    if isinstance(value, (int, float)):
        return 1.0 if float(value) >= 1 else 0.0
    s = str(value).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return 1.0
    if s in ("no", "n", "false", "0"):
        return 0.0
    raise ValueError(f"expected Yes/No, got {value!r}")


def _build_feature_vector(patient: PatientInput) -> tuple[np.ndarray, dict[str, float]]:
    """Transform raw patient input → scaled feature row for the model.

    Returns (scaled_array, processed_raw_dict). Raises HTTPException on failure.
    """
    scaler = _load_scaler()
    label_encoders = _load_label_encoders()
    feature_names = _load_feature_names()

    if scaler is None or label_encoders is None or feature_names is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessing artifacts missing. Run `python main.py` first.",
        )

    try:
        raw: dict[str, float] = {
            "age": float(patient.age),
            "sex": 1.0 if str(patient.sex).strip().lower() == "male" else 0.0,
            "cp": float(_first_char_int(patient.cp)),
            "trestbps": float(patient.trestbps),
            "chol": float(patient.chol),
            "fbs": _yes_no(patient.fbs),
            "restecg": float(_first_char_int(patient.restecg)),
            "thalach": float(patient.thalach),
            "exang": _yes_no(patient.exang),
            "oldpeak": float(patient.oldpeak),
            "slope": float(_first_char_int(patient.slope)),
            "ca": float(_first_char_int(patient.ca)),
            "thal": float(_first_char_int(patient.thal)),
        }
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Invalid patient field: {e}")

    # Apply label encoders exactly like app.py does
    for col_name in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        if col_name in label_encoders:
            le = label_encoders[col_name]
            val_str = str(int(raw[col_name]))
            if val_str in le.classes_:
                raw[col_name] = float(le.transform([val_str])[0])
            else:
                raw[col_name] = float(le.transform([le.classes_[0]])[0])

    feature_df = pd.DataFrame(
        [[raw[f] for f in feature_names]], columns=feature_names
    )
    scaled = scaler.transform(feature_df)
    return scaled, raw


def _compute_risk_factors(raw: dict[str, float]) -> list[RiskFactor]:
    flags: list[RiskFactor] = []
    for feat, (lo, hi) in NORMAL_RANGES.items():
        val = raw.get(feat)
        if val is None:
            continue
        if val < lo or val > hi:
            flags.append(RiskFactor(
                feature=feat,
                label=RISK_LABELS.get(feat, feat),
                value=float(val),
                low=float(lo),
                high=float(hi),
                direction="above" if val > hi else "below",
            ))
    return flags


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "REST API exposing the GA/PSO-optimized ANN models for heart disease "
        "prediction. Intended to back a React frontend."
    ),
    version="1.0.0",
)

# CORS — permissive during development so the React dev server (Vite/CRA on
# 5173 / 3000) can call us directly. Tighten `allow_origins` for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Meta / health
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Basic liveness + readiness info for the frontend to display."""
    models = _available_models()
    return {
        "status": "ok",
        "models_available": models,
        "preprocessing_ready": all([
            _load_scaler() is not None,
            _load_label_encoders() is not None,
            _load_feature_names() is not None,
        ]),
        "results_available": os.path.exists(RESULTS_CSV),
        "cv_available": os.path.exists(CV_CSV),
        "roc_available": os.path.exists(ROC_FILE),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@app.get("/api/models", tags=["models"])
def list_models() -> dict[str, Any]:
    """Return all trained models currently available on disk, with descriptions."""
    models = _available_models()
    return {
        "models": [
            {
                "name": name,
                "description": MODEL_DESCRIPTIONS.get(name, ""),
                "file": MODEL_FILES.get(name),
            }
            for name in models
        ]
    }


# ---------------------------------------------------------------------------
# Features / form metadata (lets React render the input form dynamically)
# ---------------------------------------------------------------------------
@app.get("/api/features", tags=["metadata"])
def feature_metadata() -> dict[str, Any]:
    """Everything the frontend needs to render the patient-input form."""
    return {
        "help": FEATURE_HELP,
        "normal_ranges": {k: {"low": v[0], "high": v[1]} for k, v in NORMAL_RANGES.items()},
        "risk_labels": RISK_LABELS,
        "options": {
            "sex": ["Male", "Female"],
            "cp": CP_OPTIONS,
            "fbs": ["No", "Yes"],
            "restecg": RESTECG_OPTIONS,
            "exang": ["No", "Yes"],
            "slope": SLOPE_OPTIONS,
            "ca": CA_OPTIONS,
            "thal": THAL_OPTIONS,
        },
        "ranges": {
            "age": {"min": 20, "max": 100, "default": 55, "step": 1},
            "trestbps": {"min": 80, "max": 220, "default": 130, "step": 1},
            "chol": {"min": 100, "max": 600, "default": 245, "step": 1},
            "thalach": {"min": 60, "max": 220, "default": 150, "step": 1},
            "oldpeak": {"min": 0.0, "max": 7.0, "default": 1.0, "step": 0.1},
        },
    }


@app.get("/api/samples", tags=["metadata"])
def list_sample_patients() -> dict[str, Any]:
    """Pre-filled example patients (matches the Streamlit 'Load Sample Patient' selector)."""
    return {"samples": SAMPLE_PATIENTS}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@app.post("/api/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(req: PredictionRequest) -> PredictionResponse:
    """Run a prediction for a single patient with the chosen model."""
    available = _available_models()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No trained models found. Run `python main.py` first.",
        )

    if req.model_name not in MODEL_FILES and req.model_name not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model_name}'. Available: {available}",
        )

    model = _resolve_model(req.model_name)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model file for '{req.model_name}' not found on disk.",
        )

    scaled, raw = _build_feature_vector(req.patient)

    try:
        prediction = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0]
        prob_healthy = float(proba[0])
        prob_disease = float(proba[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return PredictionResponse(
        model_name=req.model_name,
        prediction=prediction,
        label=("Heart Disease Detected" if prediction == 1
               else "No Heart Disease Detected"),
        probability_disease=prob_disease,
        probability_healthy=prob_healthy,
        risk_factors=_compute_risk_factors(raw),
        features_used=raw,
    )


# ---------------------------------------------------------------------------
# Model comparison / evaluation results
# ---------------------------------------------------------------------------
def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """pandas → JSON-safe list-of-dicts (NaN → None)."""
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/api/comparison", tags=["evaluation"])
def model_comparison() -> dict[str, Any]:
    """Single-split comparison metrics from `results/comparison_results.csv`."""
    if not os.path.exists(RESULTS_CSV):
        raise HTTPException(
            status_code=404,
            detail="Comparison results not found. Run `python main.py` first.",
        )
    df = pd.read_csv(RESULTS_CSV)
    metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
                   if c in df.columns]

    best: dict[str, Any] | None = None
    if "Accuracy" in df.columns and len(df) > 0:
        best_idx = df["Accuracy"].idxmax()
        best = {
            "model": df.loc[best_idx, "Model"],
            "accuracy": float(df.loc[best_idx, "Accuracy"]),
            "auc": (float(df.loc[best_idx, "AUC"])
                    if "AUC" in df.columns else None),
        }

    return {
        "metrics": metric_cols,
        "results": _df_to_records(df),
        "best": best,
    }


@app.get("/api/roc", tags=["evaluation"])
def roc_curves() -> dict[str, Any]:
    """ROC-curve data (fpr / tpr / auc) for each trained model."""
    if not os.path.exists(ROC_FILE):
        raise HTTPException(
            status_code=404,
            detail="ROC data not found. Run `python main.py` first.",
        )
    roc_data = joblib.load(ROC_FILE)
    curves = []
    for name, (fpr, tpr, auc) in roc_data.items():
        curves.append({
            "model": name,
            "fpr": np.asarray(fpr).tolist(),
            "tpr": np.asarray(tpr).tolist(),
            "auc": float(auc),
        })
    return {"curves": curves}


@app.get("/api/cv", tags=["evaluation"])
def cross_validation() -> dict[str, Any]:
    """5-fold stratified cross-validation summary from `results/cv_results.csv`."""
    if not os.path.exists(CV_CSV):
        raise HTTPException(
            status_code=404,
            detail="Cross-validation results not found. Run `python main.py` first.",
        )
    cv = pd.read_csv(CV_CSV)
    return {
        "results": _df_to_records(cv),
        "metrics": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
    }


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "GET  /api/health",
            "GET  /api/models",
            "GET  /api/features",
            "GET  /api/samples",
            "POST /api/predict",
            "GET  /api/comparison",
            "GET  /api/roc",
            "GET  /api/cv",
        ],
    }
