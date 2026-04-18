"""
FastAPI REST backend for the Heart Disease Prediction System (v2).

Exposes every capability in `app.py` as JSON endpoints so a React (or any)
frontend can consume it. Uses the fitted `HeartDiseasePreprocessor` so feature
encoding is shared with the training pipeline.

Run with:
    uvicorn api:app --reload --port 8000

Interactive docs (Swagger UI):  GET http://localhost:8000/docs
Alternative docs (ReDoc):        GET http://localhost:8000/redoc
OpenAPI JSON schema:             GET http://localhost:8000/openapi.json
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Make `src/` importable so joblib can find the pickled classes.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(BASE_DIR, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import data_preprocessing  # noqa: F401 — registers HeartDiseasePreprocessor for joblib
import ensemble_model  # noqa: F401 — registers SeedEnsembleANN for hybrid_ensemble.joblib


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "comparison_results.csv")
CV_CSV = os.path.join(RESULTS_DIR, "cv_results.csv")
ROC_FILE = os.path.join(RESULTS_DIR, "roc_data.joblib")

MODEL_FILES: dict[str, str] = {
    "Baseline ANN":      "baseline_ann.joblib",
    "GA-Optimized ANN":  "ga_ann.joblib",
    "PSO-Optimized ANN": "pso_ann.joblib",
    "Hybrid GA-PSO-ANN": "hybrid_ann.joblib",
    "XGBoost":           "xgboost.joblib",
    "Hybrid Ensemble":   "hybrid_ensemble.joblib",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "Baseline ANN": "Standard neural network — no optimization applied.",
    "GA-Optimized ANN": "Architecture, learning rate, L2 and activation evolved by Genetic Algorithm (CV fitness).",
    "PSO-Optimized ANN": "Learning rate, L2 and momentum tuned by Particle Swarm Optimization (CV fitness).",
    "Hybrid GA-PSO-ANN": "GA finds top-K architectures; PSO fine-tunes each; best overall wins.",
    "XGBoost": "Gradient-boosted decision trees — tree-based ceiling reference.",
    "Hybrid Ensemble": "Average of 5 Hybrid-config ANNs trained with different seeds.",
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
    "age": "Age", "trestbps": "Blood Pressure", "chol": "Cholesterol",
    "thalach": "Max Heart Rate", "oldpeak": "ST Depression",
}

CP_OPTIONS = ["0 - Typical Angina", "1 - Atypical Angina",
              "2 - Non-anginal Pain", "3 - Asymptomatic"]
RESTECG_OPTIONS = ["0 - Normal", "1 - ST-T Wave Abnormality",
                   "2 - Left Ventricular Hypertrophy"]
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
def _load_preprocessor():
    path = os.path.join(MODELS_DIR, "preprocessor.joblib")
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
    fname = MODEL_FILES.get(name, "ann_model.joblib")
    model = _load_model(fname)
    if model is None:
        model = _load_model("ann_model.joblib")
    return model


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PatientInput(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: str | int = Field(..., description="'Male'/'Female' or 0/1")
    cp: str | int
    trestbps: float = Field(..., ge=0)
    chol: float = Field(..., ge=0)
    fbs: str | int
    restecg: str | int
    thalach: float = Field(..., ge=0)
    exang: str | int
    oldpeak: float
    slope: str | int
    ca: str | int
    thal: str | int

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
    model_name: str
    patient: PatientInput


class RiskFactor(BaseModel):
    feature: str
    label: str
    value: float
    low: float
    high: float
    direction: str


class PredictionResponse(BaseModel):
    model_name: str
    prediction: int
    label: str
    probability_disease: float
    probability_healthy: float
    risk_factors: list[RiskFactor]
    features_used: dict[str, float]


# ---------------------------------------------------------------------------
# Risk factor computation (operates on raw input, not encoded features)
# ---------------------------------------------------------------------------
def _compute_risk_factors(raw_numeric: dict[str, float]) -> list[RiskFactor]:
    flags: list[RiskFactor] = []
    for feat, (lo, hi) in NORMAL_RANGES.items():
        val = raw_numeric.get(feat)
        if val is None:
            continue
        if val < lo or val > hi:
            flags.append(RiskFactor(
                feature=feat, label=RISK_LABELS.get(feat, feat),
                value=float(val), low=float(lo), high=float(hi),
                direction="above" if val > hi else "below",
            ))
    return flags


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "REST API exposing the GA/PSO-optimized ANN models. "
        "Use **/docs** for Swagger UI (try-it-out requests)."
    ),
    version="2.0.0",
    # Explicitly enable OpenAPI + Swagger; never rely on implicit defaults alone.
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health", tags=["meta"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_available": _available_models(),
        "preprocessor_ready": _load_preprocessor() is not None,
        "results_available": os.path.exists(RESULTS_CSV),
        "cv_available": os.path.exists(CV_CSV),
        "roc_available": os.path.exists(ROC_FILE),
    }


@app.get("/api/models", tags=["models"])
def list_models() -> dict[str, Any]:
    models = _available_models()
    return {
        "models": [
            {"name": n, "description": MODEL_DESCRIPTIONS.get(n, ""),
             "file": MODEL_FILES.get(n)}
            for n in models
        ]
    }


@app.get("/api/features", tags=["metadata"])
def feature_metadata() -> dict[str, Any]:
    return {
        "help": FEATURE_HELP,
        "normal_ranges": {k: {"low": v[0], "high": v[1]}
                          for k, v in NORMAL_RANGES.items()},
        "risk_labels": RISK_LABELS,
        "options": {
            "sex": ["Male", "Female"],
            "cp": CP_OPTIONS, "fbs": ["No", "Yes"],
            "restecg": RESTECG_OPTIONS, "exang": ["No", "Yes"],
            "slope": SLOPE_OPTIONS, "ca": CA_OPTIONS, "thal": THAL_OPTIONS,
        },
        "ranges": {
            "age":      {"min": 20,  "max": 100, "default": 55,  "step": 1},
            "trestbps": {"min": 80,  "max": 220, "default": 130, "step": 1},
            "chol":     {"min": 100, "max": 600, "default": 245, "step": 1},
            "thalach":  {"min": 60,  "max": 220, "default": 150, "step": 1},
            "oldpeak":  {"min": 0.0, "max": 7.0, "default": 1.0, "step": 0.1},
        },
    }


@app.get("/api/samples", tags=["metadata"])
def list_sample_patients() -> dict[str, Any]:
    return {"samples": SAMPLE_PATIENTS}


@app.post("/api/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(req: PredictionRequest) -> PredictionResponse:
    available = _available_models()
    if not available:
        raise HTTPException(
            status_code=503,
            detail="No trained models found. Run `python main.py` first.")

    preprocessor = _load_preprocessor()
    if preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessor not found. Run `python main.py` first.")

    if req.model_name not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model_name}'. Available: {available}")

    model = _resolve_model(req.model_name)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model file for '{req.model_name}' not found on disk.")

    raw_dict = req.patient.model_dump()

    try:
        scaled = preprocessor.transform_patient(raw_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not encode patient: {e}")

    try:
        prediction = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0]
        prob_healthy = float(proba[0]); prob_disease = float(proba[1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    # Compute risk factors on the user's raw numeric input.
    raw_numeric = {
        "age": float(req.patient.age),
        "trestbps": float(req.patient.trestbps),
        "chol": float(req.patient.chol),
        "thalach": float(req.patient.thalach),
        "oldpeak": float(req.patient.oldpeak),
    }

    feature_names = _load_feature_names() or []
    features_used = {
        name: float(val)
        for name, val in zip(feature_names, np.asarray(scaled).ravel().tolist())
    }

    return PredictionResponse(
        model_name=req.model_name,
        prediction=prediction,
        label=("Heart Disease Detected" if prediction == 1
               else "No Heart Disease Detected"),
        probability_disease=prob_disease,
        probability_healthy=prob_healthy,
        risk_factors=_compute_risk_factors(raw_numeric),
        features_used=features_used,
    )


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/api/comparison", tags=["evaluation"])
def model_comparison() -> dict[str, Any]:
    if not os.path.exists(RESULTS_CSV):
        raise HTTPException(status_code=404, detail="Results not found. Run main.py.")
    df = pd.read_csv(RESULTS_CSV)
    metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
                   if c in df.columns]
    best = None
    if "Accuracy" in df.columns and len(df) > 0:
        idx = df["Accuracy"].idxmax()
        best = {
            "model": df.loc[idx, "Model"],
            "accuracy": float(df.loc[idx, "Accuracy"]),
            "auc": (float(df.loc[idx, "AUC"]) if "AUC" in df.columns else None),
        }
    return {"metrics": metric_cols, "results": _df_to_records(df), "best": best}


@app.get("/api/roc", tags=["evaluation"])
def roc_curves() -> dict[str, Any]:
    if not os.path.exists(ROC_FILE):
        raise HTTPException(status_code=404, detail="ROC data not found. Run main.py.")
    roc_data = joblib.load(ROC_FILE)
    return {
        "curves": [
            {"model": name, "fpr": np.asarray(fpr).tolist(),
             "tpr": np.asarray(tpr).tolist(), "auc": float(auc)}
            for name, (fpr, tpr, auc) in roc_data.items()
        ]
    }


@app.get("/api/cv", tags=["evaluation"])
def cross_validation() -> dict[str, Any]:
    if not os.path.exists(CV_CSV):
        raise HTTPException(status_code=404, detail="CV results not found. Run main.py.")
    cv = pd.read_csv(CV_CSV)
    return {
        "results": _df_to_records(cv),
        "metrics": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
    }


@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    return {
        "name": "Heart Disease Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "swagger_ui": "/docs",
        "redoc": "/redoc",
        "openapi_json": "/openapi.json",
        "endpoints": [
            "GET  /api/health", "GET  /api/models", "GET  /api/features",
            "GET  /api/samples", "POST /api/predict",
            "GET  /api/comparison", "GET  /api/roc", "GET  /api/cv",
        ],
    }
