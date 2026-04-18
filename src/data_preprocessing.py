"""
Data Preprocessing Module for Heart Disease Prediction System.

Key improvements over the previous version:
  * Raw UCI value normalisation (cp 1..4 -> 0..3, slope 1..3 -> 0..2) so that
    the training data and the Streamlit/React inference form use the SAME
    coding. The previous pipeline silently mismatched here.
  * True one-hot encoding for genuinely nominal columns (cp, restecg, slope,
    thal) instead of LabelEncoder, which was imposing fake ordinal structure.
  * `transform_patient(raw_dict)` — single entry point for single-row
    inference shared by app.py and api.py, so encoding logic lives in ONE
    place and can never drift again.
  * KNN imputation for missing values (UCI Hungarian / Switzerland / VA are
    very sparse on `ca` and `thal`; median imputation was crude).
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Column schema (fixed for the UCI heart-disease dataset)
# ---------------------------------------------------------------------------
RAW_COLUMNS: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

CONTINUOUS_COLS: list[str] = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
]
BINARY_COLS: list[str] = ["sex", "fbs", "exang"]
ONEHOT_COLS: list[str] = ["cp", "restecg", "slope", "thal"]

# Canonical categorical values after normalisation.
CATEGORIES: dict[str, list[int]] = {
    "cp":      [0, 1, 2, 3],
    "restecg": [0, 1, 2],
    "slope":   [0, 1, 2],
    "thal":    [3, 6, 7],
}


class HeartDiseasePreprocessor:
    """Fit a preprocessing pipeline once, then transform training data and
    single-patient inputs through the same transformation."""

    def __init__(self, n_neighbors: int = 5):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.feature_names: list[str] | None = None
        self.fitted: bool = False

    # ------------------------------------------------------------------ #
    # Loading                                                            #
    # ------------------------------------------------------------------ #
    def load_data(self, file_path: str | None = None,
                  file_paths: Iterable[str] | None = None) -> pd.DataFrame:
        """Load one or more UCI heart-disease files (or fetch via ucimlrepo)."""
        def _read_one(path: str) -> pd.DataFrame:
            try:
                _df = pd.read_csv(path, header=None, sep=",", na_values="?")
            except Exception:
                _df = pd.read_csv(path, header=None, sep=r"\s+", na_values="?")
            if _df.shape[1] >= 14:
                _df = _df.iloc[:, :14].copy()
                _df.columns = RAW_COLUMNS
            elif _df.shape[1] == 13:
                _df.columns = RAW_COLUMNS[:-1]
                _df["target"] = 0
            else:
                _df.columns = RAW_COLUMNS[: _df.shape[1]]
            return _df

        if file_paths:
            frames = []
            for fp in file_paths:
                part = _read_one(fp)
                print(f"  Loaded {fp}: {len(part)} rows")
                frames.append(part)
            df = pd.concat(frames, ignore_index=True)
        elif file_path:
            df = _read_one(file_path)
        else:
            try:
                from ucimlrepo import fetch_ucirepo
                print("Fetching dataset from UCI ML Repository...")
                heart = fetch_ucirepo(id=45)
                df = pd.concat([heart.data.features, heart.data.targets], axis=1)
                if "target" not in df.columns and len(heart.data.targets.columns) > 0:
                    df = df.rename(columns={heart.data.targets.columns[0]: "target"})
            except Exception as e:
                raise RuntimeError(f"Could not load dataset: {e}")

        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        return df

    # ------------------------------------------------------------------ #
    # Raw value normalisation                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_raw_values(df: pd.DataFrame) -> pd.DataFrame:
        """UCI files use 1-based indexing for `cp` and `slope`; the Streamlit
        form (and our canonical categories) use 0-based. Reconcile them here
        so the model sees the same codes at train and inference time."""
        df = df.replace("?", np.nan).copy()

        # Everything should be numeric.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # cp: UCI stores 1..4 → shift to 0..3.
        if "cp" in df.columns:
            cp_max = df["cp"].max(skipna=True)
            if pd.notna(cp_max) and cp_max >= 4:
                df["cp"] = df["cp"] - 1

        # slope: UCI stores 1..3 → shift to 0..2.
        if "slope" in df.columns:
            slope_max = df["slope"].max(skipna=True)
            if pd.notna(slope_max) and slope_max >= 3:
                df["slope"] = df["slope"] - 1

        # thal stays {3, 6, 7}; restecg stays {0, 1, 2}.
        return df

    # ------------------------------------------------------------------ #
    # Feature expansion (one-hot) — deterministic column order           #
    # ------------------------------------------------------------------ #
    @classmethod
    def _expand_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Build the final feature matrix with the canonical column order.
        Works identically for bulk training data and single-row inference."""
        out = pd.DataFrame(index=df.index)

        # Continuous columns passed through.
        for col in CONTINUOUS_COLS:
            out[col] = df[col] if col in df.columns else np.nan

        # Binary columns clamped to {0, 1}.
        for col in BINARY_COLS:
            if col in df.columns:
                out[col] = df[col]
            else:
                out[col] = np.nan

        # One-hot columns in a deterministic order based on CATEGORIES.
        for col, cats in CATEGORIES.items():
            series = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
            for c in cats:
                out[f"{col}_{c}"] = (series == c).astype(int)

        return out

    # ------------------------------------------------------------------ #
    # Fit (training path)                                                #
    # ------------------------------------------------------------------ #
    def _fit_feature_pipeline(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Turn a raw dataframe (with `target`) into a fully-processed X."""
        df = self._normalise_raw_values(df_raw)

        # Impute missing values on the raw schema BEFORE one-hot, so that
        # categorical columns still have numeric codes available.
        impute_cols = CONTINUOUS_COLS + BINARY_COLS + list(CATEGORIES.keys())
        impute_cols = [c for c in impute_cols if c in df.columns]
        imputed = self.imputer.fit_transform(df[impute_cols])
        df_imp = pd.DataFrame(imputed, columns=impute_cols, index=df.index)

        # Round imputed values in integer columns back to their integer grid.
        for col in BINARY_COLS + list(CATEGORIES.keys()):
            if col in df_imp.columns:
                df_imp[col] = df_imp[col].round()
                # Snap to nearest valid category for one-hot columns.
                if col in CATEGORIES:
                    valid = np.array(CATEGORIES[col])
                    idx = np.abs(df_imp[col].values[:, None] - valid[None, :]).argmin(axis=1)
                    df_imp[col] = valid[idx]

        initial_rows = len(df_imp)
        df_imp["target"] = df["target"].values
        df_imp = df_imp.drop_duplicates()
        removed = initial_rows - len(df_imp)
        if removed > 0:
            print(f"Removed {removed} duplicate rows")

        X_expanded = self._expand_features(df_imp)
        self.feature_names = list(X_expanded.columns)

        # Fit scaler only on truly continuous columns; binary / one-hot kept as 0/1.
        self.scaler.fit(X_expanded[CONTINUOUS_COLS].values)
        X_final = X_expanded.copy()
        X_final[CONTINUOUS_COLS] = self.scaler.transform(X_expanded[CONTINUOUS_COLS].values)

        X_final["target"] = df_imp["target"].values
        return X_final

    # ------------------------------------------------------------------ #
    # Transform (inference path) — single patient                        #
    # ------------------------------------------------------------------ #
    def transform_patient(self, raw: dict[str, Any]) -> np.ndarray:
        """Turn a single raw patient dict (as sent by the UI) into a scaled
        feature row matching `self.feature_names`."""
        if not self.fitted:
            raise RuntimeError("Preprocessor is not fitted yet.")

        df = pd.DataFrame([raw])
        # Coerce string categoricals (e.g. "3 - Asymptomatic") to int.
        for col in ONEHOT_COLS + BINARY_COLS:
            if col in df.columns:
                df[col] = df[col].map(self._coerce_int).astype(float)
        for col in CONTINUOUS_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Apply the same raw-value normalisation (in case form sends 1-based codes).
        df = self._normalise_raw_values_inference(df)

        X = self._expand_features(df)
        X = X.reindex(columns=self.feature_names, fill_value=0)

        X_scaled = X.copy()
        X_scaled[CONTINUOUS_COLS] = self.scaler.transform(X[CONTINUOUS_COLS].values)
        return X_scaled.values

    @staticmethod
    def _coerce_int(value: Any) -> int:
        """Accept '3 - Asymptomatic' / '3' / 3 / 'Yes' / 'No' / 'Male' / 'Female'."""
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float) and not np.isnan(value):
            return int(value)
        s = str(value).strip().lower()
        if s in ("yes", "y", "true"): return 1
        if s in ("no", "n", "false"): return 0
        if s == "male":   return 1
        if s == "female": return 0
        try:
            return int(s)
        except ValueError:
            pass
        head = s.split()[0].split("-")[0].strip()
        return int(head)

    @staticmethod
    def _normalise_raw_values_inference(df: pd.DataFrame) -> pd.DataFrame:
        """Inference-time version: assumes values are already 0-based from the
        UI, but keeps the door open if a caller sends 1-based codes."""
        df = df.copy()
        if "cp" in df.columns:
            vals = df["cp"].dropna()
            if not vals.empty and vals.max() >= 4:
                df["cp"] = df["cp"] - 1
        if "slope" in df.columns:
            vals = df["slope"].dropna()
            if not vals.empty and vals.max() >= 3:
                df["slope"] = df["slope"] - 1
        return df

    # ------------------------------------------------------------------ #
    # Target preparation                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def prepare_target(df: pd.DataFrame) -> pd.Series:
        if "target" not in df.columns:
            raise ValueError("'target' column not found.")
        y = (df["target"] > 0).astype(int)
        print(f"Target distribution: {y.value_counts().to_dict()}")
        return y

    # ------------------------------------------------------------------ #
    # Public pipeline                                                    #
    # ------------------------------------------------------------------ #
    def preprocess_pipeline(self, file_path: str | None = None,
                            file_paths: Iterable[str] | None = None,
                            test_size: float = 0.2,
                            random_state: int = 42):
        print("=" * 60)
        print("HEART DISEASE DATA PREPROCESSING PIPELINE (v2)")
        print("=" * 60)

        df_raw = self.load_data(file_path=file_path, file_paths=file_paths)
        df_final = self._fit_feature_pipeline(df_raw)
        self.fitted = True

        y = self.prepare_target(df_final)
        X = df_final.drop(columns=["target"])
        assert list(X.columns) == self.feature_names

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )
        print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}  "
              f"Features: {len(self.feature_names)}")
        print("=" * 60)
        return X_train, X_test, y_train, y_test, self.feature_names


if __name__ == "__main__":
    pp = HeartDiseasePreprocessor()
    try:
        X_tr, X_te, y_tr, y_te, names = pp.preprocess_pipeline()
        print("Feature columns:", names)
    except Exception as e:
        print(f"Error: {e}")
