"""
Save trained model, scaler, and encoders for the Streamlit web app.

Run this once after training to persist the best model to disk:
    python -m src.save_model
"""

import sys
import os
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import HeartDiseasePreprocessor
from baseline_ann import BaselineANN

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def find_datasets():
    """Locate all available heart disease dataset files."""
    base = os.path.join(os.path.dirname(__file__), '..')
    search_dirs = ['heart disease', 'heart+disease', 'data']
    site_files = [
        'processed.cleveland.data', 'processed.hungarian.data',
        'processed.switzerland.data', 'processed.va.data',
    ]
    for d in search_dirs:
        found = [os.path.join(base, d, f) for f in site_files
                 if os.path.exists(os.path.join(base, d, f))]
        if found:
            return found
    # Fallback: single cleveland file
    for d in search_dirs:
        for f in ['processed.cleveland.data', 'cleveland.data']:
            c = os.path.join(base, d, f)
            if os.path.exists(c):
                return [c]
    return []


def save_model():
    """Train the baseline ANN on the full training set and save everything."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    preprocessor = HeartDiseasePreprocessor()

    paths = find_datasets()
    if len(paths) >= 2:
        print(f"Found {len(paths)} dataset files — combining all sites")
        X_train, X_test, y_train, y_test, feature_names = \
            preprocessor.preprocess_pipeline(file_paths=paths)
    elif len(paths) == 1:
        print(f"Loading dataset from: {paths[0]}")
        X_train, X_test, y_train, y_test, feature_names = \
            preprocessor.preprocess_pipeline(file_path=paths[0])
    else:
        print("No local file found. Fetching from UCI repository...")
        X_train, X_test, y_train, y_test, feature_names = \
            preprocessor.preprocess_pipeline()

    model = BaselineANN(hidden_layer_sizes=(100, 50), max_iter=500)
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    # Save the sklearn MLPClassifier directly (smaller, avoids wrapper issues)
    joblib.dump(model.model, os.path.join(MODELS_DIR, 'ann_model.joblib'))
    joblib.dump(preprocessor.scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(preprocessor.label_encoders, os.path.join(MODELS_DIR, 'label_encoders.joblib'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))

    print(f"\nSaved to {MODELS_DIR}/:")
    print("  ann_model.joblib")
    print("  scaler.joblib")
    print("  label_encoders.joblib")
    print("  feature_names.joblib")
    print(f"\nTest accuracy: {metrics['accuracy']:.4f}")


if __name__ == '__main__':
    save_model()
