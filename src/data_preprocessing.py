"""
Data Preprocessing Module for Heart Disease Prediction System

This module handles loading, cleaning, and preprocessing of the UCI Heart Disease dataset
as described in the research proposal (Chapter 3.4).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class HeartDiseasePreprocessor:
    """
    Preprocesses heart disease dataset according to the methodology in the proposal.
    Handles missing values, normalization, encoding, and train/test splitting.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, file_path=None, file_paths=None):
        """
        Load heart disease dataset(s).

        Parameters:
        -----------
        file_path : str, optional
            Path to a single local dataset file.
        file_paths : list[str], optional
            Paths to multiple dataset files (e.g. all 4 UCI sites).
            When provided, the files are concatenated into one DataFrame.
            
        Returns:
        --------
        pd.DataFrame : Raw dataset
        """
        uci_columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]

        # --- Helper: read one file ---
        def _read_one(path):
            try:
                _df = pd.read_csv(path, header=None, sep=',', na_values='?')
            except Exception:
                _df = pd.read_csv(path, header=None, sep=r'\s+', na_values='?')
            if _df.shape[1] >= 14:
                _df = _df.iloc[:, :14].copy()
                _df.columns = uci_columns
            elif _df.shape[1] == 13:
                _df.columns = uci_columns[:-1]
                _df['target'] = 0
            else:
                _df.columns = uci_columns[:_df.shape[1]]
            return _df

        # --- Load ---
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
                print("Fetching dataset from UCI ML Repository using ucimlrepo...")
                heart_disease = fetch_ucirepo(id=45)
                X = heart_disease.data.features
                y = heart_disease.data.targets
                df = pd.concat([X, y], axis=1)
                if 'target' not in df.columns and len(y.columns) > 0:
                    df = df.rename(columns={y.columns[0]: 'target'})
                print("Dataset successfully fetched from UCI ML Repository")
            except Exception:
                print("ucimlrepo fetch failed. Attempting direct URL download...")
                try:
                    url = ("https://archive.ics.uci.edu/ml/machine-learning-"
                           "databases/heart-disease/processed.cleveland.data")
                    df = pd.read_csv(url, header=None, na_values='?')
                    df.columns = uci_columns
                except Exception as e2:
                    print(f"Error: {e2}")
                    raise

        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing or incomplete values in the dataset.
        According to proposal: replace with statistical measures or remove rows.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw dataset
            
        Returns:
        --------
        pd.DataFrame : Dataset with missing values handled
        """
        # UCI dataset uses '?' for missing values
        df = df.replace('?', np.nan)
        
        # Convert numeric columns
        numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values: replace with median for numeric, mode for categorical
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0, inplace=True)
        
        print(f"Missing values handled. Remaining missing: {df.isna().sum().sum()}")
        return df
    
    def remove_duplicates(self, df):
        """
        Remove duplicate and inconsistent entries.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
            
        Returns:
        --------
        pd.DataFrame : Dataset without duplicates
        """
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
        return df
    
    def encode_categorical(self, df):
        """
        Encode categorical variables into numerical format.
        Uses label encoding for ordinal categories.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
            
        Returns:
        --------
        pd.DataFrame : Dataset with encoded categorical variables
        """
        df_encoded = df.copy()
        
        # Categorical columns that need encoding
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def normalize_features(self, df, feature_cols=None):
        """
        Normalize numerical features using Z-score standardization.
        According to proposal: Min-Max Scaling or Z-score Standardization.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        feature_cols : list, optional
            List of feature columns to normalize. If None, normalizes all numeric columns.
            
        Returns:
        --------
        pd.DataFrame : Dataset with normalized features
        """
        df_normalized = df.copy()
        
        if feature_cols is None:
            # Exclude target column
            feature_cols = [col for col in df.columns if col != 'target']
            feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Normalize features
        df_normalized[feature_cols] = self.scaler.fit_transform(df_normalized[feature_cols])
        
        print(f"Normalized {len(feature_cols)} features using StandardScaler")
        return df_normalized
    
    def prepare_target(self, df):
        """
        Prepare target variable: convert to binary classification (0 = no disease, 1 = disease).
        Original UCI dataset has values 0-4, where 0 = no disease, 1-4 = disease.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
            
        Returns:
        --------
        pd.Series : Binary target variable
        """
        if 'target' in df.columns:
            # Convert to binary: 0 = no disease, 1 = disease
            target = (df['target'] > 0).astype(int)
            print(f"Target distribution: {target.value_counts().to_dict()}")
            return target
        else:
            raise ValueError("Target column 'target' not found in dataset")
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split dataset into training and testing sets.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Features
        y : pd.Series or np.array
            Target variable
        test_size : float
            Proportion of test set (default 0.2)
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path=None, file_paths=None,
                            test_size=0.2, random_state=42):
        """
        Complete preprocessing pipeline as described in the proposal.
        
        Steps:
        1. Load data
        2. Handle missing values
        3. Remove duplicates
        4. Encode categorical variables
        5. Normalize features
        6. Prepare target variable
        7. Split into train/test sets
        
        Parameters:
        -----------
        file_path : str, optional
            Path to a single dataset file
        file_paths : list[str], optional
            Paths to multiple dataset files (combined into one)
        test_size : float
            Proportion of test set
        random_state : int
            Random seed
            
        Returns:
        --------
        tuple : (X_train, X_test, y_train, y_test, feature_names)
        """
        print("=" * 60)
        print("HEART DISEASE DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load data
        df = self.load_data(file_path=file_path, file_paths=file_paths)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 4: Encode categorical variables
        df = self.encode_categorical(df)
        
        # Step 5: Prepare target
        y = self.prepare_target(df)
        
        # Step 6: Normalize features
        feature_cols = [col for col in df.columns if col != 'target']
        X = self.normalize_features(df, feature_cols)
        X = X[feature_cols]  # Extract only feature columns
        
        # Step 7: Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size, random_state)
        
        print("=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        
        return X_train, X_test, y_train, y_test, self.feature_names


if __name__ == "__main__":
    # Example usage
    preprocessor = HeartDiseasePreprocessor()
    
    # Try to load and preprocess data
    try:
        X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline()
        print(f"\nPreprocessed data shape:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"Features: {feature_names}")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease download the UCI Heart Disease dataset manually:")
        print("URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/")
        print("Save 'processed.cleveland.data' to the 'data/' folder")
        print("Then run: preprocessor.preprocess_pipeline(file_path='data/processed.cleveland.data')")
