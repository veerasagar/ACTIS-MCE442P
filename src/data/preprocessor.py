"""
Fed-Intel Data Preprocessor
============================
Handles feature extraction, cleaning, normalization, and
train/test splitting for the IDS model.

Both datasets share 41 numeric features (43 columns minus Label and Attack).
No cross-dataset alignment is needed.

Usage:
    from src.data.preprocessor import Preprocessor
    preprocessor = Preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare(company_data["A"])
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import IDS_INPUT_DIM

console = Console()


class Preprocessor:
    """Cleans, normalizes, and prepares data for the IDS model."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
        self._labels_preset = False

    # ─── Cleaning ───────────────────────────────────────────────────────

    @staticmethod
    def clean(df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Clean the dataset:
        - Replace infinities with NaN
        - Fill NaN with 0
        - Clip extreme outliers (> 99.9th percentile)
        """
        df = df.copy()

        # Replace inf/-inf
        df[features] = df[features].replace([np.inf, -np.inf], np.nan)

        # Fill NaN with 0
        df[features] = df[features].fillna(0)

        # Clip extreme values per column (> 99.9th percentile)
        for col in features:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                upper = df[col].quantile(0.999)
                if upper > 0:
                    df[col] = df[col].clip(upper=upper)

        return df

    # ─── Normalization ──────────────────────────────────────────────────

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler on training data and transform."""
        self._is_fitted = True
        return self.scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using already-fitted scaler."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform first.")
        return self.scaler.transform(X)

    def set_unified_labels(self, all_labels: list):
        """
        Pre-fit the label encoder with a unified set of labels.
        This ensures all clients map the same string to the same integer.
        
        Args:
            all_labels: sorted list of ALL possible label strings across all clients
        """
        self.label_encoder.fit(sorted(all_labels))
        self._labels_preset = True

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Encode string labels to integers (using pre-fitted or auto-fitted encoder)."""
        if self._labels_preset:
            return self.label_encoder.transform(labels)
        return self.label_encoder.fit_transform(labels)

    def compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute inverse-frequency class weights for imbalanced data.
        Returns a weight per class (indexed by class ID).
        """
        from collections import Counter
        counts = Counter(y)
        n_samples = len(y)
        n_classes = self.num_classes
        weights = np.ones(n_classes, dtype=np.float32)
        for cls_id, count in counts.items():
            if cls_id < n_classes:
                weights[cls_id] = n_samples / (n_classes * count)
        return weights

    def decode_labels(self, encoded: np.ndarray) -> np.ndarray:
        """Decode integer labels back to strings."""
        return self.label_encoder.inverse_transform(encoded)

    @property
    def num_classes(self) -> int:
        """Number of unique classes."""
        return len(self.label_encoder.classes_)

    @property
    def class_names(self) -> list:
        """List of class names."""
        return list(self.label_encoder.classes_)

    # ─── Main Pipeline ──────────────────────────────────────────────────

    def prepare(
        self,
        company_info: Dict,
        test_size: float = 0.2,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline for one company's data.

        Args:
            company_info: Dict from FedIntelDataLoader.load_and_partition()
            test_size: fraction for test split
            max_samples: limit rows (useful for development)
            seed: random state

        Returns:
            (X_train, X_test, y_train, y_test) as numpy arrays
        """
        df = company_info["data"]
        features = company_info["features"]
        label_col = company_info["label_col"]

        console.print(f"\n[bold]Preprocessing {company_info['dataset']}[/bold]")

        # Optional: limit samples
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
            console.print(f"  Sampled to {max_samples:,} rows")

        # Clean
        df = self.clean(df, features)
        console.print(f"  Cleaned: {len(df):,} rows × {len(features)} features")

        # Extract X and y
        X = df[features].values.astype(np.float32)
        y_raw = df[label_col]

        # Encode labels
        y = self.encode_labels(y_raw)
        console.print(f"  Classes ({self.num_classes}): {self.class_names}")

        # Train/test split (stratified if possible)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
        except ValueError:
            # Fallback: non-stratified (some classes have <2 samples)
            console.print("  [yellow]⚠ Stratified split failed, using random split[/yellow]")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )

        # Normalize
        X_train = self.fit_transform(X_train)
        X_test = self.transform(X_test)

        console.print(
            f"  Split: train={len(X_train):,} test={len(X_test):,}"
        )
        console.print(f"  Feature dim: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    def prepare_for_fl(
        self,
        company_info: Dict,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare ALL data for FL training (no test split — FL does its own eval).

        Returns:
            (X, y) as numpy arrays
        """
        df = company_info["data"]
        features = company_info["features"]
        label_col = company_info["label_col"]

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        df = self.clean(df, features)
        X = df[features].values.astype(np.float32)
        y = self.encode_labels(df[label_col])
        X = self.fit_transform(X)

        return X, y


# ─── Standalone Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data.loader import FedIntelDataLoader

    loader = FedIntelDataLoader()
    data = loader.load_and_partition()

    for company in ["A", "B", "C"]:
        prep = Preprocessor()
        X_train, X_test, y_train, y_test = prep.prepare(
            data[company], max_samples=50000
        )
        console.print(f"[green]Company {company}: X_train={X_train.shape}, "
                       f"classes={prep.num_classes}[/green]\n")
