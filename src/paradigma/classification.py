import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


class ClassifierPackage:
    def __init__(
        self,
        classifier: BaseEstimator | None = None,
        threshold: float | None = None,
        scaler: Any | None = None,
    ):
        """
        Initialize the ClassifierPackage with a classifier, threshold, and scaler.

        Parameters
        ----------
        classifier
            Trained classifier.
        threshold : float
            Classification threshold.
        scaler
            TTrained scaler (e.g., StandardScaler or MinMaxScaler).
        """
        self.classifier = classifier
        self.threshold = threshold
        self.scaler = scaler

    def transform_features(self, x) -> np.ndarray:
        """
        Transform the input features using the scaler.

        Parameters
        ----------
        x : np.ndarray
            The input features.

        Return
        ------
        np.ndarray
            Transformed features.
        """
        if not self.scaler:
            return x
        return self.scaler.transform(x)

    def update_scaler(self, x_train: np.ndarray) -> None:
        """
        Update the scaler used for feature transformation.

        Parameters
        ----------
        x_train : np.ndarray
            Training data to fit the scaler.
        """
        scaler = StandardScaler()
        self.scaler = scaler.fit(x_train)

    def predict_proba(self, x) -> float:
        """
        Make predictions using the classifier and apply the threshold.

        Parameters
        ----------
        x : np.ndarray
            The input features.

        Return
        ------
        float
            Predicted probability.

        """
        if not self.classifier:
            raise ValueError("Classifier is not loaded.")
        return self.classifier.predict_proba(x)[:, 1]

    def predict(self, x) -> int:
        """
        Make predictions using the classifier and apply the threshold.

        Parameters
        ----------
        x : np.ndarray
            The input features.

        Return
        ------
        int
            Predicted class.

        """
        if not self.classifier:
            raise ValueError("Classifier is not loaded.")
        return int(self.predict_proba(x) >= self.threshold)

    def save(self, filepath: str | Path) -> None:
        """
        Save the ClassifierPackage to a file.

        Parameters
        ----------
        filepath : str | Path
            Path to the file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path):
        """
        Load a ClassifierPackage from a file.

        Parameters
        ----------
        filepath : str | Path
            Path to the file.

        Return
        ------
        ClassifierPackage
            Loaded classifier package.
        """
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load classifier package: {e}") from e
