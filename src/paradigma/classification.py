import numpy as np
import pickle

from pathlib import Path
from sklearn.base import BaseEstimator
from typing import Any, Optional

class ClassifierPackage:
    def __init__(self, classifier: Optional[BaseEstimator] = None, 
                 threshold: Optional[float] = None, 
                 scaler: Optional[Any] = None):
        """
        Initialize the ClassifierPackage with a classifier, threshold, and scaler.

        Parameters
        ----------
        classifier
            The trained classifier.
        threshold : float
            The classification threshold.
        scaler
            The trained scaler (e.g., StandardScaler or MinMaxScaler).
        """
        self.classifier = classifier
        self.threshold = threshold
        self.scaler = scaler

    def transform_features(self, X) -> np.ndarray:
        """
        Transform the input features using the scaler.

        Parameters
        ----------
        X : np.ndarray
            The input features.

        Return
        ------
        np.ndarray
            The transformed features.
        """
        if not self.scaler:
            return X
        return self.scaler.transform(X)

    def predict_proba(self, X) -> float:
        """
        Make predictions using the classifier and apply the threshold.

        Parameters
        ----------
        X : np.ndarray
            The input features.

        Return
        ------
        float
            The predicted probability.

        """
        if not self.classifier:
            raise ValueError("Classifier is not loaded.")
        return self.classifier.predict_proba(X)[:, 1]
    
    def predict(self, X) -> int:
        """
        Make predictions using the classifier and apply the threshold.

        Parameters
        ----------
        X : np.ndarray
            The input features.

        Return
        ------
        int
            The predicted class.

        """
        if not self.classifier:
            raise ValueError("Classifier is not loaded.")
        return int(self.predict_proba(X) >= self.threshold)
    
    def save(self, filepath: str | Path) -> None:
        """
        Save the ClassifierPackage to a file.

        Parameters
        ----------
        filepath : str
            The path to the file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path):
        """
        Load a ClassifierPackage from a file.

        Parameters
        ----------
        filepath : str
            The path to the file.

        Return
        ------
        ClassifierPackage
            The loaded classifier package.
        """
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load classifier package: {e}") from e