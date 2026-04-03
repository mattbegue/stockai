"""ML model wrappers for trading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .features import FeatureSet


@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_importance: Optional[dict[str, float]] = None


class ModelWrapper(ABC):
    """Abstract base class for ML models."""

    def __init__(self, scale_features: bool = True):
        """Initialize model wrapper."""
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.model = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model instance."""
        pass

    def fit(self, feature_set: FeatureSet) -> "ModelWrapper":
        """
        Fit the model on training data.

        Args:
            feature_set: FeatureSet with X and y

        Returns:
            self for method chaining
        """
        if feature_set.y is None:
            raise ValueError("FeatureSet must have labels (y) for training")

        X = feature_set.X.values
        y = feature_set.y.values

        self.feature_names = feature_set.feature_names

        if self.scale_features:
            X = self.scaler.fit_transform(X)

        self.model = self._create_model()
        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, feature_set: FeatureSet) -> np.ndarray:
        """
        Make predictions.

        Args:
            feature_set: FeatureSet with X

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X = feature_set.X.values

        if self.scale_features:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def predict_proba(self, feature_set: FeatureSet) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            feature_set: FeatureSet with X

        Returns:
            Array of class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model does not support probability predictions")

        X = feature_set.X.values

        if self.scale_features:
            X = self.scaler.transform(X)

        return self.model.predict_proba(X)

    def evaluate(self, feature_set: FeatureSet) -> ModelMetrics:
        """
        Evaluate model on test data.

        Args:
            feature_set: FeatureSet with X and y

        Returns:
            ModelMetrics with evaluation results
        """
        if feature_set.y is None:
            raise ValueError("FeatureSet must have labels for evaluation")

        predictions = self.predict(feature_set)
        y_true = feature_set.y.values

        # Handle potential NaN in predictions
        valid_mask = ~np.isnan(predictions) & ~np.isnan(y_true)
        predictions = predictions[valid_mask]
        y_true = y_true[valid_mask]

        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_true, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_true, predictions, average="weighted", zero_division=0)

        # Feature importance
        importance = None
        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            feature_importance=importance,
        )

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features."""
        if not hasattr(self.model, "feature_importances_"):
            return []

        importance = list(zip(self.feature_names, self.model.feature_importances_))
        importance.sort(key=lambda x: x[1], reverse=True)

        return importance[:n]


class RandomForestModel(ModelWrapper):
    """Random Forest classifier for trading signals."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 20,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        """Initialize Random Forest model."""
        super().__init__(scale_features=scale_features)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _create_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )


class GradientBoostingModel(ModelWrapper):
    """Gradient Boosting classifier for trading signals."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        """Initialize Gradient Boosting model."""
        super().__init__(scale_features=scale_features)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _create_model(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )


class LogisticRegressionModel(ModelWrapper):
    """Logistic Regression classifier for trading signals.

    Simple, interpretable model that's resistant to overfitting.
    Good baseline and often works well when signal is weak.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        max_iter: int = 1000,
        scale_features: bool = True,
        random_state: int = 42,
    ):
        """Initialize Logistic Regression model.

        Args:
            C: Inverse regularization strength (smaller = more regularization)
            penalty: Regularization type ('l1', 'l2', 'elasticnet', or None)
            max_iter: Maximum iterations for solver
            scale_features: Whether to standardize features
            random_state: Random seed
        """
        super().__init__(scale_features=scale_features)
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.random_state = random_state

    def _create_model(self) -> LogisticRegression:
        solver = "saga" if self.penalty in ["l1", "elasticnet"] else "lbfgs"
        return LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            max_iter=self.max_iter,
            solver=solver,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N features by absolute coefficient value."""
        if not hasattr(self.model, "coef_"):
            return []

        # Use absolute coefficient values as importance
        coefs = np.abs(self.model.coef_[0])
        importance = list(zip(self.feature_names, coefs))
        importance.sort(key=lambda x: x[1], reverse=True)

        return importance[:n]


class EnsembleModel(ModelWrapper):
    """Ensemble of multiple models with voting."""

    def __init__(
        self,
        models: list[ModelWrapper],
        voting: str = "soft",  # 'hard' or 'soft'
    ):
        """Initialize ensemble model."""
        super().__init__(scale_features=False)  # Each model handles its own scaling
        self.models = models
        self.voting = voting

    def _create_model(self):
        return None  # Ensemble doesn't have a single model

    def fit(self, feature_set: FeatureSet) -> "EnsembleModel":
        """Fit all models in the ensemble."""
        for model in self.models:
            model.fit(feature_set)

        self.feature_names = feature_set.feature_names
        self.is_fitted = True

        return self

    def predict(self, feature_set: FeatureSet) -> np.ndarray:
        """Make ensemble predictions."""
        if self.voting == "hard":
            predictions = np.array([m.predict(feature_set) for m in self.models])
            # Majority vote
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int) + 1).argmax() - 1,
                axis=0,
                arr=predictions,
            )
        else:  # soft voting
            probas = np.array([m.predict_proba(feature_set) for m in self.models])
            avg_proba = probas.mean(axis=0)
            return np.argmax(avg_proba, axis=1) - 1  # Assuming classes are -1, 0, 1

    def predict_proba(self, feature_set: FeatureSet) -> np.ndarray:
        """Get averaged probabilities from all models."""
        probas = np.array([m.predict_proba(feature_set) for m in self.models])
        return probas.mean(axis=0)
