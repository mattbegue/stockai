"""Training utilities and walk-forward validation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .features import FeatureSet, FeatureEngineering
from .models import ModelWrapper, ModelMetrics


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""

    fold_metrics: list[ModelMetrics]
    predictions: pd.Series
    actuals: pd.Series
    fold_dates: list[tuple[pd.Timestamp, pd.Timestamp]]

    @property
    def mean_accuracy(self) -> float:
        return np.mean([m.accuracy for m in self.fold_metrics])

    @property
    def mean_f1(self) -> float:
        return np.mean([m.f1 for m in self.fold_metrics])

    def summary(self) -> str:
        lines = [
            f"Walk-Forward Validation Results",
            f"Number of folds: {len(self.fold_metrics)}",
            f"Mean Accuracy: {self.mean_accuracy:.3f}",
            f"Mean F1 Score: {self.mean_f1:.3f}",
            f"",
            f"Per-fold metrics:",
        ]

        for i, metrics in enumerate(self.fold_metrics):
            start, end = self.fold_dates[i]
            # Handle both Timestamp and integer indices
            if hasattr(start, "date"):
                date_str = f"{start.date()} to {end.date()}"
            else:
                date_str = f"idx {start} to {end}"
            lines.append(
                f"  Fold {i + 1} ({date_str}): "
                f"Acc={metrics.accuracy:.3f}, F1={metrics.f1:.3f}"
            )

        return "\n".join(lines)


class WalkForwardValidator:
    """
    Walk-forward validation for time series ML models.

    Trains on expanding window, tests on next period.
    """

    def __init__(
        self,
        model: ModelWrapper,
        train_window: int = 252,  # 1 year initial training
        test_window: int = 21,  # ~1 month test period
        expanding: bool = True,  # Expanding vs rolling window
    ):
        """
        Initialize walk-forward validator.

        Args:
            model: Model to validate
            train_window: Initial training window size
            test_window: Test window size
            expanding: If True, training window expands; if False, rolls
        """
        self.model = model
        self.train_window = train_window
        self.test_window = test_window
        self.expanding = expanding

    def validate(
        self,
        feature_set: FeatureSet,
        show_progress: bool = True,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            feature_set: Complete feature set
            show_progress: Show progress bar

        Returns:
            WalkForwardResult with all metrics and predictions
        """
        n_samples = len(feature_set.X)
        n_folds = (n_samples - self.train_window) // self.test_window

        if n_folds < 1:
            raise ValueError(
                f"Not enough data for walk-forward. Need at least "
                f"{self.train_window + self.test_window} samples, got {n_samples}"
            )

        fold_metrics = []
        all_predictions = []
        all_actuals = []
        fold_dates = []

        iterator = range(n_folds)
        if show_progress:
            iterator = tqdm(iterator, desc="Walk-forward validation")

        for fold in iterator:
            # Define train/test indices
            test_start = self.train_window + (fold * self.test_window)
            test_end = min(test_start + self.test_window, n_samples)

            if self.expanding:
                train_start = 0
            else:
                train_start = test_start - self.train_window

            train_end = test_start

            # Create train/test sets
            train_set = FeatureSet(
                X=feature_set.X.iloc[train_start:train_end],
                y=feature_set.y.iloc[train_start:train_end],
                feature_names=feature_set.feature_names,
            )

            test_set = FeatureSet(
                X=feature_set.X.iloc[test_start:test_end],
                y=feature_set.y.iloc[test_start:test_end],
                feature_names=feature_set.feature_names,
            )

            # Train and evaluate
            self.model.fit(train_set)
            predictions = self.model.predict(test_set)
            metrics = self.model.evaluate(test_set)

            fold_metrics.append(metrics)
            all_predictions.extend(predictions)
            all_actuals.extend(test_set.y.values)

            fold_dates.append((
                feature_set.X.index[test_start],
                feature_set.X.index[test_end - 1],
            ))

        # Create prediction series
        pred_index = feature_set.X.index[self.train_window:self.train_window + len(all_predictions)]
        predictions_series = pd.Series(all_predictions, index=pred_index)
        actuals_series = pd.Series(all_actuals, index=pred_index)

        return WalkForwardResult(
            fold_metrics=fold_metrics,
            predictions=predictions_series,
            actuals=actuals_series,
            fold_dates=fold_dates,
        )


class MLTrainer:
    """
    High-level trainer for ML trading models.

    Combines feature engineering, model training, and validation.
    """

    def __init__(
        self,
        model: ModelWrapper,
        feature_engineering: Optional[FeatureEngineering] = None,
    ):
        """
        Initialize ML trainer.

        Args:
            model: Model to train
            feature_engineering: Feature engineering pipeline
        """
        self.model = model
        self.feature_engineering = feature_engineering or FeatureEngineering()

    def train_single_stock(
        self,
        df: pd.DataFrame,
        target_horizon: int = 1,
        test_size: float = 0.2,
    ) -> tuple[ModelWrapper, ModelMetrics, FeatureSet]:
        """
        Train model on single stock data.

        Args:
            df: DataFrame with OHLCV data
            target_horizon: Days ahead for prediction target
            test_size: Fraction of data for testing

        Returns:
            Tuple of (trained model, test metrics, feature set)
        """
        # Create features
        feature_set = self.feature_engineering.create_features(
            df,
            target_horizon=target_horizon,
        )

        # Convert to classification labels
        feature_set.y = (feature_set.y > 0).astype(int) * 2 - 1  # 1 for up, -1 for down

        # Split
        train_set, test_set = feature_set.train_test_split(test_size=test_size)

        # Train
        self.model.fit(train_set)

        # Evaluate
        metrics = self.model.evaluate(test_set)

        return self.model, metrics, feature_set

    def walk_forward_validate(
        self,
        df: pd.DataFrame,
        target_horizon: int = 1,
        train_window: int = 252,
        test_window: int = 21,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation on single stock.

        Args:
            df: DataFrame with OHLCV data
            target_horizon: Days ahead for prediction target
            train_window: Initial training window
            test_window: Test period size

        Returns:
            WalkForwardResult with validation metrics
        """
        feature_set = self.feature_engineering.create_features(
            df,
            target_horizon=target_horizon,
        )

        # Convert to classification
        feature_set.y = (feature_set.y > 0).astype(int) * 2 - 1

        validator = WalkForwardValidator(
            model=self.model,
            train_window=train_window,
            test_window=test_window,
        )

        return validator.validate(feature_set)
