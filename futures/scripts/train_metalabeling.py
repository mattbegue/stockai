"""Training script for the metalabeling strategy."""

import argparse
import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from futures.config import get_universe, get_settings
from futures.data.fetcher import DataManager
from futures.metalabeling import (
    PrimarySignalGenerator,
    create_metalabels,
    get_label_statistics,
    MetaFeatureEngineering,
)
from futures.ml.features import FeatureSet
from futures.ml.models import (
    GradientBoostingModel,
    RandomForestModel,
    LogisticRegressionModel,
    ExtraTreesModel,
    EnsembleModel,
)
from futures.ml.training import WalkForwardValidator


def main():
    """Train the metalabeling model with walk-forward validation."""
    parser = argparse.ArgumentParser(description="Train metalabeling model")
    parser.add_argument(
        "--universe",
        choices=["small", "medium", "large"],
        default="small",
        help="Universe size: small (~50), medium (~150), large (~300)",
    )
    parser.add_argument(
        "--model",
        choices=["rf", "gbm", "lr", "et", "ensemble"],
        default="rf",
        help=(
            "Model type: rf (RandomForest, default), gbm (GradientBoosting), "
            "lr (LogisticRegression), ensemble (RF+GBM+LR averaged)"
        ),
    )
    parser.add_argument(
        "--features",
        type=int,
        default=10,
        help="Number of top features to use (0 = use all)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip probability calibration (use raw model probabilities)",
    )
    parser.add_argument(
        "--lr-C",
        type=float,
        default=0.1,
        help="Logistic Regression regularization strength (default: 0.1, higher = less regularization)",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default=None,
        help=(
            "Hard cutoff for training data (YYYY-MM-DD). "
            "Embargo is still applied on top of this. "
            "Use this to reserve a long clean test window for backtesting. "
            "Example: --train-end-date 2024-07-01 reserves 2025-01-01+ for testing."
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Metalabeling Strategy Training")
    print("=" * 60)

    # Configuration
    HOLDING_PERIOD = 5
    MIN_RETURN = 0.003    # 0.3% to cover costs (vertical barrier threshold)
    PROFIT_TARGET = 0.04  # 4% upper barrier (meaningful multi-day move)
    STOP_LOSS = 0.025     # 2.5% lower barrier (outside normal intraday noise)
    TRAIN_WINDOW = 252 * 2  # 2 years
    TEST_WINDOW = 63  # 1 quarter
    EMBARGO_DAYS = 90  # 3 months separation between training and backtest

    # Model settings from arguments
    TOP_N_FEATURES = args.features if args.features > 0 else None
    MODEL_TYPE = args.model
    LR_C = args.lr_C

    settings = get_settings()
    universe = get_universe(args.universe)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Top features: {TOP_N_FEATURES or 'all'}")
    print(f"\nUniverse: {universe}")

    # 1. Load data
    print("\n[1/6] Loading data...")
    dm = DataManager()
    data = dm.get_multi(universe.all_tickers, refresh=False, show_progress=True)

    print(f"Loaded {len(data)} tickers")
    print(f"  Tradeable: {sum(1 for t in universe.tradeable if t in data)}/{len(universe.tradeable)}")
    print(f"  Context: {sum(1 for t in universe.context if t in data)}/{len(universe.context)}")

    # 2. Generate primary signals
    print("\n[2/6] Generating primary signals...")
    signal_gen = PrimarySignalGenerator()
    tradeable_data = {t: data[t] for t in universe.tradeable if t in data}
    candidates = signal_gen.generate_candidates(tradeable_data, show_progress=True)
    print(f"Generated {len(candidates):,} candidate signals")

    # 3. Create labels
    print("\n[3/6] Creating labels...")
    labeled_df = create_metalabels(
        data,
        candidates,
        holding_period=HOLDING_PERIOD,
        min_return=MIN_RETURN,
        profit_target=PROFIT_TARGET,
        stop_loss=STOP_LOSS,
    )

    stats = get_label_statistics(labeled_df)
    print(f"Labeled signals: {stats['total']:,}")
    print(f"  Profitable: {stats['profitable']:,} ({stats['profitable_pct']:.1f}%)")
    print(f"  Unprofitable: {stats['unprofitable']:,} ({100-stats['profitable_pct']:.1f}%)")
    print(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"  Avg return (profitable): {stats['avg_return_profitable']:.2f}%")
    print(f"  Avg return (unprofitable): {stats['avg_return_unprofitable']:.2f}%")

    # Determine training cutoff date
    all_signal_dates = labeled_df["date"].sort_values()
    latest_signal_date = all_signal_dates.iloc[-1]

    if args.train_end_date:
        # User-specified hard cutoff — embargo is applied on top to set the test-start date
        hard_cutoff = pd.Timestamp(args.train_end_date)
        training_cutoff_date = hard_cutoff
        embargo_end_date = hard_cutoff + pd.Timedelta(days=EMBARGO_DAYS)
        print(f"\nUsing user-specified train-end-date: {hard_cutoff.date()}")
        print(f"  Embargo adds {EMBARGO_DAYS} days → clean test starts: {embargo_end_date.date()}")
    else:
        # Default: auto-embargo from the latest available signal
        training_cutoff_date = latest_signal_date - pd.Timedelta(days=EMBARGO_DAYS)
        embargo_end_date = latest_signal_date

    # Filter labeled data for training only (exclude embargo period)
    train_labeled_df = labeled_df[labeled_df["date"] <= training_cutoff_date].copy()
    print(f"\nTraining data separation:")
    print(f"  Training cutoff: {training_cutoff_date.date()}")
    print(f"  Embargo ends:    {embargo_end_date.date()}")
    print(f"  Training signals: {len(train_labeled_df):,} (excluded {len(labeled_df) - len(train_labeled_df):,} recent)")

    if len(train_labeled_df) < TRAIN_WINDOW:
        raise ValueError(
            f"Not enough training data after embargo. Have {len(train_labeled_df)} signals, "
            f"need at least {TRAIN_WINDOW}. Consider reducing EMBARGO_DAYS or collecting more data."
        )

    # 4. Create features
    print("\n[4/6] Creating features...")
    feature_eng = MetaFeatureEngineering(context_tickers=universe.context)
    feature_matrix = feature_eng.create_feature_matrix(
        train_labeled_df, data, show_progress=True  # Use filtered training data
    )

    # Combine features with labels
    feature_matrix = feature_matrix.fillna(0)
    labels = train_labeled_df["label"].values

    # P2-8 lesson: macro/credit/dollar/breadth/seasonality features belong in
    # regime gating (MarketRegimeClassifier), not in the per-signal ML model.
    # They are correlated with regime but hurt OOS signal discrimination.
    # Exclude them before feature selection so the selector never picks them up.
    MACRO_FEATURES_EXCLUDE = {
        # Credit spread (HYG vs LQD) — regime indicator, not signal predictor
        "hyg_lqd_spread_10d", "hyg_lqd_spread_20d", "hyg_percentile",
        # Dollar strength — macro headwind, captured by regime gating
        "uup_return_10d", "uup_percentile",
        # Commodity ratios — growth proxy, not per-signal edge
        "uso_gld_ratio_20d", "gld_return_10d",
        # Market breadth — regime-level, not signal-level
        "risk_breadth", "defensive_breadth", "breadth_net",
        # Seasonality — too coarse to discriminate individual signals
        "month", "is_quarter_end", "is_tax_selling_season",
        "is_january_effect", "is_summer",
    }
    excluded = [c for c in feature_matrix.columns if c in MACRO_FEATURES_EXCLUDE]
    if excluded:
        print(f"\n  Excluding {len(excluded)} macro/regime features from selection pool:")
        for f in excluded:
            print(f"    - {f}")
        feature_matrix = feature_matrix.drop(columns=excluded)

    # Create FeatureSet
    feature_set = FeatureSet(
        X=feature_matrix,
        y=pd.Series(labels, index=feature_matrix.index),
        feature_names=list(feature_matrix.columns),
    )

    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Features available for selection: {len(feature_set.feature_names)}")

    # Feature selection: train a quick model to get importance, then select top N.
    # IMPORTANT: fit only on the first TRAIN_WINDOW samples so feature selection
    # never sees data that belongs to later walk-forward test folds.
    if TOP_N_FEATURES and TOP_N_FEATURES < len(feature_set.feature_names):
        print(f"\n[4b/6] Selecting top {TOP_N_FEATURES} features...")
        from sklearn.ensemble import GradientBoostingClassifier

        selection_X = feature_set.X.iloc[:TRAIN_WINDOW].values
        selection_y = feature_set.y.iloc[:TRAIN_WINDOW].values

        quick_model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
        quick_model.fit(selection_X, selection_y)

        # Get top N feature indices
        importances = quick_model.feature_importances_
        top_indices = importances.argsort()[::-1][:TOP_N_FEATURES]
        top_feature_names = [feature_set.feature_names[i] for i in top_indices]

        print(f"  Selected features:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. {feature_set.feature_names[idx]}: {importances[idx]:.4f}")

        # Create reduced feature set
        reduced_X = feature_matrix[top_feature_names]
        feature_set = FeatureSet(
            X=reduced_X,
            y=pd.Series(labels, index=reduced_X.index),
            feature_names=top_feature_names,
        )
        print(f"  Reduced feature matrix: {reduced_X.shape}")

    # 5. Train with walk-forward validation
    print("\n[5/6] Running walk-forward validation...")
    print(f"  Train window: {TRAIN_WINDOW} samples (~{TRAIN_WINDOW // 252} years)")
    print(f"  Test window: {TEST_WINDOW} samples (~{TEST_WINDOW // 21} months)")

    # Create model based on selection
    def create_model(model_type: str, for_wfv: bool = False):
        if model_type == "rf":
            print("  Using Random Forest (robust to overfitting)")
            return RandomForestModel(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=50,
            )
        elif model_type == "lr":
            print(f"  Using Logistic Regression (C={LR_C})")
            return LogisticRegressionModel(
                C=LR_C,
                penalty="l2",
            )
        elif model_type == "et":
            print("  Using Extra Trees (random splits → wider probability distributions)")
            return ExtraTreesModel(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=50,
            )
        elif model_type == "gbm":
            print("  Using Gradient Boosting (regularized)")
            return GradientBoostingModel(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=50,
            )
        else:  # ensemble
            if for_wfv:
                # Use RF as a fast WFV proxy — ensemble is too slow to retrain on each fold
                print("  Using Random Forest as WFV proxy for ensemble")
                return RandomForestModel(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=50,
                )
            print("  Using Ensemble (RF + GBM + LR, calibrated separately)")
            return EnsembleModel(
                models=[
                    RandomForestModel(n_estimators=100, max_depth=5, min_samples_leaf=50),
                    GradientBoostingModel(
                        n_estimators=100, max_depth=4, learning_rate=0.05, min_samples_leaf=30
                    ),
                    LogisticRegressionModel(C=0.1, penalty="l2"),
                ]
            )

    # Walk-forward validation uses RF proxy when ensemble selected (too slow per fold)
    wfv_model_type = MODEL_TYPE if MODEL_TYPE != "ensemble" else "ensemble"
    model = create_model(wfv_model_type, for_wfv=True)

    validator = WalkForwardValidator(
        model=model,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        expanding=True,
    )

    try:
        results = validator.validate(feature_set, show_progress=True)
        print("\n" + results.summary())
    except ValueError as e:
        print(f"\nNot enough data for walk-forward validation: {e}")
        print("Falling back to simple train/test split...")

        # Simple train/test split
        train_set, test_set = feature_set.train_test_split(test_size=0.2)
        model.fit(train_set)
        metrics = model.evaluate(test_set)

        print(f"\nSimple Split Results:")
        print(f"  Train size: {len(train_set.X)}")
        print(f"  Test size: {len(test_set.X)}")
        print(f"  Accuracy: {metrics.accuracy:.3f}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall: {metrics.recall:.3f}")
        print(f"  F1 Score: {metrics.f1:.3f}")

    # 6. Train final model on all data, then calibrate probabilities (P2-1)
    print("\n[6/6] Training final model on all data...")
    CALIB_FRAC = 0.20  # Last 20% of training data reserved for calibration
    n_calib = int(len(feature_set.X) * CALIB_FRAC)
    n_train_final = len(feature_set.X) - n_calib

    train_final_set = FeatureSet(
        X=feature_set.X.iloc[:n_train_final],
        y=feature_set.y.iloc[:n_train_final],
        feature_names=feature_set.feature_names,
    )
    calib_set = FeatureSet(
        X=feature_set.X.iloc[n_train_final:],
        y=feature_set.y.iloc[n_train_final:],
        feature_names=feature_set.feature_names,
    )

    final_model = create_model(MODEL_TYPE, for_wfv=False)
    final_model.fit(train_final_set)

    # ET uses Platt (sigmoid) calibration — isotonic produces step-functions that
    # re-compress probabilities into discrete bands, defeating ET's wider distributions.
    calib_method = "sigmoid" if MODEL_TYPE == "et" else "isotonic"
    if args.no_calibrate:
        print("\n  Skipping calibration (--no-calibrate flag set).")
        calib_method = "none"
    else:
        print(f"\n  Calibrating probabilities ({calib_method}, n={n_calib:,} samples)...")
        final_model.calibrate(calib_set, method=calib_method)
        print("  Calibration complete.")

    # Feature importance (from base model before calibration wrapping)
    print("\nTop 15 most important features:")
    top_features = final_model.get_top_features(n=15)
    for i, (name, importance) in enumerate(top_features, 1):
        print(f"  {i:2d}. {name}: {importance:.4f}")

    # Save model using registry
    from futures.models.registry import ModelRegistry

    registry = ModelRegistry()
    model_info = {
        "feature_names": feature_set.feature_names,
        "holding_period": HOLDING_PERIOD,
        "min_return": MIN_RETURN,
        "profit_target": PROFIT_TARGET,
        "stop_loss": STOP_LOSS,
        "training_date": date.today().isoformat(),
        "training_end_date": training_cutoff_date.date().isoformat(),
        "embargo_end_date": embargo_end_date.date().isoformat(),
        "embargo_days": EMBARGO_DAYS,
        "n_samples": len(feature_set.X),
        "n_features": len(feature_set.feature_names),
        "model_type": MODEL_TYPE,
        "top_n_features": TOP_N_FEATURES,
        "is_calibrated": not args.no_calibrate,
        "calibration_method": calib_method,
        "calibration_frac": CALIB_FRAC,
        "universe_tradeable": universe.tradeable,
        "universe_context": universe.context,
    }

    model_path = registry.save_model(
        model=final_model,
        info=model_info,
        universe_size=args.universe,
        set_active=True,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return final_model, feature_set


def load_model(model_path: str = "models/metalabeling_model.pkl"):
    """Load a trained metalabeling model."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data


if __name__ == "__main__":
    main()
