"""Diagnostic analysis for the metalabeling strategy.

Analyzes:
1. Primary signal performance (without ML filtering)
2. ML model confidence distribution
3. Feature importance and predictive power
4. Signal filtering effectiveness
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from futures.config import get_universe, TickerUniverse
from futures.data.fetcher import DataManager
from futures.metalabeling import (
    PrimarySignalGenerator,
    create_metalabels,
    MetaFeatureEngineering,
)
from futures.models.registry import ModelRegistry


def analyze_primary_signals(labeled_df: pd.DataFrame) -> dict:
    """Analyze primary signal performance without ML filtering."""
    print("\n" + "=" * 60)
    print("PRIMARY SIGNAL ANALYSIS (No ML Filtering)")
    print("=" * 60)

    total = len(labeled_df)
    profitable = (labeled_df["label"] == 1).sum()
    win_rate = profitable / total * 100 if total > 0 else 0

    print(f"\nOverall Statistics:")
    print(f"  Total signals: {total:,}")
    print(f"  Profitable: {profitable:,} ({win_rate:.1f}%)")
    print(f"  Unprofitable: {total - profitable:,} ({100 - win_rate:.1f}%)")

    # By direction
    print(f"\nBy Direction:")
    for direction in [1, -1]:
        dir_name = "BUY" if direction == 1 else "SELL"
        mask = labeled_df["direction"] == direction
        dir_total = mask.sum()
        dir_profitable = (labeled_df.loc[mask, "label"] == 1).sum()
        dir_win_rate = dir_profitable / dir_total * 100 if dir_total > 0 else 0
        print(f"  {dir_name}: {dir_total:,} signals, {dir_win_rate:.1f}% win rate")

    # By source indicator
    print(f"\nBy Source Indicator:")
    indicator_stats = {}
    for idx, row in labeled_df.iterrows():
        for indicator in row["source_indicators"]:
            if indicator not in indicator_stats:
                indicator_stats[indicator] = {"total": 0, "profitable": 0}
            indicator_stats[indicator]["total"] += 1
            if row["label"] == 1:
                indicator_stats[indicator]["profitable"] += 1

    for indicator, stats in sorted(indicator_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        win_rate = stats["profitable"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {indicator}: {stats['total']:,} signals, {win_rate:.1f}% win rate")

    # By year
    print(f"\nBy Year:")
    labeled_df["year"] = pd.to_datetime(labeled_df["date"]).dt.year
    for year in sorted(labeled_df["year"].unique()):
        year_mask = labeled_df["year"] == year
        year_total = year_mask.sum()
        year_profitable = (labeled_df.loc[year_mask, "label"] == 1).sum()
        year_win_rate = year_profitable / year_total * 100 if year_total > 0 else 0
        print(f"  {year}: {year_total:,} signals, {year_win_rate:.1f}% win rate")

    # Average returns
    print(f"\nReturn Statistics:")
    avg_return = labeled_df["forward_return"].mean() * 100
    median_return = labeled_df["forward_return"].median() * 100
    std_return = labeled_df["forward_return"].std() * 100
    print(f"  Mean return: {avg_return:.2f}%")
    print(f"  Median return: {median_return:.2f}%")
    print(f"  Std return: {std_return:.2f}%")

    profitable_returns = labeled_df.loc[labeled_df["label"] == 1, "forward_return"]
    unprofitable_returns = labeled_df.loc[labeled_df["label"] == 0, "forward_return"]
    print(f"  Avg profitable return: {profitable_returns.mean() * 100:.2f}%")
    print(f"  Avg unprofitable return: {unprofitable_returns.mean() * 100:.2f}%")

    return {
        "total_signals": total,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "indicator_stats": indicator_stats,
    }


def analyze_ml_model(model, feature_set, labeled_df: pd.DataFrame) -> dict:
    """Analyze ML model predictions and confidence distribution."""
    print("\n" + "=" * 60)
    print("ML MODEL ANALYSIS")
    print("=" * 60)

    # Get predictions and probabilities
    y_true = feature_set.y.values

    y_pred = model.predict(feature_set)
    y_proba = model.predict_proba(feature_set)[:, 1]  # Probability of profitable

    # Overall accuracy
    accuracy = (y_pred == y_true).mean() * 100
    print(f"\nModel Performance (on training data - expect overfit):")
    print(f"  Accuracy: {accuracy:.1f}%")

    # Confusion matrix
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    print(f"  Precision: {precision:.1f}%")
    print(f"  Recall: {recall:.1f}%")
    print(f"  True Positives: {tp}, False Positives: {fp}")
    print(f"  True Negatives: {tn}, False Negatives: {fn}")

    # Confidence distribution
    print(f"\nConfidence Distribution:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(y_proba, p)
        print(f"  {p}th percentile: {val:.3f}")

    print(f"  Mean confidence: {y_proba.mean():.3f}")
    print(f"  Std confidence: {y_proba.std():.3f}")

    # Confidence vs actual outcome
    print(f"\nConfidence Calibration:")
    bins = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0]
    for i in range(len(bins) - 1):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() > 0:
            actual_win_rate = y_true[mask].mean() * 100
            print(f"  Confidence {bins[i]:.2f}-{bins[i+1]:.2f}: {mask.sum():,} signals, {actual_win_rate:.1f}% actual win rate")

    # Filtering analysis at different thresholds
    print(f"\nFiltering Analysis (at different confidence thresholds):")
    thresholds = [0.50, 0.52, 0.55, 0.60, 0.65]
    for thresh in thresholds:
        mask = y_proba >= thresh
        filtered_count = mask.sum()
        if filtered_count > 0:
            filtered_win_rate = y_true[mask].mean() * 100
            pct_passed = filtered_count / len(y_proba) * 100
            print(f"  Threshold {thresh:.2f}: {filtered_count:,} pass ({pct_passed:.1f}%), {filtered_win_rate:.1f}% win rate")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "mean_confidence": y_proba.mean(),
        "y_proba": y_proba,
        "y_true": y_true,
    }


def analyze_feature_importance(model, feature_names: list) -> dict:
    """Analyze feature importance."""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    # Get feature importance from model
    try:
        importances = model.model.feature_importances_
    except AttributeError:
        print("  Model does not support feature_importances_")
        return {}

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print(f"\nTop 20 Features:")
    for i, idx in enumerate(indices[:20]):
        print(f"  {i+1:2d}. {feature_names[idx]:<25} {importances[idx]:.4f}")

    print(f"\nBottom 10 Features (least important):")
    for i, idx in enumerate(indices[-10:]):
        print(f"  {len(indices) - 9 + i:2d}. {feature_names[idx]:<25} {importances[idx]:.4f}")

    # Group by category
    print(f"\nImportance by Feature Category:")
    categories = {
        "signal_quality": ["rsi", "macd", "bb_", "sma", "dist_", "above_", "num_indicators", "has_", "direction"],
        "momentum": ["return_", "volatility", "atr", "volume_ratio", "drawdown"],
        "market_context": ["spy_", "vxx_", "sector_", "tlt_hyg"],
    }

    for cat_name, patterns in categories.items():
        cat_importance = 0
        cat_count = 0
        for idx, name in enumerate(feature_names):
            if any(p in name.lower() for p in patterns):
                cat_importance += importances[idx]
                cat_count += 1
        print(f"  {cat_name}: {cat_importance:.4f} ({cat_count} features)")

    return {
        "importances": importances,
        "feature_names": feature_names,
        "top_features": [(feature_names[idx], importances[idx]) for idx in indices[:20]],
    }


def create_diagnostic_plots(output_dir: Path, ml_analysis: dict, primary_analysis: dict):
    """Create diagnostic plots."""
    output_dir.mkdir(exist_ok=True)

    # 1. Confidence distribution histogram
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Confidence distribution
    ax1 = axes[0, 0]
    y_proba = ml_analysis["y_proba"]
    y_true = ml_analysis["y_true"]

    ax1.hist(y_proba[y_true == 1], bins=30, alpha=0.6, label="Profitable", color="green")
    ax1.hist(y_proba[y_true == 0], bins=30, alpha=0.6, label="Unprofitable", color="red")
    ax1.axvline(x=0.5, color="black", linestyle="--", label="Threshold 0.5")
    ax1.axvline(x=0.52, color="orange", linestyle="--", label="Threshold 0.52")
    ax1.set_xlabel("Model Confidence")
    ax1.set_ylabel("Count")
    ax1.set_title("Confidence Distribution by Outcome")
    ax1.legend()

    # Calibration curve
    ax2 = axes[0, 1]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    actual_rates = []
    for i in range(len(bins) - 1):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() > 0:
            actual_rates.append(y_true[mask].mean())
        else:
            actual_rates.append(np.nan)

    ax2.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax2.plot(bin_centers, actual_rates, "bo-", label="Model")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Actual Win Rate")
    ax2.set_title("Calibration Curve")
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # Win rate by confidence threshold
    ax3 = axes[1, 0]
    thresholds = np.linspace(0.4, 0.7, 31)
    win_rates = []
    counts = []
    for thresh in thresholds:
        mask = y_proba >= thresh
        if mask.sum() > 0:
            win_rates.append(y_true[mask].mean() * 100)
            counts.append(mask.sum())
        else:
            win_rates.append(np.nan)
            counts.append(0)

    ax3.plot(thresholds, win_rates, "b-", linewidth=2)
    ax3.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Confidence Threshold")
    ax3.set_ylabel("Win Rate (%)")
    ax3.set_title("Win Rate vs Confidence Threshold")

    # Signals passing vs threshold
    ax4 = axes[1, 1]
    ax4.plot(thresholds, counts, "g-", linewidth=2)
    ax4.set_xlabel("Confidence Threshold")
    ax4.set_ylabel("Signals Passing Filter")
    ax4.set_title("Signal Count vs Threshold")

    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {output_dir / 'diagnostic_plots.png'}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose metalabeling strategy")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="diagnostics",
        help="Output directory for diagnostic plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("METALABELING STRATEGY DIAGNOSTICS")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    registry = ModelRegistry()
    model, model_info = registry.load_active()

    print(f"  Model: {model_info.get('model_name', 'unknown')}")
    print(f"  Training date: {model_info.get('training_date')}")
    print(f"  Training end: {model_info.get('training_end_date')}")
    print(f"  Embargo end: {model_info.get('embargo_end_date')}")
    print(f"  Samples: {model_info.get('n_samples')}")

    # Determine universe
    if "universe_tradeable" in model_info and "universe_context" in model_info:
        universe = TickerUniverse(
            tradeable=model_info["universe_tradeable"],
            context=model_info["universe_context"],
            name=model_info.get("universe_size", "from_model"),
        )
    else:
        universe = get_universe("small")

    print(f"  Universe: {len(universe.tradeable)} tradeable, {len(universe.context)} context")

    # Load data
    print("\nLoading data...")
    dm = DataManager()
    data = dm.get_multi(universe.all_tickers, refresh=False, show_progress=False)
    print(f"  Loaded {len(data)} tickers")

    # Generate signals and labels (for analysis)
    print("\nGenerating signals and labels...")
    signal_gen = PrimarySignalGenerator()
    tradeable_data = {t: data[t] for t in universe.tradeable if t in data}
    candidates = signal_gen.generate_candidates(tradeable_data, show_progress=False)

    labeled_df = create_metalabels(
        data,
        candidates,
        holding_period=model_info.get("holding_period", 5),
        min_return=model_info.get("min_return", 0.003),
    )
    print(f"  Generated {len(labeled_df):,} labeled signals")

    # Filter to training period only (before embargo)
    training_end = model_info.get("training_end_date")
    if training_end:
        training_end_ts = pd.Timestamp(training_end)
        train_labeled_df = labeled_df[labeled_df["date"] <= training_end_ts].copy()
        print(f"  Training period signals: {len(train_labeled_df):,}")
    else:
        train_labeled_df = labeled_df

    # Analyze primary signals
    primary_analysis = analyze_primary_signals(train_labeled_df)

    # Create features
    print("\nCreating features for ML analysis...")
    feature_eng = MetaFeatureEngineering(context_tickers=universe.context)
    feature_matrix = feature_eng.create_feature_matrix(
        train_labeled_df, data, show_progress=False
    )
    feature_matrix = feature_matrix.fillna(0)

    from futures.ml.features import FeatureSet
    feature_set = FeatureSet(
        X=feature_matrix,
        y=pd.Series(train_labeled_df["label"].values, index=feature_matrix.index),
        feature_names=list(feature_matrix.columns),
    )

    # Analyze ML model
    ml_analysis = analyze_ml_model(model, feature_set, train_labeled_df)

    # Analyze feature importance
    feature_analysis = analyze_feature_importance(model, feature_set.feature_names)

    # Create plots
    create_diagnostic_plots(output_dir, ml_analysis, primary_analysis)

    # Summary and recommendations
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    primary_win_rate = primary_analysis["win_rate"]
    ml_precision = ml_analysis["precision"]

    print(f"\nKey Findings:")
    print(f"  Primary signals win rate: {primary_win_rate:.1f}%")
    print(f"  ML model precision: {ml_precision:.1f}%")

    if primary_win_rate < 50:
        print(f"\n  WARNING: Primary signals have <50% win rate.")
        print(f"  The underlying technical signals may not have edge.")
        print(f"  Consider: different indicators, different parameters, or different assets.")

    if ml_precision < primary_win_rate + 5:
        print(f"\n  WARNING: ML model adds little value over raw signals.")
        print(f"  Consider: more features, different model, or more training data.")

    if ml_analysis["mean_confidence"] < 0.52:
        print(f"\n  NOTE: Mean confidence is low ({ml_analysis['mean_confidence']:.3f}).")
        print(f"  Model is uncertain - may need better features or more data.")

    print(f"\nDiagnostic plots saved to: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
