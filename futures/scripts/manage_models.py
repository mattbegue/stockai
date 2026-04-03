"""
Model Management CLI

List, compare, and switch between trained models.

Usage:
    python -m futures.scripts.manage_models list
    python -m futures.scripts.manage_models set metalabeling_large_20260129_v1
    python -m futures.scripts.manage_models info metalabeling_large_20260129_v1
    python -m futures.scripts.manage_models compare model1 model2
"""

import argparse
import sys

from futures.models.registry import ModelRegistry


def cmd_list(args):
    """List all available models."""
    registry = ModelRegistry()
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)
    registry.list_models(verbose=True)


def cmd_set(args):
    """Set the active model."""
    registry = ModelRegistry()
    success = registry.set_active(args.model_name)
    if success:
        print(f"\nNow using: {args.model_name}")
        print("All scripts (paper_trade, daily_signals, backtest) will use this model.")


def cmd_info(args):
    """Show detailed info about a model."""
    registry = ModelRegistry()

    try:
        model, info = registry.load(args.model_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 70)
    print(f"MODEL: {args.model_name}")
    print("=" * 70)

    print(f"\n{'Property':<25} {'Value'}")
    print("-" * 50)
    print(f"{'Universe Size':<25} {info.get('universe_size', 'unknown')}")
    print(f"{'Training Date':<25} {info.get('training_date', 'unknown')}")
    print(f"{'Training Samples':<25} {info.get('n_samples', 'unknown')}")
    print(f"{'Number of Features':<25} {len(info.get('feature_names', []))}")
    print(f"{'Holding Period':<25} {info.get('holding_period', 'unknown')} days")
    print(f"{'Min Return Threshold':<25} {info.get('min_return', 'unknown')}")

    # Universe details
    tradeable = info.get('universe_tradeable', [])
    context = info.get('universe_context', [])
    print(f"{'Tradeable Tickers':<25} {len(tradeable)}")
    print(f"{'Context ETFs':<25} {len(context)}")

    # Feature names
    feature_names = info.get('feature_names', [])
    if feature_names:
        print(f"\nFeatures ({len(feature_names)}):")
        for i, name in enumerate(feature_names[:15], 1):
            print(f"  {i:2d}. {name}")
        if len(feature_names) > 15:
            print(f"  ... and {len(feature_names) - 15} more")


def cmd_compare(args):
    """Compare two models side by side."""
    registry = ModelRegistry()

    try:
        _, info1 = registry.load(args.model1)
        _, info2 = registry.load(args.model2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(f"\n{'Property':<25} {args.model1:<25} {args.model2:<25}")
    print("-" * 75)

    props = [
        ("Universe Size", "universe_size"),
        ("Training Date", "training_date"),
        ("Training Samples", "n_samples"),
        ("Holding Period", "holding_period"),
        ("Min Return", "min_return"),
    ]

    for label, key in props:
        v1 = info1.get(key, "N/A")
        v2 = info2.get(key, "N/A")
        print(f"{label:<25} {str(v1):<25} {str(v2):<25}")

    # Feature comparison
    f1 = set(info1.get('feature_names', []))
    f2 = set(info2.get('feature_names', []))

    print(f"\n{'Features':<25} {len(f1):<25} {len(f2):<25}")

    common = f1 & f2
    only1 = f1 - f2
    only2 = f2 - f1

    print(f"{'  Common features':<25} {len(common)}")

    if only1:
        print(f"{'  Only in ' + args.model1:<25} {len(only1)}")
    if only2:
        print(f"{'  Only in ' + args.model2:<25} {len(only2)}")


def cmd_active(args):
    """Show the currently active model."""
    registry = ModelRegistry()
    path = registry.get_active_model_path()

    if path:
        print(f"\nActive model: {path.stem}")
        print(f"Path: {path}")
    else:
        print("\nNo active model set.")


def main():
    parser = argparse.ArgumentParser(
        description="Model Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all available models")
    list_parser.set_defaults(func=cmd_list)

    # Set command
    set_parser = subparsers.add_parser("set", help="Set the active model")
    set_parser.add_argument("model_name", help="Name of the model to activate")
    set_parser.set_defaults(func=cmd_set)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed model info")
    info_parser.add_argument("model_name", help="Name of the model")
    info_parser.set_defaults(func=cmd_info)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("model1", help="First model name")
    compare_parser.add_argument("model2", help="Second model name")
    compare_parser.set_defaults(func=cmd_compare)

    # Active command
    active_parser = subparsers.add_parser("active", help="Show active model")
    active_parser.set_defaults(func=cmd_active)

    args = parser.parse_args()

    if args.command is None:
        # Default to list
        cmd_list(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
