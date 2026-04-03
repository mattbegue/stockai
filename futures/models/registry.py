"""
Model Registry

Manages multiple trained models with clear naming and version tracking.

Model naming convention:
    metalabeling_{universe}_{date}_{version}.pkl

Example:
    metalabeling_large_20260129_v1.pkl
    metalabeling_small_20260130_v1.pkl

Usage:
    from futures.models.registry import ModelRegistry

    registry = ModelRegistry()

    # List available models
    registry.list_models()

    # Load the active model
    model, info = registry.load_active()

    # Load a specific model
    model, info = registry.load("metalabeling_large_20260129_v1")

    # Set active model
    registry.set_active("metalabeling_large_20260129_v1")
"""

import json
import pickle
from datetime import date
from pathlib import Path
from typing import Optional

MODELS_DIR = Path("models")
ACTIVE_MODEL_FILE = MODELS_DIR / "active_model.json"


class ModelRegistry:
    """Manages multiple trained models."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(exist_ok=True)

    def _get_model_files(self) -> list[Path]:
        """Get all model files in the directory, sorted by date/version in filename."""
        files = list(self.models_dir.glob("metalabeling_*.pkl"))

        def sort_key(p: Path) -> tuple:
            # Parse: metalabeling_{universe}_{YYYYMMDD}_v{version}.pkl
            # Returns (date_str, version) for sorting
            name = p.stem
            parts = name.split("_")
            try:
                # Find the date part (8 digits)
                date_part = next((p for p in parts if p.isdigit() and len(p) == 8), "00000000")
                # Find version part (after 'v')
                version_part = next((p for p in parts if p.startswith("v") and p[1:].isdigit()), "v0")
                version = int(version_part[1:])
                return (date_part, version)
            except (StopIteration, ValueError):
                return ("00000000", 0)

        return sorted(files, key=sort_key)

    def list_models(self, verbose: bool = True) -> list[dict]:
        """List all available models."""
        models = []

        for model_path in self._get_model_files():
            try:
                with open(model_path, "rb") as f:
                    info = pickle.load(f)

                model_info = {
                    "name": model_path.stem,
                    "path": str(model_path),
                    "universe": info.get("universe_size", "unknown"),
                    "training_date": info.get("training_date", "unknown"),
                    "training_end_date": info.get("training_end_date"),
                    "embargo_end_date": info.get("embargo_end_date"),
                    "n_samples": info.get("n_samples", "unknown"),
                    "n_features": len(info.get("feature_names", [])),
                }
                models.append(model_info)
            except Exception as e:
                models.append({
                    "name": model_path.stem,
                    "path": str(model_path),
                    "error": str(e),
                })

        # Check which is active
        active_name = self._get_active_model_name()

        if verbose:
            print(f"\n{'Model Name':<40} {'Universe':<8} {'Train End':<12} {'Embargo':<12} {'Samples':>8}")
            print("-" * 85)

            for m in models:
                if "error" in m:
                    print(f"{m['name']:<40} ERROR: {m['error']}")
                else:
                    active_marker = " *" if m["name"] == active_name else ""
                    train_end = m.get('training_end_date', 'legacy')[:10] if m.get('training_end_date') else 'legacy'
                    embargo = m.get('embargo_end_date', '-')[:10] if m.get('embargo_end_date') else '-'
                    print(f"{m['name']:<40} {m['universe']:<8} {train_end:<12} {embargo:<12} {m['n_samples']:>8}{active_marker}")

            if active_name:
                print(f"\n* = active model")
            print(f"\nNote: 'legacy' models lack embargo metadata - backtest validation may be limited")

        return models

    def _get_active_model_name(self) -> Optional[str]:
        """Get the name of the currently active model."""
        if not ACTIVE_MODEL_FILE.exists():
            return None

        with open(ACTIVE_MODEL_FILE, "r") as f:
            data = json.load(f)
        return data.get("active_model")

    def get_active_model_path(self, verbose: bool = False) -> Optional[Path]:
        """
        Get the path to the active model.

        Fallback order:
        1. Explicitly set active model (from active_model.json)
        2. Most recently modified model in directory
        3. Legacy model (metalabeling_model.pkl)
        """
        active_name = self._get_active_model_name()

        if active_name:
            path = self.models_dir / f"{active_name}.pkl"
            if path.exists():
                return path
            elif verbose:
                print(f"Warning: Active model '{active_name}' not found, falling back...")

        # Fallback: use most recent model (sorted by mtime, last is newest)
        model_files = self._get_model_files()
        if model_files:
            most_recent = model_files[-1]
            if verbose:
                print(f"Using most recent model: {most_recent.stem}")
            return most_recent

        # Fallback: look for legacy model
        legacy_path = self.models_dir / "metalabeling_model.pkl"
        if legacy_path.exists():
            if verbose:
                print(f"Using legacy model: {legacy_path}")
            return legacy_path

        return None

    def set_active(self, model_name: str) -> bool:
        """Set the active model by name."""
        # Remove .pkl extension if provided
        model_name = model_name.replace(".pkl", "")

        model_path = self.models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            print(f"Error: Model '{model_name}' not found.")
            print(f"Available models:")
            self.list_models()
            return False

        with open(ACTIVE_MODEL_FILE, "w") as f:
            json.dump({
                "active_model": model_name,
                "set_date": date.today().isoformat(),
            }, f, indent=2)

        print(f"Active model set to: {model_name}")
        return True

    def load_active(self, verbose: bool = True) -> tuple:
        """
        Load the currently active model.

        If no model is explicitly set as active, loads the most recent model.
        """
        model_path = self.get_active_model_path(verbose=verbose)

        if model_path is None:
            raise FileNotFoundError("No model found. Train a model first.")

        return self.load(model_path.stem)

    def load(self, model_name: str) -> tuple:
        """Load a specific model by name."""
        model_name = model_name.replace(".pkl", "")
        model_path = self.models_dir / f"{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found at {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        return data["model"], data

    def generate_model_name(self, universe_size: str, version: int = 1) -> str:
        """Generate a model name based on universe and date."""
        today = date.today().strftime("%Y%m%d")
        return f"metalabeling_{universe_size}_{today}_v{version}"

    def get_next_version(self, universe_size: str) -> int:
        """Get the next version number for today's models."""
        today = date.today().strftime("%Y%m%d")
        prefix = f"metalabeling_{universe_size}_{today}_v"

        existing = [p for p in self._get_model_files() if prefix in p.stem]

        if not existing:
            return 1

        versions = []
        for p in existing:
            try:
                v = int(p.stem.split("_v")[-1])
                versions.append(v)
            except ValueError:
                pass

        return max(versions) + 1 if versions else 1

    def save_model(self, model, info: dict, universe_size: str, set_active: bool = True) -> Path:
        """Save a model with proper naming."""
        version = self.get_next_version(universe_size)
        model_name = self.generate_model_name(universe_size, version)
        model_path = self.models_dir / f"{model_name}.pkl"

        # Add universe info to model data
        info["universe_size"] = universe_size
        info["model_name"] = model_name

        with open(model_path, "wb") as f:
            pickle.dump({"model": model, **info}, f)

        print(f"Model saved: {model_path}")

        if set_active:
            self.set_active(model_name)

        return model_path


# Convenience functions
def load_active_model():
    """Load the currently active model."""
    registry = ModelRegistry()
    return registry.load_active()


def list_models():
    """List all available models."""
    registry = ModelRegistry()
    return registry.list_models()


def set_active_model(model_name: str):
    """Set the active model."""
    registry = ModelRegistry()
    return registry.set_active(model_name)
