"""
Backtest Registry

Ties backtest results to git commits so we can track how strategy changes
affect performance over time.

Usage:
    # List all recorded runs
    python -m futures.scripts.backtest_registry

    # Show comparison table (all runs)
    python -m futures.scripts.backtest_registry --compare

    # Show details for a specific run
    python -m futures.scripts.backtest_registry --show bt_001

    # Delete a run
    python -m futures.scripts.backtest_registry --delete bt_001
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

REGISTRY_FILE = Path("backtest_registry.json")

# Metrics shown in the comparison table, in order
COMPARISON_METRICS = [
    ("sharpe_ratio",           "Sharpe",       "{:+.3f}"),
    ("annualized_return_pct",  "Ann. Ret %",   "{:+.1f}%"),
    ("max_drawdown_pct",       "Max DD %",     "{:.1f}%"),
    ("win_rate_pct",           "Win Rate",     "{:.1f}%"),
    ("total_trades",           "Trades",       "{:,}"),
    ("spy_total_return_pct",   "SPY Ret %",    "{:+.1f}%"),
    ("total_return_pct",       "Total Ret %",  "{:+.1f}%"),
]


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------

def _load() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"runs": []}


def _save(registry: dict):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def _git_info() -> tuple[str, str]:
    """Return (short_commit_hash, branch_name). Falls back gracefully."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit, branch
    except Exception:
        return "unknown", "unknown"


def _next_id(registry: dict) -> str:
    n = len(registry["runs"]) + 1
    return f"bt_{n:03d}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record(
    metrics: dict,
    params: dict,
    comment: str = "",
    tags: Optional[list[str]] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    Record a backtest run to the registry.

    Args:
        metrics:  Performance metrics dict (sharpe_ratio, annualized_return_pct, etc.)
        params:   Backtest configuration (universe, dates, thresholds, …)
        comment:  Human-readable note explaining what changed / what was tested
        tags:     Optional labels (e.g. ["baseline", "phase2", "regime"])
        run_id:   Override auto-assigned id (useful for scripted runs)

    Returns:
        The assigned run id.
    """
    registry = _load()
    commit, branch = _git_info()

    entry = {
        "id": run_id or _next_id(registry),
        "timestamp": datetime.now().isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "comment": comment,
        "tags": tags or [],
        "params": params,
        "metrics": metrics,
    }

    registry["runs"].append(entry)
    _save(registry)
    return entry["id"]


def list_runs(tags_filter: Optional[list[str]] = None) -> list[dict]:
    """Return all runs, optionally filtered by tags."""
    registry = _load()
    runs = registry["runs"]
    if tags_filter:
        runs = [r for r in runs if any(t in r.get("tags", []) for t in tags_filter)]
    return runs


def get_run(run_id: str) -> Optional[dict]:
    """Fetch a single run by id."""
    for run in list_runs():
        if run["id"] == run_id:
            return run
    return None


def delete_run(run_id: str) -> bool:
    """Remove a run from the registry. Returns True if found and deleted."""
    registry = _load()
    original = len(registry["runs"])
    registry["runs"] = [r for r in registry["runs"] if r["id"] != run_id]
    if len(registry["runs"]) < original:
        _save(registry)
        return True
    return False


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _fmt(value, fmt_str: str) -> str:
    try:
        if value is None:
            return "—"
        return fmt_str.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def print_comparison_table(runs: Optional[list[dict]] = None):
    """Print a side-by-side performance comparison of all (or provided) runs."""
    if runs is None:
        runs = list_runs()

    if not runs:
        print("No backtest runs recorded yet.")
        return

    # Header
    col_widths = {
        "id": 8, "commit": 8, "branch": 18, "comment": 30,
    }
    metric_width = 11

    header_parts = [
        f"{'ID':<{col_widths['id']}}",
        f"{'Commit':<{col_widths['commit']}}",
        f"{'Branch':<{col_widths['branch']}}",
    ]
    for _, label, _ in COMPARISON_METRICS:
        header_parts.append(f"{label:>{metric_width}}")
    header_parts.append(f"  {'Comment'}")

    print("\n" + "=" * (sum(col_widths.values()) + len(COMPARISON_METRICS) * (metric_width + 1) + 40))
    print("BACKTEST REGISTRY")
    print("=" * (sum(col_widths.values()) + len(COMPARISON_METRICS) * (metric_width + 1) + 40))
    print("  ".join(header_parts))
    print("-" * (sum(col_widths.values()) + len(COMPARISON_METRICS) * (metric_width + 1) + 40))

    for run in runs:
        m = run.get("metrics", {})
        row_parts = [
            f"{run['id']:<{col_widths['id']}}",
            f"{run['git_commit']:<{col_widths['commit']}}",
            f"{run['git_branch'][:col_widths['branch']]:<{col_widths['branch']}}",
        ]
        for key, _, fmt in COMPARISON_METRICS:
            row_parts.append(f"{_fmt(m.get(key), fmt):>{metric_width}}")
        comment = run.get("comment", "")[:40]
        row_parts.append(f"  {comment}")
        print("  ".join(row_parts))

    print()


def print_run_detail(run: dict):
    """Print full details for a single run."""
    print(f"\n{'=' * 60}")
    print(f"Run: {run['id']}")
    print(f"{'=' * 60}")
    print(f"  Timestamp:  {run['timestamp']}")
    print(f"  Git Commit: {run['git_commit']} ({run['git_branch']})")
    print(f"  Comment:    {run.get('comment', '')}")
    print(f"  Tags:       {', '.join(run.get('tags', [])) or 'none'}")

    print(f"\n  Params:")
    for k, v in run.get("params", {}).items():
        print(f"    {k}: {v}")

    print(f"\n  Metrics:")
    m = run.get("metrics", {})
    for key, label, fmt in COMPARISON_METRICS:
        if key in m:
            print(f"    {label:<22} {_fmt(m[key], fmt)}")

    # Any extra metrics not in COMPARISON_METRICS
    shown = {k for k, _, _ in COMPARISON_METRICS}
    extras = {k: v for k, v in m.items() if k not in shown}
    if extras:
        print(f"\n  Additional metrics:")
        for k, v in extras.items():
            print(f"    {k}: {v}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest Registry")
    parser.add_argument("--compare", action="store_true", help="Show comparison table")
    parser.add_argument("--show", metavar="RUN_ID", help="Show details for a run")
    parser.add_argument("--delete", metavar="RUN_ID", help="Delete a run")
    parser.add_argument("--tag", metavar="TAG", help="Filter by tag")
    args = parser.parse_args()

    if args.delete:
        if delete_run(args.delete):
            print(f"Deleted run: {args.delete}")
        else:
            print(f"Run not found: {args.delete}")

    elif args.show:
        run = get_run(args.show)
        if run:
            print_run_detail(run)
        else:
            print(f"Run not found: {args.show}")

    else:
        # Default: show comparison table
        tag_filter = [args.tag] if args.tag else None
        runs = list_runs(tags_filter=tag_filter)
        print_comparison_table(runs)


if __name__ == "__main__":
    main()
