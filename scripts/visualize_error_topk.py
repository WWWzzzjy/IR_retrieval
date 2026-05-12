"""Plot spectra for retrieval error top-k JSON files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.json_io import load_json_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, required=True, help="Path to errors_top5.json or errors_top10.json")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/error_plots"))
    parser.add_argument("--data_dir", type=Path, default=None, help="Optional data/raw directory for path remapping")
    parser.add_argument("--num_cases", type=int, default=20, help="Number of error cases to plot")
    parser.add_argument("--start", type=int, default=0, help="Start offset inside the errors array")
    parser.add_argument("--case_indices", type=int, nargs="*", default=None, help="Specific error-array indices to plot")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--fig_width", type=float, default=12.0)
    parser.add_argument("--row_height", type=float, default=2.0)
    parser.add_argument("--invert_x", action="store_true", help="Invert wavenumber axis for IR-style plots")
    return parser.parse_args()


def load_error_payload(path: Path) -> dict[str, Any]:
    """Load one error top-k JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("errors"), list):
        raise ValueError(f"Expected an error JSON object with an errors list: {path}")
    return payload


def candidate_paths(raw_path: str, data_dir: Path | None) -> list[Path]:
    """Return possible local filesystem paths for one spectrum path string."""

    path = Path(raw_path)
    paths = [path]
    if not path.is_absolute():
        paths.append(ROOT / path)
    if data_dir is not None:
        paths.append(data_dir / path)
        if len(path.parts) >= 2:
            paths.append(data_dir / Path(*path.parts[-2:]))
        if "raw" in path.parts:
            raw_index = path.parts.index("raw")
            if raw_index + 1 < len(path.parts):
                paths.append(data_dir / Path(*path.parts[raw_index + 1 :]))
    unique: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        key = str(item)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def resolve_spectrum_path(raw_path: str, data_dir: Path | None) -> Path:
    """Resolve a JSON path from an error file to an existing local path."""

    for path in candidate_paths(raw_path, data_dir):
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in candidate_paths(raw_path, data_dir))
    raise FileNotFoundError(f"Could not resolve spectrum path {raw_path!r}. Tried: {tried}")


def load_spectrum(raw_path: str, data_dir: Path | None) -> tuple[list[float], list[float], Path]:
    """Load x/y spectrum arrays from one JSON file."""

    path = resolve_spectrum_path(raw_path, data_dir)
    payload = load_json_file(path)
    spectrum = payload.get("spectrum")
    if not isinstance(spectrum, dict):
        raise ValueError(f"Missing spectrum object in {path}")
    y_values = spectrum.get("y")
    if not isinstance(y_values, list):
        raise ValueError(f"Missing spectrum.y array in {path}")
    x_values = spectrum.get("x")
    if not isinstance(x_values, list) or len(x_values) != len(y_values):
        x_values = list(range(len(y_values)))
    return [float(item) for item in x_values], [float(item) for item in y_values], path


def sanitize_filename(value: str) -> str:
    """Return a filesystem-friendly filename fragment."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned[:80] or "case"


def choose_errors(errors: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    """Select errors to plot."""

    if args.case_indices:
        selected: list[tuple[int, dict[str, Any]]] = []
        for index in args.case_indices:
            if index < 0 or index >= len(errors):
                raise IndexError(f"case index must be in [0, {len(errors) - 1}], got {index}")
            selected.append((index, errors[index]))
        return selected
    stop = len(errors) if args.num_cases <= 0 else min(len(errors), args.start + args.num_cases)
    return [(index, errors[index]) for index in range(args.start, stop)]


def title_for_query(query: dict[str, Any]) -> str:
    """Build the query subplot title."""

    return (
        f"QUERY / TRUTH | id={query.get('id', '')} | source={query.get('source', '')} | "
        f"true_rank={query.get('true_rank', '')} | positive_cosine={float(query.get('positive_cosine', 0.0)):.4f} | "
        f"top1-positive={float(query.get('top1_minus_positive', 0.0)):.4f}"
    )


def title_for_candidate(candidate: dict[str, Any]) -> str:
    """Build one candidate subplot title."""

    truth = " | TRUTH" if candidate.get("is_target") else ""
    same_group = " | same_group" if candidate.get("is_same_group") and not candidate.get("is_target") else ""
    return (
        f"TOP {candidate.get('rank')} | cosine={float(candidate.get('cosine', 0.0)):.4f} | "
        f"id={candidate.get('id', '')} | source={candidate.get('source', '')}{truth}{same_group}"
    )


def plot_error_case(
    error_index: int,
    error: dict[str, Any],
    top_k: int,
    args: argparse.Namespace,
) -> Path:
    """Plot one error case and return the output path."""

    query = error["query"]
    candidates = error.get("topk", [])[:top_k]
    rows = 1 + len(candidates)
    fig, axes = plt.subplots(rows, 1, figsize=(args.fig_width, args.row_height * rows), squeeze=False, sharex=False)

    x_values, y_values, resolved_path = load_spectrum(str(query["path"]), args.data_dir)
    query_ax = axes[0, 0]
    query_ax.plot(x_values, y_values, color="black", linewidth=1.2)
    query_ax.set_title(title_for_query(query), loc="left", fontsize=10)
    query_ax.set_ylabel("Abs.")
    query_ax.grid(alpha=0.25)
    query_ax.text(
        0.99,
        0.82,
        "truth raw",
        transform=query_ax.transAxes,
        ha="right",
        va="center",
        fontsize=9,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#111111", "edgecolor": "none", "alpha": 0.85},
    )
    query_ax.text(0.01, 0.08, str(resolved_path), transform=query_ax.transAxes, ha="left", fontsize=7, alpha=0.65)

    for row, candidate in enumerate(candidates, start=1):
        ax = axes[row, 0]
        x_values, y_values, resolved_path = load_spectrum(str(candidate["path"]), args.data_dir)
        is_truth = bool(candidate.get("is_target"))
        color = "#d62728" if is_truth else "#1f77b4"
        linewidth = 1.35 if is_truth else 1.0
        ax.plot(x_values, y_values, color=color, linewidth=linewidth)
        ax.set_title(title_for_candidate(candidate), loc="left", fontsize=10, color=color if is_truth else "black")
        ax.set_ylabel("Abs.")
        ax.grid(alpha=0.25)
        if is_truth:
            ax.text(
                0.99,
                0.82,
                "TRUTH",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=9,
                color="white",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#d62728", "edgecolor": "none", "alpha": 0.9},
            )
        ax.text(0.01, 0.08, str(resolved_path), transform=ax.transAxes, ha="left", fontsize=7, alpha=0.65)

    axes[-1, 0].set_xlabel("Wavenumber (cm^-1)")
    if args.invert_x:
        for ax in axes[:, 0]:
            ax.invert_xaxis()

    fig.suptitle(f"Error case #{error_index} | top{top_k}", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    query_id = sanitize_filename(str(query.get("id", query.get("index", error_index))))
    output_path = args.output_dir / f"error_{error_index:04d}_query_{query_id}_top{top_k}.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate spectrum plots for selected error cases."""

    args = parse_args()
    payload = load_error_payload(args.errors)
    top_k = int(payload.get("top_k") or len(payload["errors"][0].get("topk", [])) if payload["errors"] else 0)
    if top_k <= 0:
        raise ValueError(f"No top-k candidates found in {args.errors}")

    selected = choose_errors(payload["errors"], args)
    outputs = [plot_error_case(index, error, top_k, args) for index, error in selected]
    print(f"Wrote {len(outputs)} error plots to {args.output_dir}")
    for output in outputs[:10]:
        print(output)
    if len(outputs) > 10:
        print(f"... {len(outputs) - 10} more")


if __name__ == "__main__":
    main()
