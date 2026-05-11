"""Visualize a saved embedding index with t-SNE or optional UMAP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/embeddings_tsne.png"))
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def project_embeddings(embeddings: np.ndarray, method: str, seed: int) -> np.ndarray:
    """Project embeddings into two dimensions."""

    if method == "tsne":
        perplexity = min(30.0, max(5.0, (embeddings.shape[0] - 1) / 3.0))
        return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed).fit_transform(
            embeddings
        )
    try:
        import umap
    except ImportError as exc:
        raise ImportError("Install umap-learn to use --method umap") from exc
    return umap.UMAP(n_components=2, random_state=seed).fit_transform(embeddings)


def main() -> None:
    """Create an embedding scatter plot."""

    args = parse_args()
    payload = np.load(args.index, allow_pickle=True)
    embeddings = payload["embeddings"]
    group_ids = payload["group_ids"] if "group_ids" in payload else np.asarray([""] * embeddings.shape[0])

    if embeddings.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(embeddings.shape[0], size=args.max_points, replace=False)
        embeddings = embeddings[keep]
        group_ids = group_ids[keep]

    coords = project_embeddings(embeddings, args.method, args.seed)
    unique_groups, labels = np.unique(group_ids, return_inverse=True)
    color_values = labels if unique_groups.shape[0] <= 50 else np.zeros_like(labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=color_values, s=6, cmap="tab20", alpha=0.75)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Wrote visualization to {args.output}")


if __name__ == "__main__":
    main()

