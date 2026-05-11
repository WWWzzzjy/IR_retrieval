"""Build an embedding index from a Lightning checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IRSpectrumDataset, eval_collate
from src.training.lightning_module import IRContrastiveModule
from src.training.utils import get_device
from src.utils.config import apply_named_overrides, apply_overrides, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split_index", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/ir_embeddings.npz"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override dotted config key")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Encode spectra and write a compressed NumPy index."""

    args = parse_args()
    device = get_device(args.device)
    model = IRContrastiveModule.load_from_checkpoint(args.checkpoint, map_location=device).to(device)
    config = load_config(args.config) if args.config else model.config
    config = apply_named_overrides(config, vars(args))
    config = apply_overrides(config, args.overrides)
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})

    data_dir = Path(str(data_cfg.get("data_dir", "data/raw")))
    split_index = Path(str(data_cfg.get("split_index", "data/splits.json")))
    use_split_index = args.split is not None and split_index.exists()
    dataset = IRSpectrumDataset(
        data_dir,
        split_index if use_split_index else None,
        args.split if use_split_index else None,
        spectrum_length=int(data_cfg.get("spectrum_length", config.get("model", {}).get("spectrum_length", 460))),
        cache=bool(data_cfg.get("cache", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(eval_cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        collate_fn=eval_collate,
    )

    model.eval()
    embeddings: list[torch.Tensor] = []
    ids: list[str] = []
    group_ids: list[str] = []
    paths: list[str] = []
    for batch in loader:
        spectra = batch["spectrum"].to(device)
        wavenumbers = batch.get("x")
        embeddings.append(model(spectra, wavenumbers=wavenumbers.to(device) if wavenumbers is not None else None).cpu())
        ids.extend(str(item) for item in batch["ids"])
        group_ids.extend(str(item) for item in batch["group_ids"])
        paths.extend(str(item) for item in batch["paths"])

    embedding_tensor = torch.cat(embeddings, dim=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embedding_tensor.numpy(),
        ids=np.asarray(ids),
        group_ids=np.asarray(group_ids),
        paths=np.asarray(paths),
    )
    print(f"Wrote {embedding_tensor.shape[0]} embeddings to {args.output}")


if __name__ == "__main__":
    main()

