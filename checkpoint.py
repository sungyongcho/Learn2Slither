import os
import sys
from typing import Any, Dict, Tuple

import torch

from model import LinearQNet


def save(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer | None = None,
    **extra: Any,
) -> None:
    """Save *model* (+ optimizer + extras) exactly to *path*."""
    path = os.fspath(path)  # accept Path objects
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict() if optim else None,
            **extra,
        },
        path,
    )
    print(f"[INFO] Saved checkpoint → {path}")


def load(
    path: str | os.PathLike | None,
    input_size: int,
    hidden1_size: int,
    hidden2_size: int,
    output_size: int,
    optim=None,
    step_by_step=False,
) -> Tuple[LinearQNet, Dict[str, Any]]:
    model = LinearQNet(
        input_size,
        hidden1_size,
        hidden2_size,
        output_size,
        step_by_step,
    )
    extras: Dict[str, Any] = {}

    if path is None:
        return model, extras

    path = os.fspath(path)

    if not os.path.isfile(path):
        print(
            f"[ERROR] Checkpoint '{path}' not found. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optim and ckpt["optim"]:
        optim.load_state_dict(ckpt["optim"])
    extras = {k: v for k, v in ckpt.items() if k not in {"model", "optim"}}
    print(f"[INFO] loaded ← {path}")
    return model, extras
