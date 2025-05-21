from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --------------------------------------------------------------------------- #
#                               Neural network                                #
# --------------------------------------------------------------------------- #
class LinearQNet(nn.Module):
    """
    A simple 3‑layer fully‑connected network used as the Q‑function
    approximator.

    Architecture: input →  hidden  →  hidden//2  →  output
                  ReLU       —             —
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)

    # Forward pass ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    # --- convenience wrappers ------------------------------------------------------
    def save(
        self,
        file_name: str | None = None,
        optimizer: optim.Optimizer | None = None,
        **extra: Any,
    ) -> None:
        """
        Persist the model (and optionally the optimiser + extra metadata)
        into ./model/ by default.

        Example
        -------
        model.save(
            "checkpoint.pth",
            optimizer=my_optim,
            epsilon=current_epsilon,
            games_played=num_games,
        )
        """
        if file_name is None:
            file_name = "model.pth"

        # ensure directory exists
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        path = os.path.join(model_folder_path, file_name)

        save_checkpoint(path, self, optimizer, **extra)

    @staticmethod
    def load(
        path: str | None,
        input_size: int,
        hidden_size: int,
        output_size: int,
        optimizer: optim.Optimizer | None = None,
    ) -> "LinearQNet":
        """
        Create a network and load weights (and optionally an optimiser).

        Only the model part is GUARANTEED to be restored here – the caller
        may inspect the returned *extras* from :func:`load_checkpoint` if it
        needs epsilon, games played, etc.
        """
        model, _ = load_checkpoint(
            path, input_size, hidden_size, output_size, optimizer
        )
        return model


# --------------------------------------------------------------------------- #
#                                 Trainer                                     #
# --------------------------------------------------------------------------- #
class QTrainer:
    """One‑step TD(0) trainer for the Deep Q‑Network."""

    def __init__(self, model: LinearQNet, lr: float, gamma: float) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # --------------------------------------------------------------------- #
    #                         One training iteration                        #
    # --------------------------------------------------------------------- #
    def train_step(self, state, action, reward, next_state, done):  # noqa: C901
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        # Handle single sample (shape = (x,))
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            done = (done,)

        # 1) current Q‑values ------------------------------------------------------
        pred = self.model(state)
        target = pred.clone()

        # 2) target computation ----------------------------------------------------
        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * torch.max(
                    self.model(next_state[i].detach())
                )
            act_idx = torch.argmax(action[i]).item()
            target[i][act_idx] = q_new

        # 3) gradient descent step -------------------------------------------------
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    # Convenience wrappers ---------------------------------------------------------
    def save(self, path: str, **extra: Any) -> None:
        """Save model + optimiser + extra metadata."""
        self.model.save(path, optimizer=self.optimizer, **extra)

    def load(
        self, path: str, input_size: int, hidden_size: int, output_size: int
    ) -> Dict[str, Any]:
        """Load checkpoint into *self.model* and *self.optimizer*."""
        _, extra = load_checkpoint(
            path,
            input_size,
            hidden_size,
            output_size,
            optimizer=self.optimizer,
        )
        return extra


# --------------------------------------------------------------------------- #
#                       Generic checkpoint helpers                            #
# --------------------------------------------------------------------------- #


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    **extra: Any,
) -> None:
    """
    Save *model* (and optionally *optimizer*) to *path*.

    Extra keyword arguments are stored verbatim in the checkpoint, which
    allows you to persist epsilon, episode counters, etc.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        **extra,
    }
    torch.save(checkpoint, path)
    print(f"[INFO] Saved checkpoint to {path}")


def load_checkpoint(
    path: str | None,
    input_size: int,
    hidden_size: int,
    output_size: int,
    optimizer: optim.Optimizer | None = None,
) -> Tuple[LinearQNet, Dict[str, Any]]:
    """
    Load checkpoint from *path* and return (model, extra).

    If *optimizer* is supplied and the file contains optimiser state,
    that state is restored as well.

    Returns
    -------
    model : LinearQNet
        The freshly‑instantiated network with weights loaded.
    extra : dict
        Any additional metadata stored in the checkpoint (epsilon, games…).
    """
    model = LinearQNet(input_size, hidden_size, output_size)
    extra: Dict[str, Any] = {}

    if path and isinstance(path, (str, bytes)) and os.path.isfile(path):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

        if optimizer is not None and ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        extra = {
            k: v for k, v in ckpt.items() if k not in {"model_state", "optimizer_state"}
        }

        print(f"[INFO] Loaded checkpoint from {path}")
    else:
        if path:
            print(f"[INFO] No checkpoint found at {path}; starting fresh.")

    return model, extra
