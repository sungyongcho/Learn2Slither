from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --------------------------------------------------------------------------- #
#                               Neural network                                #
# --------------------------------------------------------------------------- #
class LinearQNet(nn.Module):
    """
    A simple 3-layer fully-connected network used as the Q-function
    approximator.

    Architecture: input →  hidden  →  hidden//2  →  output
                  ReLU       —             —
    """

    def __init__(
        self,
        input_size: int,
        hidden1_size: int,
        hidden2_size: int,
        output_size: int,
        step_by_step: bool = False,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden1_size)
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)
        self.linear3 = nn.Linear(hidden2_size, output_size)
        self.step_by_step: bool = step_by_step

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class QTrainer:
    def __init__(self, model: LinearQNet, lr: float, gamma: float) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        state: object,
        action: object,
        reward: object,
        next_state: object,
        done: object,
    ) -> None:
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

        # 1) current Q‑values
        pred = self.model(state)

        target = pred.detach().clone()

        with torch.no_grad():
            next_pred = self.model(next_state)

        # 2) target computation
        for i in range(len(done)):
            q_new = reward[i]
            if not done[i]:
                q_new = reward[i] + self.gamma * torch.max(next_pred[i])
            act_idx = torch.argmax(action[i]).item()
            target[i][act_idx] = q_new

        # 3) gradient descent step
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
