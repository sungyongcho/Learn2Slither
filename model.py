from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearQNet(nn.Module):
    def __init__(
        self: LinearQNet, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self: LinearQNet, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    def save(self: LinearQNet, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    @staticmethod
    def load(
        path: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> LinearQNet:
        """Create a network and load weights from *file_name* if it exists."""
        net = LinearQNet(input_size, hidden_size, output_size)
        if path and isinstance(path, (str, bytes)) and os.path.isfile(path):
            net.load_state_dict(torch.load(path))
            print(f"[INFO] Loaded model weights from {path}")
        else:
            if path:
                print(f"[INFO] No model found at {path}; starting fresh.")
        return net


class QTrainer:
    def __init__(
        self: QTrainer,
        model: LinearQNet,
        lr: float,
        gamma: float,
    ) -> None:
        self.lr: float = lr
        self.gamma: float = gamma
        self.model: LinearQNet = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)

        # 1: predicited Q values with current state
        pred = self.model(state)
        target = pred.clone()

        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(
                    self.model(next_state[i].detach())
                )
            act_idx = torch.argmax(action[i]).item()  # pick the action *for row i*
            target[i][act_idx] = Q_new
        # 2: Q_new = r + y * max(next predicted Q value)
        # -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
