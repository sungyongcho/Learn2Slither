from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch

from constants import BATCH_SIZE, LR, MAX_MEMORY, Direction, Pos
from environment import Environment
from model import LinearQNet, QTrainer


class Agent:
    def __init__(
        self: Agent,
        input_size: int = 20,  # Default based on your original state size
        initial_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay_rate: float = 0.995,
        load_path: str | None = None,
    ) -> None:
        self.num_games: int = 0
        # Epsilon parameters for exploration-exploitation trade-off
        self.initial_epsilon: float = initial_epsilon
        self.min_epsilon: float = min_epsilon
        self.epsilon_decay_rate: float = epsilon_decay_rate
        self.epsilon: float = self.initial_epsilon  # Current epsilon

        self.gamma: float = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() when full

        self.model = LinearQNet.load(load_path, input_size, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self: Agent, env: Environment) -> np.ndarray:
        """
        20-float vector in [F, B, L, R] order relative to the snake’s current heading:
            0-3   wall distance
            4-7   green-apple distance   (nearest one)
            8-11  red-apple  distance
            12-15 body distance
            16-19 collision-on-next-step (0/1)
        """
        # ───── short-hand ──────────────────────────────────────────────────────────
        W, H = env.width, env.height
        head: Pos = env.head
        hx, hy = head.x, head.y
        head_dir: Direction = env.direction
        body_set = {(p.x, p.y) for p in env.snake[1:]}  # exclude head

        # ───── relative deltas [F,B,L,R] ──────────────────────────────────────────
        def rel_deltas(d: Direction):
            if d == Direction.RIGHT:
                return [(1, 0), (-1, 0), (0, -1), (0, 1)]
            if d == Direction.LEFT:
                return [(-1, 0), (1, 0), (0, 1), (0, -1)]
            if d == Direction.UP:
                return [(0, -1), (0, 1), (-1, 0), (1, 0)]
            if d == Direction.DOWN:
                return [(0, 1), (0, -1), (1, 0), (-1, 0)]
            raise ValueError("Unknown direction")

        directions = rel_deltas(head_dir)  # [F, B, L, R]

        # ───── helper metrics (all ∈ [0,1]) ───────────────────────────────────────
        def wall_dist(dx: int, dy: int) -> float:
            if dx > 0:
                return (W - 1 - hx) / (W - 1)
            if dx < 0:
                return hx / (W - 1)
            if dy > 0:
                return (H - 1 - hy) / (H - 1)
            if dy < 0:
                return hy / (H - 1)
            return 0.0

        def item_dist(dx: int, dy: int, item: Optional[Pos]) -> float:
            if item is None:
                return 1.0
            ix, iy = item.x, item.y
            # item must lie on the ray in (dx,dy)
            if (dx and (iy != hy or (ix - hx) * dx <= 0)) or (
                dy and (ix != hx or (iy - hy) * dy <= 0)
            ):
                return 1.0
            manhattan = abs(ix - hx) + abs(iy - hy)
            return manhattan / ((W - 1) + (H - 1))

        def body_dist(dx: int, dy: int) -> float:
            x, y, steps = hx, hy, 0
            while True:
                x += dx
                y += dy
                steps += 1
                if x < 0 or x >= W or y < 0 or y >= H:
                    return 1.0  # no body before wall
                if (x, y) in body_set:
                    max_axis = (W - 1) if dx else (H - 1)
                    return steps / max_axis

        def will_collide(dx: int, dy: int) -> float:
            nx, ny = hx + dx, hy + dy
            return float(nx < 0 or nx >= W or ny < 0 or ny >= H or (nx, ny) in body_set)

        # nearest green apple (if any)
        green_closest = min(
            env.green_apples,
            key=lambda a: abs(a.x - hx) + abs(a.y - hy),
            default=None,
        )

        # ───── assemble state ─────────────────────────────────────────────────────
        state: list[float] = []
        for dx, dy in directions:
            state.append(wall_dist(dx, dy))  # 0-3
        for dx, dy in directions:
            state.append(item_dist(dx, dy, green_closest))  # 4-7
        for dx, dy in directions:
            state.append(item_dist(dx, dy, env.red_apple))  # 8-11
        for dx, dy in directions:
            state.append(body_dist(dx, dy))  # 12-15
        for dx, dy in directions:
            state.append(will_collide(dx, dy))  # 16-19

        return np.asarray(state, dtype=np.float32)

    def remember(
        self: Agent,
        state: np.array,
        action: list[int],
        reward: float,
        next_state: np.array,
        done: bool,
    ) -> None:
        """Stores experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self: Agent) -> None:
        """Trains the Q-network on a batch of experiences from memory."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(
        self: Agent,
        state: np.array,
        action: list[int],
        reward: float,
        next_state: np.array,
        done: bool,
    ) -> None:
        """Trains the Q-network on a single (the most recent) experience."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self: Agent, state: np.array) -> list[int]:
        """
        Determines the next action using an epsilon-greedy strategy.
        Epsilon decays exponentially with the number of games played.
        """
        # Calculate current epsilon based on exponential decay
        # Epsilon starts at self.initial_epsilon and decays towards self.min_epsilon
        self.epsilon = max(
            self.min_epsilon,
            self.initial_epsilon * (self.epsilon_decay_rate**self.num_games),
        )

        final_move: list = [0, 0, 0]  # [straight, right_turn, left_turn]

        if (
            random.random() < self.epsilon
        ):  # random.random() gives float between 0.0 and 1.0
            # Explore: take a random action
            move_idx = random.randint(0, 2)  # 0: straight, 1: right, 2: left
            final_move[move_idx] = 1
        else:
            # Exploit: take the action with the highest Q-value
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model.forward(
                state_tensor
            )  # self.model(state_tensor) also works
            move_idx: int = torch.argmax(prediction).item()
            final_move[move_idx] = 1

        return final_move
