from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch

from checkpoint import load, save
from constants import BATCH_SIZE, LR, MAX_MEMORY, Direction, Pos, RLConfig
from environment import Environment
from model import QTrainer


class Agent:
    def __init__(
        self: Agent,
        config: RLConfig,
        load_path: str | None = None,
        step_by_step: bool = False,
    ) -> None:
        self.config: RLConfig = config
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() when full
        self.num_games: int = 0
        self.epsilon: float = self.config.initial_epsilon  # Current epsilon
        self.step_by_step: bool = step_by_step
        self.model, extras = load(
            load_path,
            config.input_size,
            config.hidden1_size,
            config.hidden2_size,
            config.output_size,
            optim=None,
            step_by_step=step_by_step,
        )
        self.__dict__.update(extras)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.config.gamma)

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

        if self.step_by_step:
            # Format the state list:
            # - Floats that are whole numbers (e.g., 1.0) are formatted as integers (e.g., "1")
            # - Other floats are formatted to 6 decimal places
            formatted_state = [
                str(int(x)) if x == int(x) else f"{x:.6f}" for x in state
            ]
            print(
                f"[Agent] Got state of the current game [{', '.join(formatted_state)}]"
            )
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
        action_names = ["straight", "right_turn", "left_turn"]

        # Calculate current epsilon based on exponential decay
        # Epsilon starts at self.initial_epsilon and decays towards self.min_epsilon
        calculated_epsilon_val = self.config.initial_epsilon * (
            self.config.epsilon_decay**self.num_games
        )
        self.epsilon = max(
            self.config.min_epsilon,
            calculated_epsilon_val,
        )

        if self.step_by_step:
            print(
                f"[Agent] Epsilon calculation: initial_eps={self.config.initial_epsilon}, "
                f"min_eps={self.config.min_epsilon}, decay={self.config.epsilon_decay}, "
                f"num_games={self.num_games} -> calculated_raw_eps={calculated_epsilon_val:.6f} -> "
                f"current_epsilon={self.epsilon:.6f}"
            )

        move: list[int] = [0, 0, 0]  # [straight, right_turn, left_turn]
        random_roll = random.random()  # random.random() gives float between 0.0 and 1.0

        if self.step_by_step:
            print(
                f"[Agent] Epsilon check: random_roll={random_roll:.6f} vs current_epsilon={self.epsilon:.6f}"
            )

        if random_roll < self.epsilon:
            # Explore: take a random action
            if self.step_by_step:
                print("[Agent] Action choice: EXPLORE (random_roll < epsilon)")

            action_idx = random.randint(0, 2)
            move[action_idx] = 1

            if self.step_by_step:
                chosen_action_name = action_names[action_idx]
                print(
                    f"[Agent] Random action chosen: {chosen_action_name} (index {action_idx}). Move: {move}"
                )
        else:
            if self.step_by_step:
                print("[Agent] Action choice: EXPLOIT (random_roll >= epsilon)")
            # Exploit: take the action with the highest Q-value
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model.forward(
                state_tensor
            )  # self.model(state_tensor) also works
            if self.step_by_step:
                q_values = prediction.squeeze(
                    0
                ).tolist()  # Remove batch dim if added, convert to list
                formatted_q_values = [f"{q:.4f}" for q in q_values]
                q_value_str = ", ".join(
                    [
                        f"{name}: {val}"
                        for name, val in zip(action_names, formatted_q_values)
                    ]
                )
                print(f"[Agent] Model Q-value predictions: [{q_value_str}]")

            move_idx: int = torch.argmax(prediction).item()
            move[move_idx] = 1
            if self.step_by_step:
                chosen_action_name = action_names[move_idx]
                print(
                    f"[Agent] Model chose action: {chosen_action_name} (index {move_idx}). Move: {move}"
                )

        return move

    def save(self, save_path: str = None) -> None:
        save(
            save_path,
            self.model,
            optim=self.trainer.optimizer,
            epsilon=self.epsilon,
            num_games=self.num_games,
            gamma=self.config.gamma,
            epsilon_decay=self.config.epsilon_decay,
            initial_epsilon=self.config.initial_epsilon,
            min_epsilon=self.config.min_epsilon,
        )
