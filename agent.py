from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch

from checkpoint import load, save
from constants import BATCH_SIZE, MAX_MEMORY, Direction, Pos, RLConfig
from environment import Environment
from model import LinearQNet, QTrainer


class Agent:
    ACTION_NAMES = ["straight", "right_turn", "left_turn"]

    def __init__(
        self: Agent,
        config: RLConfig,
        load_path: str | None = None,
        step_by_step: bool = False,
    ) -> None:
        self.config: RLConfig = config
        self.memory = deque(maxlen=MAX_MEMORY)
        self.num_games: int = 0
        self.step_by_step: bool = step_by_step

        self.model, self.trainer = self._build_model_and_trainer(step_by_step)
        self.epsilon: float = self.config.initial_epsilon
        self._load_checkpoint(load_path, step_by_step)

    def _build_model_and_trainer(
        self: Agent, step_by_step: bool
    ) -> tuple[LinearQNet, QTrainer]:
        model = LinearQNet(
            self.config.input_size,
            self.config.hidden1_size,
            self.config.hidden2_size,
            self.config.output_size,
            step_by_step=step_by_step,
        )
        trainer = QTrainer(
            model,
            lr=self.config.lr,
            gamma=self.config.gamma,
        )
        return model, trainer

    def _load_checkpoint(
        self: Agent, load_path: str | None, step_by_step: bool
    ) -> None:
        if not load_path:
            return
        _, extras = load(
            load_path,
            self.config.input_size,
            self.config.hidden1_size,
            self.config.hidden2_size,
            self.config.output_size,
            model=self.model,
            optim=self.trainer.optimizer,
            step_by_step=step_by_step,
        )
        self.__dict__.update(extras)

    @staticmethod
    def _direction_vectors(head_dir: Direction) -> list[tuple[int, int]]:
        mapping = {
            Direction.RIGHT: [(1, 0), (-1, 0), (0, -1), (0, 1)],
            Direction.LEFT: [(-1, 0), (1, 0), (0, 1), (0, -1)],
            Direction.UP: [(0, -1), (0, 1), (-1, 0), (1, 0)],
            Direction.DOWN: [(0, 1), (0, -1), (1, 0), (-1, 0)],
        }
        return mapping[head_dir]

    @staticmethod
    def _wall_distance(
        W: int,
        H: int,
        hx: int,
        hy: int,
        dx: int,
        dy: int,
    ) -> float:
        if dx > 0:
            return (W - 1 - hx) / (W - 1)
        if dx < 0:
            return hx / (W - 1)
        if dy > 0:
            return (H - 1 - hy) / (H - 1)
        if dy < 0:
            return hy / (H - 1)
        return 0.0

    @staticmethod
    def _item_distance(
        W: int,
        H: int,
        hx: int,
        hy: int,
        dx: int,
        dy: int,
        item: Pos | None,
    ) -> float:
        if item is None:
            return 1.0
        ix, iy = item.x, item.y
        if (dx and (iy != hy or (ix - hx) * dx <= 0)) or (
            dy and (ix != hx or (iy - hy) * dy <= 0)
        ):
            return 1.0
        manhattan = abs(ix - hx) + abs(iy - hy)
        return manhattan / ((W - 1) + (H - 1))

    @staticmethod
    def _closest_apple_in_dir(
        hx: int,
        hy: int,
        dx: int,
        dy: int,
        apples: list[Pos],
    ) -> Pos | None:
        best = None
        best_dist = float("inf")
        for apple in apples:
            ix, iy = apple.x, apple.y
            if dx != 0:
                if iy != hy:
                    continue
                if (ix - hx) * dx <= 0:
                    continue
            else:
                if ix != hx:
                    continue
                if (iy - hy) * dy <= 0:
                    continue
            dist = abs(ix - hx) + abs(iy - hy)
            if dist < best_dist:
                best_dist = dist
                best = apple
        return best

    @staticmethod
    def _body_distance(
        W: int,
        H: int,
        hx: int,
        hy: int,
        dx: int,
        dy: int,
        body_set: set[tuple[int, int]],
    ) -> float:
        x, y, steps = hx, hy, 0
        while True:
            x += dx
            y += dy
            steps += 1
            if x < 0 or x >= W or y < 0 or y >= H:
                return 1.0
            if (x, y) in body_set:
                max_axis = (W - 1) if dx else (H - 1)
                return steps / max_axis

    @staticmethod
    def _collision_flag(
        W: int,
        H: int,
        hx: int,
        hy: int,
        dx: int,
        dy: int,
        body_set: set[tuple[int, int]],
    ) -> float:
        nx, ny = hx + dx, hy + dy
        return float(nx < 0 or nx >= W or ny < 0 or ny >= H or (nx, ny) in body_set)

    @staticmethod
    def _one_hot_action(idx: int) -> list[int]:
        move = [0, 0, 0]
        move[idx] = 1
        return move

    @staticmethod
    def _format_state_for_log(state: list[float]) -> str:
        formatted_state = [str(int(x)) if x == int(x) else f"{x:.6f}" for x in state]
        return ", ".join(formatted_state)

    def _log_q_values(self: Agent, prediction: torch.Tensor) -> None:
        q_values = prediction.squeeze(0).tolist()
        formatted_q_values = [f"{q:.4f}" for q in q_values]
        q_value_str = ", ".join(
            f"{name}: {val}" for name, val in zip(self.ACTION_NAMES, formatted_q_values)
        )
        print(f"[Agent] Model Q-value predictions: [{q_value_str}]")

    def _update_epsilon(self: Agent) -> tuple[float, float]:
        calculated_epsilon_val = self.config.initial_epsilon * (
            self.config.epsilon_decay**self.num_games
        )
        self.epsilon = max(self.config.min_epsilon, calculated_epsilon_val)
        return calculated_epsilon_val, self.epsilon

    def _exploit_action(self: Agent, state: np.ndarray) -> list[int]:
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state_tensor)
        move_idx: int = torch.argmax(prediction).item()
        move = self._one_hot_action(move_idx)

        if self.step_by_step:
            self._log_q_values(prediction)
            chosen_action_name = self.ACTION_NAMES[move_idx]
            print(
                f"[Agent] Model chose action: {chosen_action_name} (index {move_idx}). Move: {move}"
            )
        return move

    def _log_epsilon(
        self: Agent, raw_eps: float, epsilon: float, random_roll: float
    ) -> None:
        print(
            f"[Agent] Epsilon calculation: initial_eps={self.config.initial_epsilon}, "
            f"min_eps={self.config.min_epsilon}, decay={self.config.epsilon_decay}, "
            f"num_games={self.num_games} -> calculated_raw_eps={raw_eps:.6f} -> "
            f"current_epsilon={epsilon:.6f}"
        )
        print(
            f"[Agent] Epsilon check: random_roll={random_roll:.6f} vs current_epsilon={epsilon:.6f}"
        )

    def _log_explore(self: Agent, action_idx: int, move: list[int]) -> None:
        chosen_action_name = self.ACTION_NAMES[action_idx]
        print(
            f"[Agent] Action choice: EXPLORE (random_roll < epsilon). "
            f"Random action chosen: {chosen_action_name} (index {action_idx}). Move: {move}"
        )

    def get_state(self: Agent, env: Environment) -> np.ndarray:
        """
        20-float vector in [F, B, L, R] order relative to the snake’s current heading:
            0-3   wall distance
            4-7   green-apple distance   (nearest one)
            8-11  red-apple  distance
            12-15 body distance
            16-19 collision-on-next-step (0/1)
        """
        W, H = env.width, env.height
        head: Pos = env.head
        hx, hy = head.x, head.y
        head_dir: Direction = env.direction
        body_set = {(p.x, p.y) for p in env.snake[1:]}  # exclude head

        directions = self._direction_vectors(head_dir)  # [F, B, L, R]

        state: list[float] = []
        for dx, dy in directions:
            state.append(self._wall_distance(W, H, hx, hy, dx, dy))  # 0-3
        for dx, dy in directions:
            target = self._closest_apple_in_dir(hx, hy, dx, dy, env.green_apples)
            state.append(self._item_distance(W, H, hx, hy, dx, dy, target))  # 4-7
        for dx, dy in directions:
            red_target = (
                self._closest_apple_in_dir(hx, hy, dx, dy, [env.red_apple])
                if env.red_apple
                else None
            )
            state.append(self._item_distance(W, H, hx, hy, dx, dy, red_target))  # 8-11
        for dx, dy in directions:
            state.append(self._body_distance(W, H, hx, hy, dx, dy, body_set))  # 12-15
        for dx, dy in directions:
            state.append(self._collision_flag(W, H, hx, hy, dx, dy, body_set))  # 16-19

        if self.step_by_step:
            print(
                f"[Agent] Got state of the current game [{self._format_state_for_log(state)}]"
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
        raw_eps, epsilon = self._update_epsilon()
        random_roll = random.random()

        if self.step_by_step:
            self._log_epsilon(raw_eps, epsilon, random_roll)

        if random_roll < epsilon:
            action_idx = random.randint(0, 2)
            move = self._one_hot_action(action_idx)
            if self.step_by_step:
                self._log_explore(action_idx, move)
            return move

        if self.step_by_step:
            print("[Agent] Action choice: EXPLOIT (random_roll >= epsilon)")
        return self._exploit_action(state)

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
