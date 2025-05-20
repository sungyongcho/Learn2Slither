from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from constants import BATCH_SIZE, LR, MAX_MEMORY, Direction, Pos
from environment import Environment
from model import LinearQNet, QTrainer


class Agent:
    def __init__(
        self: Agent,
        input_size: int = 27,  # Default based on your original state size
        initial_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay_rate: float = 0.995,
    ) -> None:
        self.num_games: int = 0
        # Epsilon parameters for exploration-exploitation trade-off
        self.initial_epsilon: float = initial_epsilon
        self.min_epsilon: float = min_epsilon
        self.epsilon_decay_rate: float = epsilon_decay_rate
        self.epsilon: float = self.initial_epsilon  # Current epsilon

        self.gamma: float = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft() when full

        # IMPORTANT: Ensure input_size matches the actual number of features from get_state()
        self.model = LinearQNet(input_size, 256, 3)  # 3 actions: straight, right, left
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def _is_trap(
        self, game: Environment, start_point: Pos, direction: Direction
    ) -> bool:
        """Checks if a path in a given direction from start_point leads into a trap within 3 steps."""
        current_point = start_point
        for _ in range(3):  # Look ahead 3 steps
            # Move current_point one step in the given direction
            if direction == Direction.LEFT:
                current_point = Pos(current_point.x - 1, current_point.y)
            elif direction == Direction.RIGHT:
                current_point = Pos(current_point.x + 1, current_point.y)
            elif direction == Direction.UP:
                current_point = Pos(current_point.x, current_point.y - 1)
            elif direction == Direction.DOWN:
                current_point = Pos(current_point.x, current_point.y + 1)

            if game.is_collision(current_point):
                return True  # Collision detected, it's a trap
        return False  # Path is clear for at least 3 steps

    def _get_apple_direction_features(self, head: Pos, apple: Pos | None) -> list[bool]:
        """Helper to get direction features for a single apple."""
        if apple is None:
            return [False, False, False, False]  # Apple doesn't exist or not found

        return [
            apple.x < head.x,  # Apple is to the left
            apple.x > head.x,  # Apple is to the right
            apple.y < head.y,  # Apple is up
            apple.y > head.y,  # Apple is down
        ]

    def get_state(self: Agent, game: Environment) -> np.array:
        head: Pos = game.head  # Assuming game.head is the snake's head position

        # Define points around the head (potential next cells in absolute directions)
        point_l = Pos(head.x - 1, head.y)
        point_r = Pos(head.x + 1, head.y)
        point_u = Pos(head.x, head.y - 1)
        point_d = Pos(head.x, head.y + 1)

        # Current direction of the snake
        dir_is_l = game.direction == Direction.LEFT
        dir_is_r = game.direction == Direction.RIGHT
        dir_is_u = game.direction == Direction.UP
        dir_is_d = game.direction == Direction.DOWN

        # 1. Relative Danger (3 features)
        # Danger if moving straight (relative to current direction)
        danger_straight = (
            (dir_is_r and game.is_collision(point_r))
            or (dir_is_l and game.is_collision(point_l))
            or (dir_is_u and game.is_collision(point_u))
            or (dir_is_d and game.is_collision(point_d))
        )

        # Danger if turning right (relative to current direction)
        danger_right_turn = (
            (dir_is_u and game.is_collision(point_r))
            or (dir_is_d and game.is_collision(point_l))
            or (dir_is_l and game.is_collision(point_u))
            or (dir_is_r and game.is_collision(point_d))
        )

        # Danger if turning left (relative to current direction)
        danger_left_turn = (
            (dir_is_d and game.is_collision(point_r))
            or (dir_is_u and game.is_collision(point_l))
            or (dir_is_r and game.is_collision(point_u))
            or (dir_is_l and game.is_collision(point_d))
        )

        # 2. Current Direction (4 features)
        # dir_is_l, dir_is_r, dir_is_u, dir_is_d

        # 3. Food Locations (3 apples * 4 features/apple = 12 features)
        # Ensure your game object has red_apple, green_apple1, green_apple2 attributes
        # and they can be None if not present.
        red_apple_features = self._get_apple_direction_features(
            head, getattr(game, "red_apple", None)
        )
        green_apple1_features = self._get_apple_direction_features(
            head, getattr(game, "green_apple1", None)
        )
        green_apple2_features = self._get_apple_direction_features(
            head, getattr(game, "green_apple2", None)
        )

        # 4. Immediate Cell Occupancy (Collision in absolute adjacent cells) (4 features)
        collision_abs_l = game.is_collision(point_l)
        collision_abs_r = game.is_collision(point_r)
        collision_abs_u = game.is_collision(point_u)
        collision_abs_d = game.is_collision(point_d)

        # 5. Advanced Trap/Lookahead (Absolute directions from head) (4 features)
        # Is there a trap if we consider moving cardinally Left from head for 3 steps?
        trap_lookahead_l = self._is_trap(game, point_l, Direction.LEFT)
        trap_lookahead_r = self._is_trap(game, point_r, Direction.RIGHT)
        trap_lookahead_u = self._is_trap(game, point_u, Direction.UP)
        trap_lookahead_d = self._is_trap(game, point_d, Direction.DOWN)

        state = [
            danger_straight,
            danger_right_turn,
            danger_left_turn,
            dir_is_l,
            dir_is_r,
            dir_is_u,
            dir_is_d,
            *red_apple_features,
            *green_apple1_features,
            *green_apple2_features,
            collision_abs_l,
            collision_abs_r,
            collision_abs_u,
            collision_abs_d,
            trap_lookahead_l,
            trap_lookahead_r,
            trap_lookahead_u,
            trap_lookahead_d,
        ]
        # Total: 3 + 4 + 12 + 4 + 4 = 27 features

        return np.array(state, dtype=int)

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
