from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from board import Board
from constants import BATCH_SIZE, BLOCK_SIZE, LR, MAX_MEMORY, Direction, Pos
from model import LinearQNet, QTrainer


class Agent:
    def __init__(self: Agent) -> None:
        self.num_games: int = 0
        self.epsilon: int = 0
        self.gamma: float = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = LinearQNet(19, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def _is_trap(self, game: Board, start_point, direction):
        current_point = start_point
        for _ in range(3):
            if game.is_collision(current_point):
                return True  # Collision detected early (trapped)
            # Move forward
            if direction == Direction.LEFT:
                current_point = Pos(current_point.x - 1, current_point.y)
            elif direction == Direction.RIGHT:
                current_point = Pos(current_point.x + 1, current_point.y)
            elif direction == Direction.UP:
                current_point = Pos(current_point.x, current_point.y - 1)
            elif direction == Direction.DOWN:
                current_point = Pos(current_point.x, current_point.y + 1)
        return False  # Path is clear at least 'steps' tiles ahead

    def get_state(self: Agent, game: Board) -> np.array:
        head: Pos = game.snake[0]
        point_l = Pos(head.x - BLOCK_SIZE, head.y)
        point_r = Pos(head.x + BLOCK_SIZE, head.y)
        point_u = Pos(head.x, head.y - BLOCK_SIZE)
        point_d = Pos(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # food direction
            game.red_apple.x < game.head.x,
            game.red_apple.x > game.head.x,
            game.red_apple.y < game.head.y,
            game.red_apple.y > game.head.y,
            # adjacent squares occupied (body-awareness fix)
            game.is_collision(point_l),
            game.is_collision(point_r),
            game.is_collision(point_u),
            game.is_collision(point_d),
        ]
        state.extend(
            [
                self._is_trap(game, point_l, Direction.LEFT),
                self._is_trap(game, point_r, Direction.RIGHT),
                self._is_trap(game, point_u, Direction.UP),
                self._is_trap(game, point_d, Direction.DOWN),
            ]
        )
        return np.array(state, dtype=int)

    def remember(
        self: Agent,
        state: np.array,
        action,
        reward,
        next_state: np.array,
        done,
    ):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self: Agent):
        if len(self.memory) > BATCH_SIZE:
            # list of tuples
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # below is same
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(
        self: Agent,
        state: np.array,
        action,
        reward,
        next_state: np.array,
        done,
    ):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self: Agent, state: np.array):
        # random moves: tradoff exploration / exploitation
        self.epsilon = 80 - self.num_games
        final_move: list = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.forward(state0)
            move: int = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
