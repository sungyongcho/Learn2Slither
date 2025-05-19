from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

from constants import BATCH_SIZE, LR, MAX_MEMORY, Direction, Pos
from environment import Environment
from model import LinearQNet, QTrainer


class Agent:
    def __init__(self: Agent) -> None:
        self.num_games: int = 0
        self.epsilon: int = 0
        self.gamma: float = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def _is_trap(self, game: Environment, start_point, direction):
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

    # def get_state(self: Agent, game: Environment) -> np.array:
    #     head: Pos = game.snake[0]
    #     point_l = Pos(head.x - 1, head.y)
    #     point_r = Pos(head.x + 1, head.y)
    #     point_u = Pos(head.x, head.y - 1)
    #     point_d = Pos(head.x, head.y + 1)

    #     dir_l = game.direction == Direction.LEFT
    #     dir_r = game.direction == Direction.RIGHT
    #     dir_u = game.direction == Direction.UP
    #     dir_d = game.direction == Direction.DOWN

    #     red_left = game.red_apple.x < game.head.x
    #     red_right = game.red_apple.x > game.head.x
    #     red_up = game.red_apple.y < game.head.y
    #     red_down = game.red_apple.y > game.head.y
    #     # -------- choose the nearest green apple --------

    #     greens = (
    #         game.green_apples  # if you store them in a list
    #         if hasattr(game, "green_apples")
    #         else [game.green_apple1, game.green_apple2]
    #     )
    #     nearest_green = min(
    #         greens,
    #         key=lambda p: abs(p.x - head.x) + abs(p.y - head.y),  # Manhattan dist
    #     )

    #     green_left = nearest_green.x < head.x
    #     green_right = nearest_green.x > head.x
    #     green_up = nearest_green.y < head.y
    #     green_down = nearest_green.y > head.y
    #     state = [
    #         # danger straight
    #         (dir_r and game.is_collision(point_r))
    #         or (dir_l and game.is_collision(point_l))
    #         or (dir_u and game.is_collision(point_u))
    #         or (dir_d and game.is_collision(point_d)),
    #         # danger right
    #         (dir_u and game.is_collision(point_r))
    #         or (dir_d and game.is_collision(point_l))
    #         or (dir_l and game.is_collision(point_u))
    #         or (dir_r and game.is_collision(point_d)),
    #         # danger left
    #         (dir_d and game.is_collision(point_r))
    #         or (dir_u and game.is_collision(point_l))
    #         or (dir_r and game.is_collision(point_u))
    #         or (dir_l and game.is_collision(point_d)),
    #         # move direction
    #         dir_l,
    #         dir_r,
    #         dir_u,
    #         dir_d,
    #         # food direction
    #         game.red_apple.x < game.head.x,
    #         game.red_apple.x > game.head.x,
    #         game.red_apple.y < game.head.y,
    #         game.red_apple.y > game.head.y,
    #         # adjacent squares occupied (body-awareness fix)
    #         game.is_collision(point_l),
    #         game.is_collision(point_r),
    #         game.is_collision(point_u),
    #         game.is_collision(point_d),
    #     ]
    #     state.extend(
    #         [
    #             self._is_trap(game, point_l, Direction.LEFT),
    #             self._is_trap(game, point_r, Direction.RIGHT),
    #             self._is_trap(game, point_u, Direction.UP),
    #             self._is_trap(game, point_d, Direction.DOWN),
    #             green_left,
    #             green_right,
    #             green_up,
    #             green_down,
    #         ]
    #     )
    #     return np.array(state, dtype=int)

    def get_state(self, game: Environment) -> np.ndarray:
        head: Pos = game.snake[0]

        # --- 1.  immediate danger flags (4) -----------------
        point_l = Pos(head.x - 1, head.y)
        point_r = Pos(head.x + 1, head.y)
        point_u = Pos(head.x, head.y - 1)
        point_d = Pos(head.x, head.y + 1)

        danger = [
            game.is_collision(point_r),  # right
            game.is_collision(point_l),  # left
            game.is_collision(point_u),  # up
            game.is_collision(point_d),  # down
        ]

        # --- 2.  normalised distances to the nearest apple (4) -------------
        # choose nearest red / green (Manhattan)
        apples = [game.red_apple] + (
            game.green_apples
            if hasattr(game, "green_apples")
            else [game.green_apple1, game.green_apple2]
        )
        nearest = min(apples, key=lambda p: abs(p.x - head.x) + abs(p.y - head.y))

        dx = nearest.x - head.x
        dy = nearest.y - head.y
        # normalise to [-1,1]
        norm_dx = dx / (game.width // 2)
        norm_dy = dy / (game.height // 2)

        # --- 3.  distance to walls (4) -------------------------------------
        dist_wall = [
            head.x / (game.width - 1),  # left wall  (0 centre → 1 at wall)
            (game.width - 1 - head.x) / (game.width - 1),  # right wall
            head.y / (game.height - 1),  # top wall
            (game.height - 1 - head.y) / (game.height - 1),  # bottom wall
        ]

        # --- 4.  snake length mod 32 (1) -----------------------------------
        length_mod = (len(game.snake) % 32) / 31.0  # scaled to [0,1]

        # concatenate everything
        state = danger + [norm_dx, norm_dy] + dist_wall + [length_mod]
        return np.array(state, dtype=np.float32)

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
