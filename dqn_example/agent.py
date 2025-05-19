from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
from helper import plot
from model import LinearQNet, QTrainer

from game_interface import BLOCK_SIZE, Direction, Point, SnakeGame

MAX_MEMORY: int = 100_000

BATCH_SIZE: int = 1000

LR: float = 0.001


class Agent:
    def __init__(self: Agent) -> None:
        self.num_games: int = 0
        self.epsilon: int = 0
        self.gamma: float = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = LinearQNet(19, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def _is_trap(self, game, start_point, direction):
        current_point = start_point
        for _ in range(3):
            if game.is_collision(current_point):
                return True  # Collision detected early (trapped)
            # Move forward
            if direction == Direction.LEFT:
                current_point = Point(current_point.x - BLOCK_SIZE, current_point.y)
            elif direction == Direction.RIGHT:
                current_point = Point(current_point.x + BLOCK_SIZE, current_point.y)
            elif direction == Direction.UP:
                current_point = Point(current_point.x, current_point.y - BLOCK_SIZE)
            elif direction == Direction.DOWN:
                current_point = Point(current_point.x, current_point.y + BLOCK_SIZE)
        return False  # Path is clear at least 'steps' tiles ahead

    def _get_directional_vision(
        self, game: SnakeGame, direction: Direction
    ) -> list[float]:
        head = game.snake[0]
        x, y = head.x, head.y
        dx, dy = 0, 0

        if direction == Direction.LEFT:
            dx = -BLOCK_SIZE
        elif direction == Direction.RIGHT:
            dx = BLOCK_SIZE
        elif direction == Direction.UP:
            dy = -BLOCK_SIZE
        elif direction == Direction.DOWN:
            dy = BLOCK_SIZE

        wall_hit = False
        apple_seen = False
        tail_seen = False
        steps = 0
        wall_dist = apple_dist = tail_dist = 0.0

        while True:
            x += dx
            y += dy
            steps += 1

            if x < 0 or x >= game.width or y < 0 or y >= game.height:
                wall_hit = True
                wall_dist = 1.0 - ((steps - 1) * BLOCK_SIZE) / max(
                    game.width, game.height
                )
                break

            point = Point(x, y)

            if not apple_seen and point == game.red_apple:
                apple_seen = True
                apple_dist = 1.0 - (steps * BLOCK_SIZE) / max(game.width, game.height)

            if not tail_seen and point in game.snake[1:]:
                tail_seen = True
                tail_dist = 1.0 - (steps * BLOCK_SIZE) / max(game.width, game.height)

            if apple_seen and tail_seen:
                break

        return [
            wall_dist,  # inverse dist to wall
            apple_dist if apple_seen else 0.0,  # red apple
            tail_dist if tail_seen else 0.0,  # tail
        ]

    def get_state(self: Agent, game: SnakeGame) -> np.array:
        head: Point = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

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
        self: Agent, state: np.array, action, reward, next_state: np.array, done
    ):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self: Agent):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # below is same
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(
        self: Agent, state: np.array, action, reward, next_state: np.array, done
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


def train() -> None:
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.num_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
