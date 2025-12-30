from __future__ import annotations

import random

import numpy as np

from constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GREEN_APPLE_COUNT,
    Direction,
    Pos,
)
from config_loader import EnvConfig, RewardConfig


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


class Environment:
    def __init__(
        self,
        width: int = BOARD_WIDTH,
        height: int = BOARD_HEIGHT,
        reward_cfg: RewardConfig | None = None,
        env_cfg: EnvConfig | None = None,
        step_by_step: bool = False,
    ) -> None:
        if reward_cfg is None or env_cfg is None:
            raise ValueError("RewardConfig and EnvConfig are required")
        self.width: int = width
        self.height: int = height
        self.step_by_step: bool = step_by_step
        self.reward_cfg: RewardConfig = reward_cfg
        self.env_cfg: EnvConfig = env_cfg

        self.reset()

    def _log(self, msg: str) -> None:
        if self.step_by_step:
            print(msg)

    def _nearest_green_distance(self) -> int:
        """Return L1-distance from point to the closest green apple."""
        if not self.green_apples:
            return 0
        return min(manhattan(self.head, g) for g in self.green_apples)

    def _starvation_limit(self) -> int:
        if self.env_cfg is None:
            raise ValueError("EnvConfig is required")
        factor = self.env_cfg.starve_factor
        return factor * len(self.snake) if self.snake else factor

    def reset(self) -> None:
        """Return the board to its initial state."""
        self.head: Pos = Pos(
            random.randint(1, self.width - 3),
            random.randint(0, self.height - 3),
        )
        dx = -1 if self.head.x > 1 else 1

        self.snake: list[Pos] = [
            self.head,
            Pos(self.head.x + dx, self.head.y),
            Pos(self.head.x + 2 * dx, self.head.y),
        ]

        self.direction = Direction.RIGHT if dx == -1 else Direction.LEFT

        self.red_apple: Pos | None = None
        self.green_apples: list[Pos] = []
        self._place_red_apple()
        self._place_green_apples()
        self.frame: int = 0
        self.frames_since_food: int = 0
        self.prev_dist = self._nearest_green_distance()

    @staticmethod
    def _action_name(action_vec: np.ndarray | list[int]) -> str:
        if isinstance(action_vec, np.ndarray):
            action_vec = action_vec.tolist()
        if action_vec == [1, 0, 0]:
            return "straight"
        if action_vec == [0, 1, 0]:
            return "right_turn"
        if action_vec == [0, 0, 1]:
            return "left_turn"
        return str(action_vec)

    def _handle_apples(
        self,
        reward: float,
        cur_dist: int,
    ) -> tuple[float, bool, bool, int]:
        if self.reward_cfg is None:
            raise ValueError("RewardConfig is required")
        cfg = self.reward_cfg
        grew = shrink = False

        if self.red_apple and self.head == self.red_apple:
            eaten_red = self.red_apple
            reward = cfg.red_apple
            self._place_red_apple()
            self.frames_since_food = 0
            shrink = True
            self._log(
                (
                    "[Environment] Event: Ate RED apple at "
                    f"{eaten_red}. Reward: {reward}. "
                    "Frames since food reset. Flagged to shrink."
                )
            )
            placed_at = (
                self.red_apple if self.red_apple else "None (no space?)"
            )
            self._log(f"[Environment] New red apple placed at: {placed_at}")

        elif self.head in self.green_apples:
            self.green_apples.remove(self.head)
            reward = cfg.green_apple
            self.frames_since_food = 0
            grew = True
            self._log(
                (
                    "[Environment] Event: Ate GREEN apple at "
                    f"{self.head}. Reward: {reward}. Frames since food reset. "
                    "Flagged to grow."
                )
            )
            self._place_green_apples()
            self._log(
                "[Environment] Green apples updated. Current green apples: "
                f"{len(self.green_apples)} positions: {self.green_apples}"
            )
            cur_dist = self._nearest_green_distance()

        elif cur_dist < self.prev_dist:
            reward += cfg.nearest_closer
            self._log(
                (
                    "[Environment] Event: Snake head got closer to closest "
                    f"green apple. Reward: {reward}."
                )
            )

        elif cur_dist > self.prev_dist:
            reward += cfg.nearest_further
            self._log(
                (
                    "[Environment] Event: Snake head got further from closest "
                    f"green apple. Reward: {reward}."
                )
            )

        return reward, grew, shrink, cur_dist

    def _update_length(self, grew: bool, shrink: bool) -> bool:
        if not grew and self.snake:
            self.snake.pop()
        elif grew:
            self._log(
                (
                    "[Environment] Length update: Grown (green apple), "
                    f"tail not popped. Snake length now: {len(self.snake)}"
                )
            )

        if shrink and self.snake:
            popped_item = self.snake.pop()
            self._log(
                (
                    "[Environment] Length update: Shrunk (red apple), "
                    f"tail popped ({popped_item}). Snake length now: "
                    f"{len(self.snake)}"
                )
            )
            if not self.snake:
                if self.reward_cfg is None:
                    raise ValueError("RewardConfig is required")
                self._log(
                    (
                        "[Environment] GAME OVER: Snake shrunk to nothing. "
                        f"Final Reward: {self.reward_cfg.death}, "
                        f"Final Score: {len(self.snake)}"
                    )
                )
                return True
        return False

    def step(
        self,
        action: np.ndarray | list[int],
    ) -> tuple[int, bool, int]:
        """Advance one tick; return (reward, done, score)."""
        action_name = self._action_name(action)
        self.frame += 1
        self.frames_since_food += 1
        self._log(
            (
                f"[Environment] Frame: {self.frame}, Frames since food: "
                f"{self.frames_since_food}, Received action: "
                f"{action_name} ({action})"
            )
        )
        self._log(
            (
                "[Environment] Current snake head: "
                f"{self.head}, Snake length: {len(self.snake)}"
            )
        )

        self._move(action)
        self.snake.insert(0, self.head)

        self._log(
            (
                "[Environment] Movement: Action "
                f"'{action_name}' led to new head position: {self.head}."
            )
        )

        cur_dist = self._nearest_green_distance()

        if self.reward_cfg is None:
            raise ValueError("RewardConfig is required")
        reward = self.reward_cfg.living_step
        self._log(f"[Environment] Initial reward for step: {reward}")
        reward, grew, shrink, cur_dist = self._handle_apples(reward, cur_dist)

        if self._update_length(grew, shrink):
            return self.reward_cfg.death, True, len(self.snake)

        if self.is_collision():
            self._log(
                (
                    "[Environment] GAME OVER: Collision detected at head "
                    f"position {self.head}. Snake: {self.snake}. "
                    f"Final Reward: {self.reward_cfg.death}, "
                    f"Final Score: {len(self.snake)}"
                )
            )
            return self.reward_cfg.death, True, len(self.snake)

        starvation_limit = self._starvation_limit()
        if self.frames_since_food > starvation_limit:
            self._log(
                (
                    "[Environment] GAME OVER: Starvation. Frames since "
                    f"food ({self.frames_since_food}) > "
                    "limit "
                    f"({self.env_cfg.starve_factor} * {len(self.snake)} = "
                    f"{starvation_limit}). Final Reward: "
                    f"{self.reward_cfg.death}, Final Score: {len(self.snake)}"
                )
            )
            return self.reward_cfg.death, True, len(self.snake)

        self._log(
            (
                "[Environment] Step finished successfully. "
                f"Reward: {reward}, Game Over: False, Score: {len(self.snake)}"
            )
        )

        self.prev_dist = cur_dist
        return reward, False, len(self.snake)

    def _random_empty_tile(self) -> Pos:
        """Return a random tile not occupied by the snake or apples."""
        obstacles = set(self.snake + self.green_apples)
        if self.red_apple:
            obstacles.add(self.red_apple)
        max_attempts = self.width * self.height
        for _ in range(max_attempts):
            p = Pos(
                random.randrange(self.width),
                random.randrange(self.height),
            )
            if p not in obstacles:
                return p
        return Pos(0, 0)

    def _place_red_apple(self) -> None:
        self.red_apple = self._random_empty_tile()

    def _place_green_apples(self) -> None:
        while len(self.green_apples) < GREEN_APPLE_COUNT:
            self.green_apples.append(self._random_empty_tile())

    def is_collision(self, pt: Pos | None = None) -> bool:
        """Check collision at pt; default is current head."""
        check_point = pt if pt is not None else self.head

        if not (
            0 <= check_point.x < self.width
            and 0 <= check_point.y < self.height
        ):
            return True

        if check_point == self.head and check_point in self.snake[1:]:
            return True
        if pt is not None and check_point in self.snake:
            return True
        return False

    def _move(self, action: np.ndarray | list[int]) -> None:
        """Update the head position based on action."""
        clockwise_directions = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
        current_direction_idx = clockwise_directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_direction = self.direction
        elif np.array_equal(action, [0, 1, 0]):
            new_direction_idx = (current_direction_idx + 1) % 4
            new_direction = clockwise_directions[new_direction_idx]
        else:
            new_direction_idx = (current_direction_idx - 1 + 4) % 4
            new_direction = clockwise_directions[new_direction_idx]

        self.direction = new_direction

        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
        self.head = Pos(x, y)
