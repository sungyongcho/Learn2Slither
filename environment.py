from __future__ import annotations

import random
from typing import List, Optional, Tuple, Union

import numpy as np

from constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GREEN_APPLE_COUNT,
    STARVE_FACTOR,
    Direction,
    Pos,
)


class Environment:
    """Snake game logic on an (m x n) grid, no rendering involved."""

    def __init__(
        self,
        width: int = BOARD_WIDTH,
        height: int = BOARD_HEIGHT,
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.reset()

    def reset(self) -> None:
        """Return the board to its initial state."""
        self.direction: Direction = Direction.RIGHT
        self.head: Pos = Pos(
            random.randint(0, self.width), random.randint(0, self.height)
        )
        self.snake: List[Pos] = [
            self.head,
            Pos(self.head.x - 1, self.head.y),
            Pos(self.head.x - 2, self.head.y),
        ]
        self.score: int = 0
        self.red_apple: Optional[Pos] = None
        self.green_apples: List[Pos] = []
        self._place_red_apple()
        self._place_green_apples()
        self.frame_counter: int = 0
        self.frames_since_food: int = 0

    def step(
        self,
        action: Union[np.ndarray, List[int]],
    ) -> Tuple[int, bool, int]:
        """Advance one tick. Return (reward, game_over, score)."""
        self.frame_counter += 1
        self.frames_since_food += 1

        # move head
        self._move(action)
        self.snake.insert(0, self.head)

        reward: float = -0.5  # living cost
        grew: bool = False

        # collision or starvation
        if self.is_collision() or (
            self.frames_since_food > (STARVE_FACTOR * len(self.snake))
        ):
            return -1, True, self.score

        # red apple
        if self.head == self.red_apple:
            self.score += 1
            reward = -5
            self._place_red_apple()
            self.frames_since_food = 0
        # green apple
        elif self.head in self.green_apples:
            self.green_apples.remove(self.head)
            self.score += 10
            reward = 10
            self._place_green_apples()
            self.frames_since_food = 0
            grew = True

        # drop tail if did not grow
        if not grew:
            self.snake.pop()

        return int(reward), False, self.score

    def _random_empty_tile(self) -> Pos:
        """Return a random tile not occupied by the snake or apples."""
        while True:
            p = Pos(
                random.randrange(self.width),
                random.randrange(self.height),
            )
            if (
                p not in self.snake
                and p != self.red_apple
                and p not in self.green_apples
            ):
                return p

    def _place_red_apple(self) -> None:
        self.red_apple = self._random_empty_tile()

    def _place_green_apples(self) -> None:
        while len(self.green_apples) < GREEN_APPLE_COUNT:
            self.green_apples.append(self._random_empty_tile())

    def is_collision(self, pt: Optional[Pos] = None) -> bool:
        if pt is None:
            pt = self.head
        # wall
        if pt.x < 0 or pt.x >= self.width or pt.y < 0 or pt.y >= self.height:
            return True
        # self
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action: Union[np.ndarray, List[int]]) -> None:
        """Update the head position based on action."""
        clockwise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
        idx = clockwise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):  # straight
            new_dir = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            new_dir = clockwise[(idx + 1) % 4]
        else:  # left turn
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head.x, self.head.y
        match self.direction:
            case Direction.RIGHT:
                x += 1
            case Direction.LEFT:
                x -= 1
            case Direction.DOWN:
                y += 1
            case Direction.UP:
                y -= 1

        self.head = Pos(x, y)
