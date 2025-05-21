from __future__ import annotations

import random
from typing import List, Optional, Tuple, Union

import numpy as np

from constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GREEN_APPLE_COUNT,
    REWARD_DEATH,
    REWARD_GREEN_APPLE,
    REWARD_LIVING_STEP,
    REWARD_RED_APPLE,
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
        self.direction: Direction = random.choice(list(Direction))
        self.head: Pos = Pos(
            random.randint(0, self.width - 3),  # Ensure space for initial
            random.randint(0, self.height - 3),
        )
        self.snake: List[Pos] = [
            self.head,
            Pos(
                self.head.x - 1 if self.head.x > 0 else self.head.x + 1,
                self.head.y,
            ),
            Pos(
                self.head.x - 2 if self.head.x > 1 else self.head.x + 2,
                self.head.y,
            ),
        ]

        self.red_apple: Optional[Pos] = None
        self.green_apples: List[Pos] = []
        self._place_red_apple()
        self._place_green_apples()  # Ensure this can always place apples
        self.frame: int = 0
        self.frames_since_food: int = 0

    def step(
        self,
        action: Union[np.ndarray, List[int]],
    ) -> Tuple[int, bool, int]:
        """Advance one tick. Return (learning_reward, game_over, display_score)."""

        self.frame += 1
        self.frames_since_food += 1

        # 1. Move head
        self._move(action)
        self.snake.insert(0, self.head)  # new head at index 0

        grew = shrink = False
        reward = REWARD_LIVING_STEP

        # 2. Apple checks ────────────────
        if self.red_apple and self.head == self.red_apple:
            reward = REWARD_RED_APPLE
            self._place_red_apple()
            self.frames_since_food = 0
            shrink = True

        elif self.head in self.green_apples:
            self.green_apples.remove(self.head)
            reward = REWARD_GREEN_APPLE
            self._place_green_apples()
            self.frames_since_food = 0
            grew = True

        # 3. Update length ──────────────
        if not grew and self.snake:
            self.snake.pop()  # normal move → drop tail
        if shrink and self.snake:
            self.snake.pop()  # red apple → shrink extra
            if not self.snake:  # shrunk away completely
                return REWARD_DEATH, True, len(self.snake)

        # 4. Wall / body collision (after tail possibly removed)
        if self.is_collision():
            return REWARD_DEATH, True, len(self.snake)

        # 5. Starvation check (uses final length)
        if self.frames_since_food > STARVE_FACTOR * len(self.snake):
            return REWARD_DEATH, True, len(self.snake)

        return reward, False, len(self.snake)

    def _random_empty_tile(self) -> Pos:
        """Return a random tile not occupied by the snake or apples."""
        # Add a limit to prevent infinite loops if the board is nearly full,
        # though for typical Snake, this is rare.
        max_attempts = self.width * self.height
        for _ in range(max_attempts):
            p = Pos(
                random.randrange(self.width),
                random.randrange(self.height),
            )
            # Check against current snake, red apple, and all green apples
            all_obstacles = self.snake + self.green_apples
            if self.red_apple:
                all_obstacles.append(self.red_apple)

            if p not in all_obstacles:
                return p
        # Fallback or error if no empty tile is found (board is full)
        # This case should ideally not be reached in a playable game.
        # For simplicity, let's assume it's always found for now.
        # A robust solution might raise an error or return a default if board is full.
        # print("Warning: Could not find an empty tile. Board might be full.")
        return Pos(0, 0)  # Placeholder if no empty tile found, needs better handling

    def _place_red_apple(self) -> None:
        self.red_apple = self._random_empty_tile()

    def _place_green_apples(self) -> None:
        # Ensure GREEN_APPLE_COUNT doesn't exceed available board space
        # For simplicity, assuming it's a reasonable number.
        while len(self.green_apples) < GREEN_APPLE_COUNT:
            # What if adding an apple fails due to full board?
            # _random_empty_tile needs to be robust.
            self.green_apples.append(self._random_empty_tile())

    def is_collision(self, pt: Optional[Pos] = None) -> bool:
        """Checks for collision at point pt. If pt is None, checks current head."""
        check_point = pt if pt is not None else self.head

        # Wall collision
        if not (0 <= check_point.x < self.width and 0 <= check_point.y < self.height):
            return True

        # Self-collision
        # If pt is None, we check self.head against self.snake[1:]
        # If pt is provided, we check pt against the *entire* current snake body
        # The original logic was: if pt in self.snake[1:]. This is for self.head.
        # If pt is an arbitrary point, it should be checked against the whole snake.
        # However, for checking potential next moves for the agent's state,
        # pt would be a cell adjacent to current head.

        # For `is_collision()` called from `step()` (pt is None, so check_point is self.head):
        if (
            check_point == self.head and check_point in self.snake[1:]
        ):  # Head collided with its body
            return True
        # For `is_collision(some_other_point)` (e.g. from agent's get_state):
        elif (
            pt is not None and check_point in self.snake
        ):  # The given point is occupied by any part of the snake
            return True

        return False

    def _move(self, action: Union[np.ndarray, List[int]]) -> None:
        """Update the head position based on action."""
        # action is [straight, right_turn, left_turn]
        # e.g., [1,0,0] is straight, [0,1,0] is right, [0,0,1] is left

        clockwise_directions = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
        current_direction_idx = clockwise_directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  # Straight
            new_direction = self.direction
        elif np.array_equal(action, [0, 1, 0]):  # Right turn
            new_direction_idx = (current_direction_idx + 1) % 4
            new_direction = clockwise_directions[new_direction_idx]
        else:  # np.array_equal(action, [0, 0, 1]) # Left turn
            new_direction_idx = (
                current_direction_idx - 1 + 4
            ) % 4  # +4 to handle negative result
            new_direction = clockwise_directions[new_direction_idx]

        self.direction = new_direction

        # Update head position
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
