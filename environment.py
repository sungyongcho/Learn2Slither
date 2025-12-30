from __future__ import annotations

import random

import numpy as np

from constants import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    GREEN_APPLE_COUNT,
    REWARD_DEATH,
    REWARD_GREEN_APPLE,
    REWARD_LIVING_STEP,
    REWARD_NEAREST_CLOSER,
    REWARD_NEAREST_FURTHER,
    REWARD_RED_APPLE,
    STARVE_FACTOR,
    Direction,
    Pos,
)


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


class Environment:
    """Snake game logic on an (m x n) grid, no rendering involved."""

    def __init__(
        self,
        width: int = BOARD_WIDTH,
        height: int = BOARD_HEIGHT,
        step_by_step: bool = False,
    ) -> None:
        self.width: int = width
        self.height: int = height
        self.step_by_step: bool = step_by_step

        self.reset()

    def _log(self, msg: str) -> None:
        if self.step_by_step:
            print(msg)

    def _nearest_green_distance(self) -> int:
        """Return L1-distance from point to the closest green apple."""
        if not self.green_apples:  # failsafe – should not happen
            return 0
        return min(manhattan(self.head, g) for g in self.green_apples)

    def _starvation_limit(self) -> int:
        return STARVE_FACTOR * len(self.snake) if self.snake else STARVE_FACTOR

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
        grew = shrink = False

        if self.red_apple and self.head == self.red_apple:
            eaten_red = self.red_apple
            reward = REWARD_RED_APPLE
            self._place_red_apple()
            self.frames_since_food = 0
            shrink = True
            self._log(
                f"[Environment] Event: Ate RED apple at {eaten_red}. Reward: {reward}. "
                f"Frames since food reset. Flagged to shrink."
            )
            self._log(
                f"[Environment] New red apple placed at: {self.red_apple if self.red_apple else 'None (no space?)'}"
            )

        elif self.head in self.green_apples:
            self.green_apples.remove(self.head)
            reward = REWARD_GREEN_APPLE
            self.frames_since_food = 0
            grew = True
            self._log(
                f"[Environment] Event: Ate GREEN apple at {self.head}. Reward: {reward}. "
                f"Frames since food reset. Flagged to grow."
            )
            self._place_green_apples()
            self._log(
                f"[Environment] Green apples updated. Current green apples: {len(self.green_apples)} positions: {self.green_apples}"
            )
            cur_dist = self._nearest_green_distance()

        elif cur_dist < self.prev_dist:
            reward += REWARD_NEAREST_CLOSER
            self._log(
                f"[Environment] Event: Snake head got closer to closest green apple. Reward: {reward}."
            )

        elif cur_dist > self.prev_dist:
            reward += REWARD_NEAREST_FURTHER
            self._log(
                f"[Environment] Event: Snake head got further from closest green apple. Reward: {reward}."
            )

        return reward, grew, shrink, cur_dist

    def _update_length(self, grew: bool, shrink: bool) -> bool:
        if not grew and self.snake:
            self.snake.pop()
        elif grew:
            self._log(
                f"[Environment] Length update: Grown (green apple), tail not popped. Snake length now: {len(self.snake)}"
            )

        if shrink and self.snake:
            popped_item = self.snake.pop()
            self._log(
                f"[Environment] Length update: Shrunk (red apple), tail popped ({popped_item}). Snake length now: {len(self.snake)}"
            )
            if not self.snake:
                self._log(
                    f"[Environment] GAME OVER: Snake shrunk to nothing. Final Reward: {REWARD_DEATH}, Final Score: {len(self.snake)}"
                )
                return True
        return False

    def step(
        self,
        action: np.ndarray | list[int],
    ) -> tuple[int, bool, int]:
        """Advance one tick. Return (learning_reward, game_over, display_score)."""

        action_name = self._action_name(action)
        self._log(
            f"[Environment] Frame: {self.frame + 1}, Frames since food: {self.frames_since_food + 1}, "
            f"Received action: {action_name} ({action})"
        )
        self._log(
            f"[Environment] Current snake head: {self.head}, Snake length: {len(self.snake)}"
        )

        self.frame += 1
        self.frames_since_food += 1

        # 1. Move head
        self._move(action)
        self.snake.insert(0, self.head)  # new head at index 0

        self._log(
            f"[Environment] Movement: Action '{action_name}' led to new head position: {self.head}."
        )

        cur_dist = self._nearest_green_distance()

        reward = REWARD_LIVING_STEP
        self._log(f"[Environment] Initial reward for step: {reward}")
        reward, grew, shrink, cur_dist = self._handle_apples(reward, cur_dist)

        if self._update_length(grew, shrink):
            return REWARD_DEATH, True, len(self.snake)

        # 4. Wall / body collision (after tail possibly removed)
        if (
            self.is_collision()
        ):  # is_collision should check self.head against walls and self.snake[1:]
            self._log(
                f"[Environment] GAME OVER: Collision detected at head position {self.head}. "
                f"Snake: {self.snake}. Final Reward: {REWARD_DEATH}, Final Score: {len(self.snake)}"
            )
            return REWARD_DEATH, True, len(self.snake)

        starvation_limit = self._starvation_limit()
        if self.frames_since_food > starvation_limit:
            self._log(
                f"[Environment] GAME OVER: Starvation. Frames since food ({self.frames_since_food}) > "
                f"limit ({STARVE_FACTOR} * {len(self.snake)} = {starvation_limit}). "
                f"Final Reward: {REWARD_DEATH}, Final Score: {len(self.snake)}"
            )
            return REWARD_DEATH, True, len(self.snake)

        self._log(
            f"[Environment] Step finished successfully. Reward: {reward}, Game Over: False, Score: {len(self.snake)}"
        )

        self.prev_dist = cur_dist
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

    def is_collision(self, pt: Pos | None = None) -> bool:
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

    def _move(self, action: np.ndarray | list[int]) -> None:
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
