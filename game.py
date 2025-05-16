from __future__ import annotations

import random
from collections import namedtuple
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import pygame

pygame.init()
font: pygame.font.Font = pygame.font.Font("arial.ttf", 25)


class Direction(Enum):
    RIGHT: int = 1
    LEFT: int = 2
    UP: int = 3
    DOWN: int = 4


Point = namedtuple("Point", "x, y")

# RGB colours
WHITE: Tuple[int, int, int] = (255, 255, 255)
RED: Tuple[int, int, int] = (200, 0, 0)
BLUE1: Tuple[int, int, int] = (0, 0, 255)
BLUE2: Tuple[int, int, int] = (0, 100, 255)
BLACK: Tuple[int, int, int] = (0, 0, 0)

BLOCK_SIZE: int = 20
SPEED: int = 20


class SnakeGame:
    """Classic Snake game implemented with Pygame."""

    def __init__(self: SnakeGame, w: int = 640, h: int = 480) -> None:
        self.w: int = w
        self.h: int = h
        self.display: pygame.Surface = pygame.display.set_mode(
            (self.w, self.h),
        )
        pygame.display.set_caption("Snake")
        self.clock: pygame.time.Clock = pygame.time.Clock()
        self.reset()

    def reset(self: SnakeGame) -> None:
        """Reset the game to its initial state."""
        self.direction: Direction = Direction.RIGHT
        self.head: Point = Point(self.w / 2, self.h / 2)
        self.snake: List[Point] = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score: int = 0
        self.food: Optional[Point] = None
        self._place_food()
        self.frame_iteration: int = 0

    def _place_food(self: SnakeGame) -> None:
        """Place food at a random empty cell (≤ 79 chars per line)."""
        max_x: int = (self.w - BLOCK_SIZE) // BLOCK_SIZE
        max_y: int = (self.h - BLOCK_SIZE) // BLOCK_SIZE

        x: int = random.randint(0, max_x) * BLOCK_SIZE
        y: int = random.randint(0, max_y) * BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def play_step(
        self: SnakeGame,
        action: Union[np.ndarray, List[int]],
    ) -> Tuple[int, bool, int]:
        """Advance one frame.

        Args:
            action: One-hot [straight, right, left].

        Returns:
            (reward, game_over, current_score)
        """
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        reward: int = 0
        game_over: bool = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward += 10
            self._place_food()
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self: SnakeGame, pt: Optional[Point] = None) -> bool:
        """Return True if *pt* hits a wall or the snake itself."""
        if pt is None:
            pt = self.head
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self: SnakeGame) -> None:
        """Render the current frame."""
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(
                self.display,
                BLUE1,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display,
                BLUE2,
                pygame.Rect(pt.x + 4, pt.y + 4, 12, 12),
            )
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(
        self: SnakeGame,
        action: Union[np.ndarray, List[int]],
    ) -> None:
        """Update head position according to *action*."""
        clockwise: List[Direction] = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
        idx: int = clockwise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir: Direction = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clockwise[(idx + 1) % 4]
        else:  # [0, 0, 1]
            new_dir = clockwise[(idx - 1) % 4]

        self.direction = new_dir

        x: float = self.head.x
        y: float = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


if __name__ == "__main__":
    game = SnakeGame()

    while True:
        # Replace with real control/AI input
        random_action: List[int] = random.choice(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        reward, game_over, score = game.play_step(random_action)
        if game_over:
            break

    print("Final Score", score)
    pygame.quit()
