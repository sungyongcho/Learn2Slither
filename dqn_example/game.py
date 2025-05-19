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
GREEN: Tuple[int, int, int] = (0, 200, 0)
BLUE1: Tuple[int, int, int] = (0, 0, 255)
BLUE2: Tuple[int, int, int] = (0, 100, 255)
BLACK: Tuple[int, int, int] = (0, 0, 0)


GREEN_APPLE_COUNT: int = 2

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
        self.red_apple: Optional[Point] = None
        self.green_apples: List[Point] = []
        self._place_red_apple()
        self._place_green_apple()
        self.frame_iteration: int = 0
        self.frames_since_food: int = 0

    def _random_empty_cell(self) -> Point:
        """Return a random grid-aligned point not occupied by snake/food."""
        max_x = (self.w - BLOCK_SIZE) // BLOCK_SIZE
        max_y = (self.h - BLOCK_SIZE) // BLOCK_SIZE

        while True:
            p = Point(
                random.randint(0, max_x) * BLOCK_SIZE,
                random.randint(0, max_y) * BLOCK_SIZE,
            )
            if (
                p not in self.snake  # vacant
                and p != self.red_apple  # not the red apple
                and p not in self.green_apples
            ):
                return p  # found a free cell

    def _place_red_apple(self) -> None:
        """Spawn / replace the single red apple."""
        self.red_apple = self._random_empty_cell()

    def _place_green_apple(self) -> None:
        """Append one green apple to the board."""
        while len(self.green_apples) < GREEN_APPLE_COUNT:
            self.green_apples.append(self._random_empty_cell())

    def play_step(
        self: SnakeGame,
        action: Union[np.ndarray, List[int]],
    ) -> Tuple[int, bool, int]:
        """Advance one frame and return (reward, game_over, score)."""
        self.frame_iteration += 1  # counts total frames
        self.frames_since_food += 1  # counts frames since last apple

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)

        # ----- default reward (living cost) -----
        reward: float = -0.05
        game_over: bool = False
        grew: bool = False  # track whether we keep the tail

        # ----- death or starvation -----
        if self.is_collision() or self.frames_since_food > 100 * len(self.snake):
            reward = -10
            game_over = True
            return reward, game_over, self.score

        # ── red apple  → no growth (tail will be dropped) ─────────────────────
        if self.head == self.red_apple:
            self.score += 1
            reward = 10
            self._place_red_apple()
            self.frames_since_food = 0
            self.snake.pop()

        # ── green apple → grow (keep tail) ────────────────────────────────────
        elif self.head in self.green_apples:
            self.green_apples.remove(self.head)
            self.score += 10
            reward = 20  # give a bigger reward if you like
            self._place_green_apple()
            self.frames_since_food = 0
            grew = True  # don’t remove tail

        # ── normal move (or red-apple move) → drop tail ───────────────────────
        if not grew:
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
            pygame.Rect(self.red_apple.x, self.red_apple.y, BLOCK_SIZE, BLOCK_SIZE),
        )
        for green_apple in self.green_apples:
            pygame.draw.rect(
                self.display,
                GREEN,
                pygame.Rect(green_apple.x, green_apple.y, BLOCK_SIZE, BLOCK_SIZE),
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
