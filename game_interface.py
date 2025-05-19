from __future__ import annotations

import random
from typing import List

import pygame

from constants import BLOCK_SIZE, SPEED, Colors
from environment import Environment


class PygameInterface:
    """Thin Pygame layer to visualise a :class:`Board`."""

    def __init__(self, board: Environment) -> None:
        pygame.init()
        self.board = board
        self.w_px = board.width * BLOCK_SIZE
        self.h_px = board.height * BLOCK_SIZE
        self.display = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font("arial.ttf", 25)

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self, fps: int = 20) -> None:
        while True:
            action = self._random_action()  # replace with human/AI policy
            reward, game_over, score = self.board.step(action)
            self._handle_pygame_events()
            self._render()
            self.clock.tick(SPEED)
            if game_over:
                print("Game over | score:", score)
                break
        pygame.quit()

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _random_action() -> List[int]:
        return random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def _handle_pygame_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def _render(self) -> None:
        bs = BLOCK_SIZE
        self.display.fill(Colors.BLACK)
        # snake
        for t in self.board.snake:
            pygame.draw.rect(
                self.display,
                Colors.BLUE1,
                pygame.Rect(t.x * bs, t.y * bs, bs, bs),
            )
            pygame.draw.rect(
                self.display,
                Colors.BLUE2,
                pygame.Rect(t.x * bs + 4, t.y * bs + 4, 12, 12),
            )

        # apples
        r = self.board.red_apple
        pygame.draw.rect(
            self.display,
            Colors.RED,
            pygame.Rect(r.x * bs, r.y * bs, bs, bs),
        )
        for g in self.board.green_apples:
            pygame.draw.rect(
                self.display,
                Colors.GREEN,
                pygame.Rect(g.x * bs, g.y * bs, bs, bs),
            )

        # score
        text = self.font.render(
            f"Score: {self.board.score}",
            True,
            Colors.WHITE,
        )
        self.display.blit(text, [0, 0])

        pygame.display.flip()
