from __future__ import annotations

import random
from typing import List

import pygame

from constants import BLOCK_SIZE, Colors
from environment import Environment


class PygameInterface:
    """Pygame layer to visualization."""

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

    # def run(self, fps: int = 20) -> None:
    #     while True:
    #         action = self._random_action()  # replace with human/AI policy
    #         reward, game_over, score = self.board.step(action)
    #         self._handle_pygame_events()
    #         if game_over:
    #             self._render(dead=True)  # one last frame in red
    #             pygame.time.wait(800)  # 0.8 s pause
    #             print("Game over | score:", score)
    #             break

    #         self._render()
    #         self.clock.tick(SPEED)
    #     pygame.quit()

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _random_action() -> List[int]:
        return random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def _handle_pygame_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def _render(self, dead: bool = False) -> None:
        bs = BLOCK_SIZE
        self.display.fill(Colors.BLACK)

        body_color_1 = Colors.RED1 if dead else Colors.BLUE1
        body_color_2 = Colors.RED2 if dead else Colors.BLUE2
        head_color_1 = Colors.RED1 if dead else Colors.YELLOW1
        head_color_2 = Colors.RED2 if dead else Colors.YELLOW2

        head, *body = self.board.snake  # unpack once
        # head
        # body segments
        for seg in body:
            pygame.draw.rect(
                self.display,
                body_color_1,
                pygame.Rect(seg.x * bs, seg.y * bs, bs, bs),
            )
            pygame.draw.rect(
                self.display,
                body_color_2,
                pygame.Rect(seg.x * bs + 4, seg.y * bs + 4, bs - 8, bs - 8),
            )

        pygame.draw.rect(
            self.display,
            head_color_1,
            pygame.Rect(head.x * bs, head.y * bs, bs, bs),
        )
        pygame.draw.rect(
            self.display,
            head_color_2,
            pygame.Rect(head.x * bs + 4, head.y * bs + 4, bs - 8, bs - 8),
        )

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

        # ── grid ───────────────────────────
        w_max = self.w_px - 1  # last visible pixel column
        h_max = self.h_px - 1  # last visible pixel row

        # vertical lines
        for x in range(0, self.w_px, bs):
            pygame.draw.line(self.display, Colors.GRAY, (x, 0), (x, h_max))
        pygame.draw.line(
            self.display, Colors.GRAY, (w_max, 0), (w_max, h_max)
        )  # right edge

        # horizontal lines
        for y in range(0, self.h_px, bs):
            pygame.draw.line(self.display, Colors.GRAY, (0, y), (w_max, y))
        pygame.draw.line(
            self.display, Colors.GRAY, (0, h_max), (w_max, h_max)
        )  # bottom edge

        # ── score & flip ───────────────────
        self.display.blit(
            self.font.render(f"Score: {self.board.score}", True, Colors.WHITE),
            (0, 0),
        )
        pygame.display.flip()
