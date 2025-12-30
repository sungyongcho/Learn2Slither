from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass(frozen=True)
class Pos:
    x: int
    y: int


BOARD_WIDTH: int = 10
BOARD_HEIGHT: int = 10

GREEN_APPLE_COUNT: int = 2

BLOCK_SIZE = 40

WHITE: tuple[int, int, int] = (255, 255, 255)
RED: tuple[int, int, int] = (200, 0, 0)
GREEN: tuple[int, int, int] = (0, 200, 0)
BLUE1: tuple[int, int, int] = (0, 0, 255)
BLUE2: tuple[int, int, int] = (0, 100, 255)
BLACK: tuple[int, int, int] = (0, 0, 0)


class Colors:
    WHITE: tuple[int, int, int] = (255, 255, 255)
    RED: tuple[int, int, int] = (200, 0, 0)
    GREEN: tuple[int, int, int] = (0, 200, 0)
    GRAY: tuple[int, int, int] = (128, 128, 128)
    BLUE1: tuple[int, int, int] = (0, 0, 255)
    BLUE2: tuple[int, int, int] = (0, 100, 255)
    BLACK: tuple[int, int, int] = (0, 0, 0)
    YELLOW1: tuple[int, int, int] = (255, 212, 0)
    YELLOW2: tuple[int, int, int] = (255, 240, 120)
    RED1: tuple[int, int, int] = (200, 30, 30)
    RED2: tuple[int, int, int] = (255, 100, 100)
