from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass(frozen=True)
class Pos:
    x: int
    y: int


BOARD_WIDTH: int = 20
BOARD_HEIGHT: int = 20

GREEN_APPLE_COUNT: int = 2  # simultaneous green apples
STARVE_FACTOR: int = 100  # frames before starvation per body length

BLOCK_SIZE = 20  # pixels per tile

# RGB colours
WHITE: Tuple[int, int, int] = (255, 255, 255)
RED: Tuple[int, int, int] = (200, 0, 0)
GREEN: Tuple[int, int, int] = (0, 200, 0)
BLUE1: Tuple[int, int, int] = (0, 0, 255)
BLUE2: Tuple[int, int, int] = (0, 100, 255)
BLACK: Tuple[int, int, int] = (0, 0, 0)


class Colors:
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    RED: Tuple[int, int, int] = (200, 0, 0)
    GREEN: Tuple[int, int, int] = (0, 200, 0)
    BLUE1: Tuple[int, int, int] = (0, 0, 255)
    BLUE2: Tuple[int, int, int] = (0, 100, 255)
    BLACK: Tuple[int, int, int] = (0, 0, 0)


MAX_MEMORY: int = 100_000
BATCH_SIZE: int = 1000
LR: float = 0.001
SPEED: int = 20
