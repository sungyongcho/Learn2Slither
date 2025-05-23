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


BOARD_WIDTH: int = 10
BOARD_HEIGHT: int = 10

GREEN_APPLE_COUNT: int = 2  # simultaneous green apples

BLOCK_SIZE = 40  # pixels per tile

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
    GRAY: Tuple[int, int, int] = (128, 128, 128)
    BLUE1: Tuple[int, int, int] = (0, 0, 255)
    BLUE2: Tuple[int, int, int] = (0, 100, 255)
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    YELLOW1: Tuple[int, int, int] = (255, 212, 0)  # outer
    YELLOW2: Tuple[int, int, int] = (255, 240, 120)  # inner highlight
    RED1: Tuple[int, int, int] = (200, 30, 30)  # dark red body
    RED2: Tuple[int, int, int] = (255, 100, 100)  # light red inner square


MAX_MEMORY: int = 100_000
BATCH_SIZE: int = 1000
LR: float = 0.001
SPEED: int = 20


REWARD_LIVING_STEP: int = -0.01
REWARD_GREEN_APPLE: int = 50
REWARD_RED_APPLE: int = -25
REWARD_DEATH: int = -100
STARVE_FACTOR: int = 50  # frames before starvation per body length


@dataclass
class RLConfig:
    input_size: int = 20
    hidden1_size: int = 256
    hidden2_size: int = 128
    output_size: int = 3
    gamma: float = 0.90
    lr: float = LR
    initial_epsilon: float = 1.0
    min_epsilon: float = 0.01
    epsilon_decay: float = 0.99
