from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _require_keys(
    d: dict[str, Any], expected: set[str], *, where: str
) -> None:
    unknown = set(d) - expected
    missing = expected - set(d)
    if unknown:
        raise ValueError(
            f"[ConfigError] {where}: unknown keys: {sorted(unknown)}"
        )
    if missing:
        raise ValueError(
            f"[ConfigError] {where}: missing keys: {sorted(missing)}"
        )


def _require_mapping(v: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise ValueError(f"[ConfigError] {where} must be a mapping/object")
    return v


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    max_memory: int
    batch_size: int
    lr: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TrainingConfig":
        _require_keys(d, {"max_memory", "batch_size", "lr"}, where="training")
        return TrainingConfig(
            max_memory=int(d["max_memory"]),
            batch_size=int(d["batch_size"]),
            lr=float(d["lr"]),
        )


@dataclass(slots=True, frozen=True)
class GUIConfig:
    speed: int

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GUIConfig":
        _require_keys(d, {"speed"}, where="gui")
        return GUIConfig(speed=int(d["speed"]))


@dataclass(slots=True, frozen=True)
class EnvConfig:
    starve_factor: int

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EnvConfig":
        _require_keys(d, {"starve_factor"}, where="env")
        return EnvConfig(starve_factor=int(d["starve_factor"]))


@dataclass(slots=True, frozen=True)
class RewardConfig:
    living_step: float
    green_apple: float
    nearest_closer: float
    nearest_further: float
    red_apple: float
    death: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RewardConfig":
        _require_keys(
            d,
            {
                "living_step",
                "green_apple",
                "nearest_closer",
                "nearest_further",
                "red_apple",
                "death",
            },
            where="reward",
        )
        return RewardConfig(
            living_step=float(d["living_step"]),
            green_apple=float(d["green_apple"]),
            nearest_closer=float(d["nearest_closer"]),
            nearest_further=float(d["nearest_further"]),
            red_apple=float(d["red_apple"]),
            death=float(d["death"]),
        )


@dataclass(slots=True, frozen=True)
class AgentConfig:
    input_size: int
    hidden1_size: int
    hidden2_size: int
    output_size: int

    gamma: float
    lr: float

    initial_epsilon: float
    min_epsilon: float
    epsilon_decay: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AgentConfig":
        _require_keys(
            d,
            {
                "input_size",
                "hidden1_size",
                "hidden2_size",
                "output_size",
                "gamma",
                "lr",
                "initial_epsilon",
                "min_epsilon",
                "epsilon_decay",
            },
            where="agent",
        )
        return AgentConfig(
            input_size=int(d["input_size"]),
            hidden1_size=int(d["hidden1_size"]),
            hidden2_size=int(d["hidden2_size"]),
            output_size=int(d["output_size"]),
            gamma=float(d["gamma"]),
            lr=float(d["lr"]),
            initial_epsilon=float(d["initial_epsilon"]),
            min_epsilon=float(d["min_epsilon"]),
            epsilon_decay=float(d["epsilon_decay"]),
        )


@dataclass(slots=True, frozen=True)
class Config:
    training: TrainingConfig
    gui: GUIConfig
    env: EnvConfig
    reward: RewardConfig
    agent: AgentConfig

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Config":
        _require_keys(
            d, {"training", "gui", "env", "reward", "agent"}, where="root"
        )

        training = _require_mapping(d["training"], where="training")
        gui = _require_mapping(d["gui"], where="gui")
        env = _require_mapping(d["env"], where="env")
        reward = _require_mapping(d["reward"], where="reward")
        agent = _require_mapping(d["agent"], where="agent")

        return Config(
            training=TrainingConfig.from_dict(training),
            gui=GUIConfig.from_dict(gui),
            env=EnvConfig.from_dict(env),
            reward=RewardConfig.from_dict(reward),
            agent=AgentConfig.from_dict(agent),
        )


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            "[ConfigError] top-level YAML must be a mapping/object"
        )
    return Config.from_dict(raw)
