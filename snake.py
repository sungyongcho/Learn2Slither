from __future__ import annotations

"""Main entry‑point for training or evaluating the Snake RL agent.

Usage examples
--------------
$ python play.py                           # train, visualise, no plot
$ python play.py --visualize false         # head‑less training
$ python play.py --plot                    # show live score plot while training
$ python play.py --dontlearn --load best.pth  # greedy evaluation, no plot
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from agent import Agent
from constants import SPEED, RLConfig
from environment import Environment
from game_interface import PygameInterface
from plot_graph import plot as plot_scores


@dataclass
class Config:
    """Immutable run‑time configuration produced by :func:`parse_args`."""

    sessions: int = 0  # 0 == unlimited
    load_path: Optional[Path] = None
    save_path: Optional[Path] = None
    visualize: bool = True
    learn: bool = True
    plot: bool = False
    step_by_step: bool = False

    @property
    def plotting_enabled(self) -> bool:
        return self.plot  # purely controlled by --plot flag




def parse_args() -> Config:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument(
        "--session", type=int, default=0, help="Episodes to run (0 = unlimited)"
    )
    p.add_argument("--load", type=Path, help="Path to .pth checkpoint to load")
    p.add_argument(
        "--save", type=Path, help="Path to save model when a new record is hit"
    )

    p.add_argument(
        "--visualize",
        choices=["true", "false"],
        default="true",
        help="true/false: enable Pygame visualisation",
    )

    p.add_argument("--plot", action="store_true", help="Show live score plot")

    p.add_argument(
        "--dontlearn",
        action="store_true",
        help="Run in inference-only mode (no training)",
    )
    p.add_argument(
        "--step-by-step",
        dest="step_by_step",
        action="store_true",
        help="Verbose Step-By-Step console output",
    )

    ns = p.parse_args()
    if ns.dontlearn and ns.save:
        p.error(
            "When --dontlearn is set the model cannot be saved "
            "(--save flag is invalid)."
        )

    if ns.dontlearn and ns.plot:
        p.error(
            "When running without learning the score plot cannot be shown "
            "(remove the --plot flag)."
        )

    return Config(
        sessions=max(0, ns.session),
        load_path=ns.load,
        save_path=ns.save if not ns.dontlearn else None,
        visualize=(ns.visualize == "true"),
        learn=not ns.dontlearn,
        plot=ns.plot,
        step_by_step=ns.step_by_step,
    )


def play(cfg: Config) -> None:
    if cfg.plotting_enabled:
        plt.ion()
        _ = plt.figure()
        scores: list[int] = []
        means: list[float] = []
        total_len: int = 0

    record = 0

    agent = Agent(
        RLConfig(
            initial_epsilon=0.0 if not cfg.learn else 1.0,
            min_epsilon=0.0 if not cfg.learn else 0.01,
        ),
        load_path=str(cfg.load_path) if cfg.load_path else None,
        step_by_step=cfg.step_by_step,
    )

    start_game = agent.num_games
    target_game = start_game + cfg.sessions if cfg.sessions else None

    if cfg.load_path and cfg.sessions:
        print(
            f"[INFO] Resumed at game {start_game}. "
            f"Will play {cfg.sessions} more → stop at {target_game}."
        )

    board = Environment(step_by_step=cfg.step_by_step)
    gui = PygameInterface(board) if cfg.visualize else None

    while True:
        state_old = agent.get_state(board)
        action = agent.get_action(state_old)
        reward, done, length = board.step(action)

        if cfg.learn:
            state_new = agent.get_state(board)
            agent.train_short_memory(
                state_old,
                action,
                reward,
                state_new,
                done,
            )
            agent.remember(state_old, action, reward, state_new, done)

        if gui:
            gui._handle_pygame_events()
            gui._render(dead=done)
            gui.clock.tick(SPEED)

        if done:
            board.reset()
            agent.num_games += 1

            if cfg.learn:
                agent.train_long_memory()
                if length > record:
                    record = length
                    if cfg.save_path:
                        agent.save(str(cfg.save_path))
                    else:
                        print("[WARN] New record but no --save path given → not saved")

            print(f"Game {agent.num_games:<4}  Length {length:<4}  Record {record}")

            if cfg.plotting_enabled:
                scores.append(length)
                total_len += length
                means.append(total_len / agent.num_games)
                plot_scores(scores, means)

            if target_game is not None and agent.num_games >= target_game:
                print("[INFO] Target sessions reached")
                if cfg.save_path:
                    print(
                        f"[INFO] Model saved to {cfg.save_path} (after {agent.num_games} games)"
                    )
                    agent.save(str(cfg.save_path))
                break


def main() -> None:
    cfg = parse_args()

    print("\n── Configuration ─────────────────────────────────────────────")
    for k, v in cfg.__dict__.items():
        print(f"{k:>15}: {v}")
    print("──────────────────────────────────────────────────────────────\n")

    play(cfg)


if __name__ == "__main__":
    main()
