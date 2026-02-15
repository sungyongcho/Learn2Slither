"""Main entry-point for training or evaluating the Snake RL agent.

Usage examples
--------------
$ python snake.py --session 100 --visualize true
$ python snake.py --session 100 --visualize false
$ python snake.py --session 100 --plot
$ python snake.py --session 10 --dontlearn --load model/1000.pth
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from config_loader import Config as RunConfig
from config_loader import load_config


@dataclass
class Config:
    sessions: int = 0  # 0 == unlimited
    load_path: Optional[Path] = None
    save_path: Optional[Path] = None
    visualize: bool = True
    learn: bool = True
    plot: bool = False
    step_by_step: bool = False
    config_path: Optional[Path] = None

    @property
    def plotting_enabled(self) -> bool:
        return self.plot  # purely controlled by --plot flag


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        "--session",
        type=int,
        default=0,
        help="Episodes to run (0 = unlimited)",
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
    p.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config (defaults to configs/default.yaml)",
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
        config_path=ns.config,
    )


def play(cfg: Config, run_cfg: RunConfig) -> None:
    import matplotlib.pyplot as plt

    from agent import Agent
    from environment import Environment
    from game_interface import PygameInterface
    from plot_graph import plot as plot_scores

    if cfg.plotting_enabled:
        plt.ion()
        _ = plt.figure()
        scores: list[int] = []
        means: list[float] = []
        total_len: int = 0

    record = 0

    agent_cfg = run_cfg.agent
    if not cfg.learn:
        agent_cfg = replace(agent_cfg, initial_epsilon=0.0, min_epsilon=0.0)

    agent = Agent(
        agent_cfg,
        run_cfg.training,
        load_path=str(cfg.load_path) if cfg.load_path else None,
        step_by_step=cfg.step_by_step,
    )

    start_game = agent.num_games
    target_game = start_game + cfg.sessions if cfg.sessions else None

    if cfg.load_path and cfg.sessions:
        print(
            "[INFO] Resumed at game "
            f"{start_game}. Will play {cfg.sessions} more → "
            f"stop at {target_game}."
        )

    board = Environment(
        reward_cfg=run_cfg.reward,
        env_cfg=run_cfg.env,
        step_by_step=cfg.step_by_step,
    )
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
            gui.clock.tick(run_cfg.gui.speed)

        if done:
            board.reset()
            agent.num_games += 1

            if cfg.learn:
                repeat_times = max(3, length // 10)
                for _ in range(repeat_times):
                    agent.train_long_memory()
                if length > record:
                    record = length
                    if cfg.save_path:
                        agent.save(str(cfg.save_path))
                    else:
                        print(
                            "[WARN] New record but no --save path given "
                            "→ not saved"
                        )

            print(
                f"Game {agent.num_games:<4} Length {length:<4} Record {record}"
            )

            if cfg.plotting_enabled:
                scores.append(length)
                total_len += length
                means.append(total_len / agent.num_games)
                plot_scores(scores, means)

            if target_game is not None and agent.num_games >= target_game:
                print("[INFO] Target sessions reached")
                if cfg.save_path:
                    print(
                        "[INFO] Model saved to "
                        f"{cfg.save_path} (after {agent.num_games} games)"
                    )
                    agent.save(str(cfg.save_path))
                break


def main() -> None:
    cfg = parse_args()

    cfg_path = cfg.config_path or Path("configs/default.yaml")
    run_cfg = load_config(cfg_path)

    print("\n── Configuration ─────────────────────────────────────────────")
    for k, v in cfg.__dict__.items():
        print(f"{k:>15}: {v}")
    print("──────────────────────────────────────────────────────────────\n")

    play(cfg, run_cfg)


if __name__ == "__main__":
    main()
