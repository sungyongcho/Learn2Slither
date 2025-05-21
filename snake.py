import argparse

import matplotlib.pyplot as plt

from agent import Agent
from constants import SPEED
from environment import Environment
from game_interface import PygameInterface
from plot_graph import plot as actual_plot_function


# ------------------------------------------------------------------------
# Core loop (private)
# ------------------------------------------------------------------------
def _play_loop(
    *,
    sessions: int = 0,
    load_path: str | None = None,
    save_path: str | None = None,
    visualize: bool = True,  # For Pygame visualization
    learn: bool = True,
    enable_plotting: bool = True,  # For score plotting
) -> None:
    if enable_plotting:
        plt.ion()  # Turn on interactive mode only if plotting
        plot_figure = plt.figure()  # Create a figure for plotting
        plt.show(block=False)  # Show the figure window once, non-blockingly

        plot_scores, plot_mean_scores = [], []
        total_length = 0
    else:  # Ensure these exist even if not used, or handle more carefully
        plot_scores, plot_mean_scores = None, None

    record = 0

    # Agent
    agent_kwargs = {"load_path": load_path}
    if not learn:  # freeze policy
        agent_kwargs.update(initial_epsilon=0.0, min_epsilon=0.0)
    agent = Agent(**agent_kwargs)

    board = Environment()
    gui = (
        PygameInterface(board) if visualize else None
    )  # Pygame GUI based on 'visualize'

    while True:
        # ------------ one frame ----------------------------------------
        state_old = agent.get_state(board)
        action = agent.get_action(state_old)
        reward, done, length = board.step(action)
        state_new = agent.get_state(board)

        if learn:
            agent.train_short_memory(
                state_old,
                action,
                reward,
                state_new,
                done,
            )
            agent.remember(state_old, action, reward, state_new, done)

        # ------------ draw (Pygame visualization) --------------------
        if visualize:
            if gui:  # Ensure gui exists
                gui._handle_pygame_events()
                gui._render(dead=done)
                gui.clock.tick(SPEED)

        # ------------ episode end --------------------------------------
        if done:
            board.reset()
            agent.num_games += 1

            if learn:
                agent.train_long_memory()
                if length > record:
                    record = length
                    if save_path:
                        agent.model.save(save_path)
                    else:
                        print(
                            "Warning: New record, but no --save path provided."
                            + "Model not saved."
                        )

            print(f"Game {agent.num_games}  length {length}  Record {record}")

            # Score plotting based on 'enable_plotting'

            if (
                enable_plotting and plot_figure
            ):  # Check if plotting is on and figure exists
                plot_scores.append(length)
                total_length += length
                plot_mean_scores.append(total_length / agent.num_games)
                # actual_plot_function from import (look top)
                actual_plot_function(plot_scores, plot_mean_scores)

            if sessions and agent.num_games >= sessions:
                print(f"Reached the number of sessions {sessions}")
                break


# ------------------------------------------------------------------------
# Public wrappers
# ------------------------------------------------------------------------
def train(**kwargs) -> None:
    """Train with learning and model saving."""
    _play_loop(learn=True, **kwargs)


def run(**kwargs) -> None:
    """Run inference only (no learning, no saving)."""
    kwargs.pop("save_path", None)
    _play_loop(learn=False, **kwargs)


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session", type=int, default=0, help="Episodes to run (0 = unlimited)."
    )
    parser.add_argument(
        "--load",
        help="Path to .pth file to load (optional)",
    )
    parser.add_argument(
        "--save",
        help="Path to save model when a new record is hit",
    )
    parser.add_argument(
        "--visualize",  # For Pygame visualization
        choices=["true", "false"],
        default="true",
        help="Enable/disable Pygame visualization.",
    )
    parser.add_argument(
        "--plot",  # For score plotting
        choices=["true", "false"],
        default="true",
        help="Enable/disable score plotting. Overridden by --dontlearn.",
    )
    parser.add_argument(
        "--dontlearn",
        action="store_true",
        help="Run without learning, also disables score plotting.",
    )
    args = parser.parse_args()

    # Pygame visualization flag - directly from argument,
    # not affected by --dontlearn
    visualize_flag = args.visualize.lower() == "true"

    # Score plotting flag - can be overridden by --dontlearn
    initial_plot_request = args.plot.lower() == "true"
    enable_plotting_flag = initial_plot_request
    if args.dontlearn:
        # Override score plotting if --dontlearn is set
        enable_plotting_flag = False

    sessions = max(0, args.session)

    # -------- summary ---------------------------------------------------
    print("===== Configuration =====")
    print(f"Sessions       : {sessions or 'unlimited'}")
    print(f"Load path      : {args.load if args.load else '—'}")
    print(
        f"Save path      : {(args.save if args.save else '—') if not args.dontlearn else '— (Learning disabled)'}"
    )

    # Summary for Pygame visualization (simple enabled/disabled)
    print(f"Visualization  : {'Enabled' if visualize_flag else 'Disabled'}")

    # Summary for score plotting (can be overridden)
    plot_summary_message = ""
    if args.dontlearn and initial_plot_request:
        plot_summary_message = "Disabled (overridden by --dontlearn)"
    elif enable_plotting_flag:
        plot_summary_message = "Enabled"
    else:
        plot_summary_message = "Disabled"
    print(f"Plotting       : {plot_summary_message}")

    print(f"Learning       : {'Disabled' if args.dontlearn else 'Enabled'}")
    print("=========================\n")

    common = dict(
        sessions=sessions,
        load_path=args.load,
        save_path=args.save if not args.dontlearn else None,
        visualize=visualize_flag,  # For Pygame visualization
        enable_plotting=enable_plotting_flag,  # For score plotting
    )
    (run if args.dontlearn else train)(**common)


if __name__ == "__main__":
    main()
    main()
    main()
