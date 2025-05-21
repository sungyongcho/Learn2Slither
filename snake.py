import argparse

from agent import Agent
from constants import SPEED
from environment import Environment
from game_interface import PygameInterface
from plot_graph import plot


# ------------------------------------------------------------------------
# Core loop (private)
# ------------------------------------------------------------------------
def _play_loop(
    *,
    sessions: int = 0,
    load_path: str | None = None,
    save_path: str | None = None,
    visualize: bool = True,
    learn: bool = True,
) -> None:
    plot_scores, plot_mean_scores = [], []
    total_length = record = 0

    # Agent
    agent_kwargs = {"load_path": load_path}
    if not learn:  # freeze policy
        agent_kwargs.update(initial_epsilon=0.0, min_epsilon=0.0)
    agent = Agent(**agent_kwargs)

    board = Environment()
    gui = PygameInterface(board) if visualize else None

    while True:
        # ------------ one frame ----------------------------------------
        state_old = agent.get_state(board)
        action = agent.get_action(state_old)
        reward, done, length = board.step(action)
        state_new = agent.get_state(board)

        if learn:
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

        # ------------ draw ---------------------------------------------
        if visualize:
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
                    agent.model.save(save_path)

            print(f"Game {agent.num_games}  length {length}  Record {record}")

            plot_scores.append(length)
            total_length += length
            plot_mean_scores.append(total_length / agent.num_games)
            plot(plot_scores, plot_mean_scores)

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
    kwargs.pop("save_path", None)  # avoid accidental overwrite
    _play_loop(learn=False, **kwargs)


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session", type=int, default=0, help="Episodes to run (0 = unlimited)."
    )
    parser.add_argument("--load", help="Path to .pth file to load (optional)")
    parser.add_argument("--save", help="Path to save model when a new record is hit")
    parser.add_argument(
        "--visualize",
        choices=["true", "false"],
        default="true",
        help="Enable/disable visualization.",
    )
    parser.add_argument(
        "--dontlearn",
        action="store_true",
        help="Run without learning (pure inference).",
    )
    args = parser.parse_args()

    visualize_flag = args.visualize.lower() == "true"
    sessions = max(0, args.session)

    # -------- summary ---------------------------------------------------
    print("===== Configuration =====")
    print(f"Sessions       : {sessions or 'unlimited'}")
    print(f"Load path      : {args.load}")
    print(f"Save path      : {args.save if not args.dontlearn else '—'}")
    print(f"Visualization  : {'Enabled' if visualize_flag else 'Disabled'}")
    print(f"Learning       : {'Disabled' if args.dontlearn else 'Enabled'}")
    print("=========================\n")

    common = dict(
        sessions=sessions,
        load_path=args.load,
        save_path=args.save,
        visualize=visualize_flag,
    )
    (run if args.dontlearn else train)(**common)


if __name__ == "__main__":
    main()
