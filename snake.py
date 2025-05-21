import argparse

from agent import Agent
from constants import SPEED
from environment import Environment
from game_interface import PygameInterface
from plot_graph import plot


def train(
    sessions: int = 0,
    load_path: str = None,
    save_path: str = None,
    visualize: bool = True,
    learn: bool = True,
) -> None:
    plot_scores, plot_mean_scores = [], []
    total_length = record = 0
    if learn:
        agent = Agent(load_path=load_path)
    else:
        agent = Agent(
            load_path=load_path,
            initial_epsilon=0.00,  # ← disable exploration
            min_epsilon=0.00,  # ← no random moves at all
        )
    board = Environment()
    gui = PygameInterface(board) if visualize else None

    while True:
        # -------- play one frame ----------------------------------------
        state_old = agent.get_state(board)
        action = agent.get_action(state_old)
        reward, done, length = board.step(action)
        state_new = agent.get_state(board)

        if learn:
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)

        # -------- draw ---------------------------------------------------
        if visualize:
            gui._handle_pygame_events()
            gui._render(dead=done)  # ← pass flag
            gui.clock.tick(SPEED)

        # -------- episode finished --------------------------------------
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


def main():
    # TODO: without training
    # board = Environment()
    # interface = PygameInterface(board)
    #  interface.run()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        type=int,
        default=0,
        help="Number of episodes to run (0 = unlimited).",
    )
    parser.add_argument("--load", help="Path to .pth file to load (optional)")
    parser.add_argument("--save", help="Path to save model when a new record is hit")
    parser.add_argument(
        "--visualize",
        choices=["true", "false"],
        default="true",
        help="Enable or disable visualization (true/false). Default is true.",
    )
    parser.add_argument(
        "--dontlearn",
        action="store_true",
        help="Run without learning (no training updates, no model saving).",
    )
    args = parser.parse_args()
    visualize_flag = args.visualize.lower() == "true"
    learn_flag = not args.dontlearn
    sessions = max(0, args.session)  # ensure non-negative
    train(
        sessions=sessions,
        load_path=args.load,
        save_path=args.save,
        visualize=visualize_flag,
        learn=learn_flag,
    )


if __name__ == "__main__":
    main()
