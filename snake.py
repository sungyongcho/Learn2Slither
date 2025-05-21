import argparse

from agent import Agent
from constants import SPEED
from environment import Environment
from game_interface import PygameInterface
from helper import plot


def train(visualize: bool = True) -> None:
    plot_scores, plot_mean_scores = [], []
    total_length = record = 0

    agent = Agent()
    board = Environment()
    gui = PygameInterface(board) if visualize else None

    while True:
        # -------- play one frame ----------------------------------------
        state_old = agent.get_state(board)
        action = agent.get_action(state_old)
        reward, done, length = board.step(action)
        state_new = agent.get_state(board)

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
            agent.train_long_memory()

            if length > record:
                record = length
                agent.model.save()

            print(f"Game {agent.num_games}  length {length}  Record {record}")

            plot_scores.append(length)
            total_length += length
            plot_mean_scores.append(total_length / agent.num_games)
            plot(plot_scores, plot_mean_scores)


def main():
    # TODO: without training
    # board = Environment()
    # interface = PygameInterface(board)
    #  interface.run()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize",
        choices=["true", "false"],
        default="true",
        help="Enable or disable visualization (true/false). Default is true.",
    )
    args = parser.parse_args()
    visualize_flag = args.visualize.lower() == "true"
    train(visualize=visualize_flag)


if __name__ == "__main__":
    main()
