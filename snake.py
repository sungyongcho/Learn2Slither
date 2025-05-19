import argparse

from agent import Agent
from constants import SPEED
from environment import Environment
from game_interface import PygameInterface
from helper import plot


def train(visualize: bool = True) -> None:
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    board = Environment()
    interface = PygameInterface(board) if visualize else None

    while True:
        state_old = agent.get_state(board)
        final_move = agent.get_action(state_old)

        reward, done, score = board.step(final_move)
        state_new = agent.get_state(board)

        agent.train_short_memory(
            state_old,
            final_move,
            reward,
            state_new,
            done,
        )
        agent.remember(state_old, final_move, reward, state_new, done)

        if visualize:
            interface._handle_pygame_events()
            interface._render()
            interface.clock.tick(SPEED)

        if done:
            board.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.num_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
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
