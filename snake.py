from board import Board
from game_interface import PygameInterface


def main():
    board = Board()
    interface = PygameInterface(board)
    interface.run()


if __name__ == "__main__":
    main()
