import chess
from chess import Move, Board
from Agents import RandomAgent
import time

class ChessUtils:
    WHITE = 0
    BLACK = 1

class Agent():
    def get_move(board_state: Board) -> Move:
        pass

    def log_info(board_state: Board):
        pass

class ChessGame():
    def __init__(self, black_agent = RandomAgent(), white_agent = RandomAgent()):
        self.board = chess.Board()
        self.black = black_agent
        self.white = white_agent
    
    def get_board(self) -> Board:
        return self.board
    
    def turn(self) -> int:
        return ChessUtils.WHITE if self.board.turn else ChessUtils.BLACK

    def set_players(self, white: Agent, black: Agent):
        self.white = white
        self.black = black

    def print_state(self):
        print("\nTurn:", "WHITE" if self.board.turn else "BLACK")
        print(self.board,end="\n")

    def get_next_player(self) -> Agent:
        return self.white if self.turn() == ChessUtils.WHITE else self.black


def main():
    game = ChessGame()
    board_state = game.get_board()
 
    while not board_state.is_game_over():

        player = game.get_next_player()
        next_move = player.get_move(board_state)
        board_state.push(next_move)

        game.print_state()

    print("\n",board_state.result())

if __name__ == "__main__":
    main()