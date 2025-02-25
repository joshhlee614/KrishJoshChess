from chess import Move, Board
import ai.ai as ai
import Evaluators as ev
import random
import functools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager


avg = []
timer_on = False

def timer(func):
    def wrapper(*args, **kwargs):
        if not timer_on:
            return func(*args, **kwargs)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        fn_time = time.perf_counter() - start
        print(f"Time taken in seconds : {fn_time:0.4f}")
        avg.append(fn_time)
        print(f"Average time per move : {sum(avg)/len(avg):0.4f}")
        return result
    return wrapper

class Agent():
    def __init__(self) -> None:
        self.evaluator = ev.nn_eval
        self.depth = 4

    def get_move(self, board_state: Board) -> Move:
        pass

    def log_info(self, board_state: Board):
        pass

class MinimaxAgent(Agent):
    def get_move(self, board_state: Board) -> Move:
        def minimax(board, depth, Player): #Player will be a value true if it is the player false if it is the oponenet
            # return evaluation if depth is zero or if the board state is gameover
            if depth == 0 or board.is_game_over():
                boardfen =  board.fen()
                return self.evaluator(board)
            if board_state.turn == Player:
                max_eval = float("-inf")
                for move in list(board.legal_moves):
                    child_board = board.copy()
                    child_board.push(move)
                    evaluation = minimax(child_board , depth - 1, False)
                    max_eval = max(max_eval, evaluation)
                return max_eval
            else:
                min_eval = float("inf")
                for move in list(board.legal_moves):
                    child_board = board.copy()
                    child_board.push(move)
                    evaluation = minimax(child_board , depth - 1, True)
                    min_eval = min(min_eval, evaluation)
                return min_eval

        best_move = None;
        best_score = float('-inf')

        for move in list(board_state.legal_moves):
            test_board = board_state.copy()
            test_board.push(move)
            score = minimax(test_board, self.depth, board_state.turn)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move;


    def log_info(self, board_state: Board):
        pass


class AlphaBetaAgent(Agent):
    # @timer
    def get_move(self, board_state: Board) -> Move:
        def minimax(board, depth, alpha, beta, Player):
            if depth == 0 or board.is_game_over():  # Add your game_over function
                return self.evaluator(board)
            
            if Player:
                max_eval = float("-inf")
                for move in board.legal_moves:
                    board.push(move)
                    evaluation = minimax(board, depth - 1, alpha, beta, False)
                    board.pop()
                    max_eval = max(max_eval, evaluation)
                    alpha = max(alpha, evaluation)
                    if beta <= alpha:
                        break
                return max_eval
            else:
                min_eval = float("inf")
                for move in board.legal_moves:
                    board.push(move)
                    evaluation = minimax(board, depth - 1, alpha, beta, True)
                    board.pop()
                    min_eval = min(min_eval, evaluation)
                    beta = min(beta, evaluation)
                    if beta <= alpha:
                        break
                return min_eval

        best_move = None;
        best_score = float('-inf')

        for move in list(board_state.legal_moves):
            test_board = board_state.copy()
            test_board.push(move)
            score = minimax(test_board, self.depth, float("-inf"), float("inf"), board_state.turn)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move;


    def log_info(self, board_state: Board):
        pass


class TranspositionAgent(Agent):
    def __init__(self):
        super().__init__()
        self.transposition_table = {}

    def minimax(self, board, depth, alpha, beta, Player):
        board_hash = board.fen()
        if board_hash in self.transposition_table:
            return self.transposition_table[board_hash]

        if depth == 0 or board.is_game_over():
            score = self.evaluator(board)
            self.transposition_table[board_hash] = (score, None)
            return (score, None)

        best_move = None
        if Player:
            max_eval = float("-inf")
            for move in board.legal_moves:
                board.push(move)
                evaluation, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
        else:
            max_eval = float("inf")
            for move in board.legal_moves:
                board.push(move)
                evaluation, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                if evaluation < max_eval:
                    max_eval = evaluation
                    best_move = move
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break

        self.transposition_table[board_hash] = (max_eval, best_move)
        return (max_eval, best_move)
    
    def get_move(self, board_state: Board) -> Move:
        self.transposition_table = {}
        best_score, best_move = self.minimax(board_state, self.depth, float("-inf"), float("inf"), board_state.turn)
        return best_move


class SortedTranspositionAgent(Agent):
    def __init__(self):
        super().__init__()
        self.transposition_table = {}
        self.things_done = 0

    # @functools.lru_cache(maxsize=100000)
    def move_sorter(self, board):
        scores = {}
        for move in board.legal_moves:
            if board.is_capture(move):
                scores[move] = 10
            elif board.gives_check(move):
                scores[move] = 9
            else:
                scores[move] = 1
        return scores

    # @functools.lru_cache(maxsize=100000)
    def minimax(self, board, depth, alpha, beta, maximizing):
        # self.things_done += 1
        board_hash = str(board.fen())
        if board_hash in self.transposition_table:
            return self.transposition_table[board_hash]

        if depth == 0 or board.is_game_over():
            score = self.evaluator(board)
            self.transposition_table[board_hash] = (score, None)
            return (score, None)

        best_move = None
        moves = list(board.legal_moves)
        move_scores = self.move_sorter(board)
        moves.sort(key=lambda move: move_scores.get(move, 0), reverse=True)

        if maximizing:
            min_eval = float("-inf")
            for move in moves:
                board.push(move)
                evaluation, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                if evaluation > min_eval:
                    min_eval = evaluation
                    best_move = move
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
        else:
            min_eval = float("inf")
            for move in moves:
                board.push(move)
                evaluation, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break

        self.transposition_table[board_hash] = (min_eval, best_move)
        return (min_eval, best_move)

    @timer
    # @functools.lru_cache(maxsize=100000)
    def get_move(self, board_state: Board) -> Move:
        self.transposition_table = {}
        best_move = None
        # curr = self.things_done
        _, best_move = self.minimax(board_state, self.depth, float("-inf"), float("inf"), board_state.turn)
        # print(self.things_done - 0, "computations done")
        return best_move


# class MultiProcessAgent(Agent):
#     def __init__(self):
#         super().__init__()
#         # self.transposition_table = {}
#         manager = Manager()
#         self.transposition_table = manager.dict()

#         # self.things_done = 0

#     def move_sorter(self, board):
#         scores = {}
#         for move in board.legal_moves:
#             if board.is_capture(move):
#                 scores[move] = 10
#             elif board.gives_check(move):
#                 scores[move] = 9
#             else:
#                 scores[move] = 1
#         return scores

#     def minimax(self, board, depth, alpha, beta, maximizing):
#         board_hash = str(board.fen())
#         if board_hash in self.transposition_table:
#             return self.transposition_table[board_hash]
        
#         if depth == 0 or board.is_game_over():
#             score = self.evaluator(board)
#             self.transposition_table[board_hash] = (score, None)
#             return (score, None)

#         best_move = None
#         moves = list(board.legal_moves)
#         move_scores = self.move_sorter(board)
#         moves.sort(key=lambda move: move_scores.get(move, 0), reverse=True)

#         min_eval = float("-inf") if maximizing else float("inf")

#         with ProcessPoolExecutor() as executor:
#         # Create future to move mapping
#             future_to_move = {}
#             for move in moves:
#                 new_board = board.copy()
#                 new_board.push(move)
#                 future = executor.submit(self.minimax, new_board, depth - 1, alpha, beta, not maximizing)
#                 future_to_move[future] = move
            
#             for future in as_completed(future_to_move):
#                 move = future_to_move[future]
#                 try:
#                     evaluation, _ = future.result()
#                 except Exception as e:
#                     print(f"Exception: {e}")
#                     continue
                
#                 if maximizing:
#                     if evaluation > min_eval:
#                         min_eval = evaluation
#                         best_move = move
#                     alpha = max(alpha, evaluation)
#                 else:
#                     if evaluation < min_eval:
#                         min_eval = evaluation
#                         best_move = move
#                     beta = min(beta, evaluation)
                    
#                 if beta <= alpha:
#                     break

#         self.transposition_table[board_hash] = (min_eval, best_move)
#         return (min_eval, best_move)

#     @timer
#     def get_move(self, board_state: Board) -> Move:
#         self.transposition_table = {}
#         best_move = None
#         _, best_move = self.minimax(board_state, self.depth, float("-inf"), float("inf"), board_state.turn)
#         return best_move


class FirstAgent(Agent):
    def get_move(self, board_state: Board) -> Move:
        return list(board_state.legal_moves)[0]

    def log_info(self, board_state: Board):
        pass


class RandomAgent(Agent):
    # @timer
    def get_move(self, board_state: Board) -> Move:
        options = list(board_state.legal_moves)
        return random.choice(options)

    def log_info(self, board_state: Board):
        pass


class MouseAgent(Agent): 
    def __init__(self, ui):
        self.ui = ui

        if not self.ui:
            raise Exception("MouseAgent requires a UI to be passed in")

    def get_move(self, board_state: Board) -> Move:
        return self.ui.get_user_request()

    def log_info(self, board_state: Board):
        pass


class StockfishAgent(Agent):
    def __init__(self) -> None:
        self.skill_level = 4
        # self.ai = ai
        ai.stockfish.set_skill_level(self.skill_level)

    def get_move(self, board_state: Board) -> Move:
        best_move = ai.getMove(board_state.fen())
        best_move = Move.from_uci(best_move)
        return best_move
    
    def set_skill(self, skill):
        if skill < 20:
            self.skill_level = skill
            # self.ai.stockfish.set_skill_level(skill)

    def log_info(self, board_state: Board):
        pass
