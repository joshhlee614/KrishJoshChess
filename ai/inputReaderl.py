import chess
import chess.pgn as pgn
import torch
import torch.nn as nn
import ai
import csv
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ai.neural_net import SimpleNN
import functools
import os



# Define your input size
input_size = 189



def neural_net_input(board):
    piece_index = {"p": 5, "n": 0, "b":1, "r":2, "q":3, "k": 4}
    inputs = [0] * 189
    current_player = board.turn
    for i in range (8):
        for j in range (8):
            square = chess.SQUARES[chess.square(i,j)]
            piece = str(board.piece_at(square))
            color = 1
            if piece == piece.lower():
                color *= -1
            piece = piece.lower()
            rank = j
            if color == -1:
                rank = 7 - j
            file = i
            if file > 3:
                file = 7 - i
            if piece != "none":
                index = len(inputs) - 1 - piece_index[piece]
                inputs[index] += color
                if piece != "p":
                    inputs[piece_index[piece] * 32 + rank * 4 + file] += color
                else:
                    inputs[piece_index[piece] * 32 + (rank - 1) * 4 + file] += color
    return inputs


# def saveNNInputs():
# This function save the list of pgns into nn inputs for the game to understand
def main():
    # Initialize variables
    input_params = 0
    pos_list = []
    evaluations_file = "pos.txt"
    fileFound = os.path.exists(evaluations_file)
    permissions = "r+" if fileFound else "w+"
    done = False
    
    # Open the PGN file
    with open("ai/temp.pgn") as pgn_file:
        # Loop through each game in the PGN file
        while True:
            # Read a game from the PGN file
            game = pgn.read_game(pgn_file)
            if game is None:
                break  # No more games to read
            
            # Initialize a chess board for the current game
            board = chess.Board()
            
            # Iterate through the moves of the mainline of the game
            for number, move in enumerate(game.mainline_moves()):
                board.push(move)
                
                # Extract features from the board
                net_inputs = neural_net_input(board)
                
                # Write features to file if it doesn't exist
                if not fileFound:
                    if input_params % 1000 == 0:
                        print("evaluating position from stockfish", input_params)
                    with open(evaluations_file, "a") as f:
                        print(str(net_inputs), file=f)
                
                # Increment input_params counter
                input_params += 1
            
            # Check if done flag is set
            if done:
                break


def load_checkpoint():
    print("testing")
    model = SimpleNN() 
    checkpoint = torch.load("model_checkpoint.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    board = chess.Board()
    board.set_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    

    net_inputs = neural_net_input(board)  # Assuming neural_net_input function returns the input features for the neural network
    inputs = torch.tensor(net_inputs, dtype=torch.float)

    with torch.no_grad():
        output = model(inputs)
        value = output.item()  # Extract the scalar value from the tensor
        print(board)
        print("Expected value:", ai.getPositionEval(board.fen()))
        print("Predicted value:", value)
        


if __name__ == "__main__":
    load_checkpoint()
    # load_checkpoint()