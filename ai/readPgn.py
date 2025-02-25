import chess
import chess.pgn as pgn

pgn_file_path = 'temp.pgn'


with open(pgn_file_path, 'r') as pgn_file:
    for line in pgn_file:
        if line[0] == '1':
            print(line.strip())
            print()
