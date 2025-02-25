import chess
from stockfish import Stockfish
from sys import platform

stockfish = None
# if platform == "win32":
#     stockfish = Stockfish("C:\\Users\\hursh\Desktop\\ECLAIR\\Chess_Teacher\\ai\\stockfish-windows-x86-64.exe")
# else: 
#     stockfish = Stockfish("./lib/stockfish")
# stockfish = Stockfish("/usr/local/Cellar/stockfish/15.1/bin/stockfish")#Chess_Teacher/ai/stockfish-windows-x86-64.exe
# stockfish = Stockfish("C:\\Users\\hursh\Desktop\\ECLAIR\\Chess_Teacher\\ai\\stockfish-windows-x86-64.exe")
# Instead put the directory of the stockfish location
stockfish = Stockfish("/opt/homebrew/bin/stockfish")
stockfish.set_depth(1)
# stockfish.set_skill_level(20)

def getMove(fen):
    stockfish.set_fen_position(fen)
    return stockfish.get_best_move_time(1)


def getEval(): 
    eval_sf = stockfish.get_evaluation()
    if eval_sf['type'] == 'cp':
        return eval_sf['value'] / 100
    else:
        return f"mate in {eval_sf['value']}"
    
def getPositionEval(fen):
    stockfish.set_fen_position(fen)
    sf_eval = stockfish.get_evaluation()
    if sf_eval['type'] == 'cp':
        return sf_eval['value'] / 100
    else:
        return sf_eval['value'] 

def boardPrint():
    print(stockfish.get_board_visual())





