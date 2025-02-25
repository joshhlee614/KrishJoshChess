#include <limits>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdio>

// --- Begin Board Definition ---
#ifndef MINIMAX_BOARD_HPP
#define MINIMAX_BOARD_HPP

#include "bitboard.hpp"
#include "move.hpp"
#include "square.hpp"
#include "piece.hpp"
#include <string>
#include <vector>

class Board {
public:
    // For illustration, we use a single Bitboard to represent occupancy.
    libchess::Bitboard occupancy;

    Board() {
        // Initialize board (for demonstration, set occupancy to 0)
        occupancy = libchess::Bitboard(0ULL);
    }

    std::string fen() const {
        char buffer[20];
        snprintf(buffer, sizeof(buffer), "%llx", occupancy.value());
        return std::string(buffer);
    }

    bool is_game_over() const {
        return false;
    }

    int evaluate() const {
        return occupancy.count();
    }

    // Now returns a vector of libchess::Move
    std::vector<libchess::Move> generate_moves() const {
        return {
            libchess::Move(libchess::Normal, squares::A2, squares::A3, Piece::Pawn),
            libchess::Move(libchess::Normal, squares::B2, squares::B3, Piece::Pawn),
            libchess::Move(libchess::Normal, squares::C2, squares::C3, Piece::Pawn)
        };
    }

    void make_move(libchess::Move move) {
        // For demonstration, toggle a bit at the 'from' square.
        occupancy ^= libchess::Bitboard(1ULL << static_cast<int>(move.from()));
    }

    void undo_move(libchess::Move move) {
        occupancy ^= libchess::Bitboard(1ULL << static_cast<int>(move.from()));
    }
};

#endif  // MINIMAX_BOARD_HPP
// --- End Board Definition ---

#include "libchess/move.hpp"

// Include pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace libchess;  // Now Move is defined

// Global transposition table: key is board FEN, value is pair of (evaluation, best move)
std::unordered_map<std::string, std::pair<int, Move>> transposition_table;

// The minimax function using alpha-beta pruning, returns a pair (score, best move)
std::pair<int, Move> minimax(Board &board, int depth, int alpha, int beta, bool maximizing) {
    std::string board_key = board.fen();
    if (transposition_table.find(board_key) != transposition_table.end()) {
        return transposition_table[board_key];
    }

    if (depth == 0 || board.is_game_over()) {
        int score = board.evaluate();
        transposition_table[board_key] = std::make_pair(score, Move());
        return std::make_pair(score, Move());
    }

    Move best_move;  // best_move is of type libchess::Move

    if (maximizing) {
        int max_eval = std::numeric_limits<int>::min();
        auto moves = board.generate_moves();
        for (const auto &move : moves) {
            board.make_move(move);
            auto [evaluation, _] = minimax(board, depth - 1, alpha, beta, false);
            board.undo_move(move);
            if (evaluation > max_eval) {
                max_eval = evaluation;
                best_move = move;
            }
            alpha = std::max(alpha, evaluation);
            if (beta <= alpha)
                break;
        }
        transposition_table[board.fen()] = std::make_pair(max_eval, best_move);
        return std::make_pair(max_eval, best_move);
    } else {
        int min_eval = std::numeric_limits<int>::max();
        Move best_move_local;
        auto moves = board.generate_moves();
        for (const auto &move : moves) {
            board.make_move(move);
            auto [evaluation, _] = minimax(board, depth - 1, alpha, beta, true);
            board.undo_move(move);
            if (evaluation < min_eval) {
                min_eval = evaluation;
                best_move_local = move;
            }
            beta = std::min(beta, evaluation);
            if (beta <= alpha)
                break;
        }
        transposition_table[board_key] = std::make_pair(min_eval, best_move_local);
        return std::make_pair(min_eval, best_move_local);
    }
}

PYBIND11_MODULE(chess_minimax, m) {
    m.doc() = "C++ minimax algorithm using libchess and pybind11";

    // Expose the Board class to Python.
    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def("fen", &Board::fen)
        .def("make_move", &Board::make_move)
        .def("undo_move", &Board::undo_move)
        .def("is_game_over", &Board::is_game_over)
        .def("generate_moves", &Board::generate_moves)
        .def("evaluate", &Board::evaluate);

    // Expose the minimax function.
    m.def("minimax", [](Board &board, int depth, int alpha, int beta, bool maximizing) {
        auto result = minimax(board, depth, alpha, beta, maximizing);
        // Convert the Move to a string via its to_string() method.
        std::string best_move_uci = (result.second == Move()) ? "" : static_cast<std::string>(result.second);
        return py::make_tuple(result.first, best_move_uci);
    }, "Run the minimax algorithm on a given board");
}
