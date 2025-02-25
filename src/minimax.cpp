// src/minimax.cpp
#include <limits>
#include <unordered_map>
#include <vector>
#include <string>
// src/board.hpp
#ifndef MINIMAX_BOARD_HPP
#define MINIMAX_BOARD_HPP

#include "bitboard.hpp"
#include <string>
#include <vector>

// A very simplified board representation.
// In a complete engine, you’d track piece placement, side to move, etc.
class Board {
public:
    // For illustration, we use a single Bitboard to represent occupancy.
    // In practice, you’d likely have one Bitboard per piece type and color.
    libchess::Bitboard occupancy;

    Board() {
        // Initialize board (e.g., set up starting positions)
        // This is just a placeholder.
        occupancy = libchess::Bitboard(/* some initial mask */);
    }

    // Return a FEN-like string (you may wish to implement a proper FEN generator)
    std::string fen() const {
        // For simplicity, return the hexadecimal value of the occupancy.
        char buffer[20];
        snprintf(buffer, sizeof(buffer), "%llx", occupancy.value());
        return std::string(buffer);
    }

    // Dummy game-over check (update with real rules)
    bool is_game_over() const {
        return false;
    }

    // Dummy evaluator: count bits (replace with your evaluation logic)
    int evaluate() const {
        return occupancy.count();
    }

    // Dummy move generation: returns a vector of moves (here, moves can be defined as integers)
    // You’d normally define a Move type. For simplicity, we use int.
    std::vector<int> generate_moves() const {
        // Return some dummy moves
        return {0, 1, 2};
    }

    // Dummy move application: update the occupancy (replace with actual move logic)
    void make_move(int move) {
        // For demonstration, toggle a bit; in a real engine, update piece positions.
        occupancy ^= libchess::Bitboard(1ULL << move);
    }

    // Dummy move undo: reverse the move
    void undo_move(int move) {
        // For demonstration, toggle the same bit again.
        occupancy ^= libchess::Bitboard(1ULL << move);
    }
};

#endif  // MINIMAX_BOARD_HPP

// Include libchess headers (update paths as needed)
#include "libchess/move.hpp"


// Include pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace libchess;  // Assuming libchess uses this namespace

// Create a transposition table keyed by board FEN (or a board hash, if available)
std::unordered_map<std::string, std::pair<int, Move>> transposition_table;

// The C++ minimax function, modeled after your Python version
std::pair<int, int> minimax(Board &board, int depth, int alpha, int beta, bool maximizing) {
    std::string board_key = board.fen();
    if (transposition_table.find(board_key) != transposition_table.end()) {
        return transposition_table[board_key];
    }

    if (depth == 0 || board.is_game_over()) {
        int score = board.evaluate();
        transposition_table[board_key] = {score, libchess::Move()};
        return {score, libchess::Move()};

    }

    int best_move = -1; // Now using an integer instead of libchess::Move

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
        transposition_table[board.fen()] = std::make_pair(max_eval, libchess::Move(best_move));
        return {max_eval, best_move};
    } else {
        int min_eval = std::numeric_limits<int>::max();
        auto moves = board.generate_moves();
        for (const auto &move : moves) {
            board.make_move(move);
            auto [evaluation, _] = minimax(board, depth - 1, alpha, beta, true);
            board.undo_move(move);
            if (evaluation < min_eval) {
                min_eval = evaluation;
                best_move = move;
            }
            beta = std::min(beta, evaluation);
            if (beta <= alpha)
                break;
        }
        transposition_table[board_key] = {min_eval, best_move};
        return {min_eval, best_move};
    }
}

PYBIND11_MODULE(chess_minimax, m) {
    m.doc() = "C++ minimax algorithm using libchess and pybind11";

    // Expose the libchess Board class
    py::class_<Board>(m, "Board")
        .def(py::init<>())  // Default constructor, which sets up the initial board
        .def("fen", &Board::fen)
        .def("make_move", &Board::make_move)
        .def("undo_move", &Board::undo_move)
        .def("is_game_over", &Board::is_game_over)
        .def("generate_moves", &Board::generate_moves)
        .def("evaluate", &Board::evaluate);

    // Expose the minimax function
    m.def("minimax", [](Board &board, int depth, int alpha, int beta, bool maximizing) {
        auto result = minimax(board, depth, alpha, beta, maximizing);
        
        // Convert move from int to a UCI string for Python compatibility
        std::string best_move_uci = (result.second == -1) ? "" : std::to_string(result.second);

        return py::make_tuple(result.first, best_move_uci);
    }, "Run the minimax algorithm on a given board");
}
