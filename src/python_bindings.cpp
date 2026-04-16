// ============================================================================
// PYTHON BINDINGS
//
// This file uses pybind11 to make your C++ chess engine callable from Python.
// After building, you can do:
//
//     import skakspil
//     game = skakspil.Game()
//     print(game.get_turn())        # "white"
//     moves = game.get_legal_moves() # list of (from_r, from_c, to_r, to_c)
//     game.make_move(1, 4, 3, 4)    # e2e4
//
// pybind11 is a header-only C++ library that generates the glue code between
// C++ and Python. Each py::class_ block tells Python "here's a class" and
// each .def() adds a method to it.
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>            // automatic std::vector <-> Python list conversion
#include <pybind11/numpy.h>          // lets us return numpy arrays to Python

#include "game.h"
#include "fen.h"
#include "encoding.h"

namespace py = pybind11;


// ============================================================================
// Wrapper functions
// These adapt your existing C++ API to be more Python-friendly.
// ============================================================================

// Make a move. Returns a string status instead of the enum.
static std::string pyMakeMove(Game& game, int fromRow, int fromCol,
                               int toRow, int toCol,
                               const std::string& promo = "") {
    Move move;
    move.from = {fromRow, fromCol};
    move.to   = {toRow, toCol};

    // Parse promotion string
    Piecetype promoType = Piecetype::None;
    if (promo == "q") promoType = Piecetype::Queen;
    else if (promo == "r") promoType = Piecetype::Rook;
    else if (promo == "b") promoType = Piecetype::Bishop;
    else if (promo == "n") promoType = Piecetype::Knight;

    MoveStatus status = game.tryMove(move, promoType);

    switch (status) {
        case MoveStatus::Success:       return "success";
        case MoveStatus::IllegalMove:   return "illegal";
        case MoveStatus::NotYourTurn:   return "not_your_turn";
        case MoveStatus::MovingEmpty:   return "no_piece";
        case MoveStatus::KingThreatened:return "king_exposed";
        case MoveStatus::CheckMate:     return "checkmate";
        case MoveStatus::StaleMate:     return "stalemate";
        case MoveStatus::nullPromotion: return "needs_promotion";
        default:                        return "error";
    }
}

// Get all legal moves as a list of tuples: [(from_r, from_c, to_r, to_c), ...]
static py::list pyGetLegalMoves(const Game& game) {
    auto moves = game.generateAllMoves(game.getTurn());
    py::list result;
    for (const auto& m : moves) {
        result.append(py::make_tuple(m.from.row, m.from.col,
                                     m.to.row,   m.to.col));
    }
    return result;
}

// Get the board tensor as a numpy array with shape (19, 8, 8)
// This is what the neural network takes as input.
static py::array_t<float> pyGetBoardTensor(const Game& game) {
    auto tensor = Encoding::boardToTensor(game.getBoard(), game.getTurn());

    // Create a numpy array with shape (19, 8, 8) pointing to our data
    // The network will see this as 19 channels of 8x8 "images"
    auto result = py::array_t<float>({Encoding::NUM_PLANES,
                                       Encoding::BOARD_SIZE,
                                       Encoding::BOARD_SIZE});
    auto buf = result.mutable_unchecked<3>();
    for (int p = 0; p < Encoding::NUM_PLANES; p++)
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                buf(p, r, c) = tensor[p * 64 + r * 8 + c];

    return result;
}

// Convert a move to a policy index (0-4671)
static int pyMoveToIndex(int fromRow, int fromCol, int toRow, int toCol,
                          const std::string& promo = "") {
    Move m;
    m.from = {fromRow, fromCol};
    m.to   = {toRow, toCol};

    Piecetype promoType = Piecetype::None;
    if (promo == "q") promoType = Piecetype::Queen;
    else if (promo == "r") promoType = Piecetype::Rook;
    else if (promo == "b") promoType = Piecetype::Bishop;
    else if (promo == "n") promoType = Piecetype::Knight;

    return Encoding::moveToIndex(m, promoType);
}

// Check if the game is over. Returns: "playing", "checkmate", "stalemate"
static std::string pyGameStatus(const Game& game) {
    Color turn = game.getTurn();
    if (game.inCheckmate(turn)) return "checkmate";
    if (game.inStalemate(turn)) return "stalemate";
    return "playing";
}

// Clone the game (needed for MCTS — you explore branches without
// modifying the original game state)
static Game pyCloneGame(const Game& game) {
    return Game(game); // uses copy constructor
}


// ============================================================================
// MODULE DEFINITION
//
// PYBIND11_MODULE(skakspil, m) says:
//   "Create a Python module called 'skakspil', and 'm' is the module object
//    I'll attach things to."
// ============================================================================

PYBIND11_MODULE(skakspil_py, m) {
    m.doc() = "Skakspil chess engine - Python bindings";

    // Expose the Game class
    py::class_<Game>(m, "Game")
        .def(py::init<>())                              // Game() constructor

        // --- Core gameplay ---
        .def("make_move", &pyMakeMove,
             py::arg("from_row"), py::arg("from_col"),
             py::arg("to_row"),   py::arg("to_col"),
             py::arg("promo") = "",
             "Make a move. Returns status string: 'success', 'illegal', 'checkmate', etc.")

        .def("get_legal_moves", &pyGetLegalMoves,
             "Returns list of (from_row, from_col, to_row, to_col) tuples")

        .def("get_turn", [](const Game& g) {
            return g.getTurn() == Color::White ? "white" : "black";
        })

        .def("status", &pyGameStatus,
             "Returns 'playing', 'checkmate', or 'stalemate'")

        .def("in_check", [](const Game& g) {
            return g.inCheck(g.getTurn());
        })

        // --- For the neural network ---
        .def("get_board_tensor", &pyGetBoardTensor,
             "Returns board state as numpy array shape (19, 8, 8)")

        // --- Utilities ---
        .def("load_fen", [](Game& g, const std::string& fen) {
            FEN::loadPosition(g, fen);
        })

        .def("clone", &pyCloneGame,
             "Returns a deep copy of this game state")

        .def("print_board", [](const Game& g) {
            g.getBoard().printBoard();
        });

    // --- Encoding utilities (not tied to a specific game) ---
    m.def("move_to_index", &pyMoveToIndex,
          py::arg("from_row"), py::arg("from_col"),
          py::arg("to_row"),   py::arg("to_col"),
          py::arg("promo") = "",
          "Convert a move to policy head index (0-4671)");

    m.def("index_to_move", [](int index) {
        Piecetype promo;
        Move m = Encoding::indexToMove(index, promo);

        std::string promoStr = "";
        if (promo == Piecetype::Knight) promoStr = "n";
        else if (promo == Piecetype::Bishop) promoStr = "b";
        else if (promo == Piecetype::Rook) promoStr = "r";

        return py::make_tuple(m.from.row, m.from.col,
                              m.to.row, m.to.col, promoStr);
    }, "Convert policy index back to (from_row, from_col, to_row, to_col, promo)");

    // Constants the Python side needs to know
    m.attr("NUM_MOVE_INDICES") = Encoding::NUM_MOVE_INDICES;  // 4672
    m.attr("TENSOR_PLANES")    = Encoding::NUM_PLANES;        // 19
}