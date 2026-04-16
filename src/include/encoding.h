#pragma once
#include "board.h"
#include "move.h"
#include "piece.h"
#include <array>
#include <vector>

// ============================================================================
// ENCODING - Translates chess positions and moves into numbers the neural
//            network can work with.
//
// Two encodings are needed:
//
// 1) BOARD TENSOR  (input to the network)
//    The network sees the board as 19 "planes" of 8x8, stacked.
//    Think of it like 19 separate chessboards, each one answering a yes/no
//    question per square:
//      Planes  0-5:  Is there a white [pawn/rook/knight/bishop/queen/king] here?
//      Planes  6-11: Is there a black [pawn/rook/knight/bishop/queen/king] here?
//      Plane  12:    Is it white's turn? (all 1s or all 0s)
//      Planes 13-14: White can castle [kingside/queenside]? (entire plane = 1 or 0)
//      Planes 15-16: Black can castle [kingside/queenside]?
//      Plane  17:    En passant target square (1 on that square, 0 elsewhere)
//      Plane  18:    Constant plane of 1s (helps the network learn biases)
//
//    Total: 19 * 8 * 8 = 1216 floats
//
// 2) MOVE INDEX  (output of the network's policy head)
//    The network outputs a probability for each POSSIBLE move in chess.
//    AlphaZero encodes moves as (from_square, move_type):
//      - 56 "queen-style" move types: 8 directions x 7 distances
//      -  8 knight move types
//      -  9 underpromotion types: 3 directions x 3 pieces (rook/bishop/knight)
//      Queen promotions use the normal queen-style forward move.
//    
//    Total: 8 * 8 * 73 = 4672 possible move indices
//    Most are illegal in any given position — the network's output gets
//    masked to only legal moves before picking.
// ============================================================================

namespace Encoding {

    constexpr int NUM_PLANES  = 19;
    constexpr int BOARD_SIZE  = 8;
    constexpr int TENSOR_SIZE = NUM_PLANES * BOARD_SIZE * BOARD_SIZE; // 1216

    constexpr int NUM_MOVE_TYPES  = 73;  // 56 queen + 8 knight + 9 underpromo
    constexpr int NUM_MOVE_INDICES = BOARD_SIZE * BOARD_SIZE * NUM_MOVE_TYPES; // 4672

    // --- Board tensor ---
    // Returns a flat vector of 1216 floats representing the position.
    // Layout: plane-major, i.e. [plane][row][col] flattened.
    std::vector<float> boardToTensor(const Board& board, Color turn);

    // --- Move encoding ---
    // Converts a Move struct to an index in [0, 4672).
    // promotionType is only relevant for pawn promotions.
    int moveToIndex(const Move& move, Piecetype promotionType = Piecetype::None);

    // Converts an index back to a Move struct.
    // Also returns what promotion piece it encodes (None if not a promotion).
    Move indexToMove(int index, Piecetype& outPromotionType);
}