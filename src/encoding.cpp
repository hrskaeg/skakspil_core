#include "encoding.h"
#include <cmath>
#include <stdexcept>

namespace Encoding {

// ============================================================================
// BOARD TENSOR
// ============================================================================

// Helper: set a value at [plane][row][col] in the flat array
static inline void setPlane(std::vector<float>& tensor, int plane, int row, int col, float val) {
    tensor[plane * 64 + row * 8 + col] = val;
}

std::vector<float> boardToTensor(const Board& board, Color turn) {
    // Start with all zeros
    std::vector<float> tensor(TENSOR_SIZE, 0.0f);

    // --- Planes 0-11: Piece positions ---
    // Each piece type gets its own plane, one set for white, one for black.
    // If there's a white pawn on e2 (row=1, col=4), then tensor[0][1][4] = 1.0
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            const Piece& p = board.getPiece(r, c);
            if (p.type == Piecetype::None) continue;

            // Map piece type to plane offset (0-5)
            int typeOffset = -1;
            switch (p.type) {
                case Piecetype::Pawn:   typeOffset = 0; break;
                case Piecetype::Rook:   typeOffset = 1; break;
                case Piecetype::Knight: typeOffset = 2; break;
                case Piecetype::Bishop: typeOffset = 3; break;
                case Piecetype::Queen:  typeOffset = 4; break;
                case Piecetype::King:   typeOffset = 5; break;
                default: break;
            }
            if (typeOffset < 0) continue;

            // White pieces: planes 0-5, Black pieces: planes 6-11
            int plane = (p.color == Color::White) ? typeOffset : typeOffset + 6;
            setPlane(tensor, plane, r, c, 1.0f);
        }
    }

    // --- Plane 12: Side to move ---
    // Entire plane is 1.0 if white to move, 0.0 if black
    if (turn == Color::White) {
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                setPlane(tensor, 12, r, c, 1.0f);
    }

    // --- Planes 13-16: Castling rights ---
    // We infer castling rights from hasMoved flags, same as your engine does.
    // Plane 13: white kingside  (king on e1 + rook on h1, neither moved)
    // Plane 14: white queenside (king on e1 + rook on a1, neither moved)
    // Plane 15: black kingside  (king on e8 + rook on h8, neither moved)
    // Plane 16: black queenside (king on e8 + rook on a8, neither moved)

    auto canCastle = [&](int kingRow, int kingCol, int rookCol) -> bool {
        const Piece& king = board.getPiece(kingRow, kingCol);
        const Piece& rook = board.getPiece(kingRow, rookCol);
        return king.type == Piecetype::King && !king.hasMoved &&
               rook.type == Piecetype::Rook && !rook.hasMoved;
    };

    // Fill entire plane with 1 if that castling right exists
    bool castleRights[4] = {
        canCastle(0, 4, 7),  // white kingside
        canCastle(0, 4, 0),  // white queenside
        canCastle(7, 4, 7),  // black kingside
        canCastle(7, 4, 0),  // black queenside
    };
    for (int i = 0; i < 4; i++) {
        if (castleRights[i]) {
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    setPlane(tensor, 13 + i, r, c, 1.0f);
        }
    }

    // --- Plane 17: En passant target ---
    Position ep = board.getEnPassantTarget();
    if (ep.row >= 0 && ep.row < 8 && ep.col >= 0 && ep.col < 8) {
        setPlane(tensor, 17, ep.row, ep.col, 1.0f);
    }

    // --- Plane 18: Constant 1s (bias plane) ---
    for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            setPlane(tensor, 18, r, c, 1.0f);

    return tensor;
}


// ============================================================================
// MOVE ENCODING
//
// The 73 move types per source square:
//
//  Indices 0-55:  "Queen-style" moves (sliding in 8 directions, 1-7 distance)
//                  direction * 7 + (distance - 1)
//                  Directions: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
//
//  Indices 56-63: Knight moves (8 possible L-shapes)
//                  Ordered by (dRow, dCol) pairs
//
//  Indices 64-72: Underpromotions (pawn reaches last rank, promotes to non-queen)
//                  direction * 3 + piece
//                  direction: 0=straight, 1=capture-left, 2=capture-right
//                  piece: 0=knight, 1=bishop, 2=rook
//                  (Queen promotion uses the normal queen-style move instead)
// ============================================================================

// The 8 directions a queen/rook/bishop can move: {dRow, dCol}
static const int QUEEN_DIRS[8][2] = {
    { 1,  0}, // N
    { 1,  1}, // NE
    { 0,  1}, // E
    {-1,  1}, // SE
    {-1,  0}, // S
    {-1, -1}, // SW
    { 0, -1}, // W
    { 1, -1}, // NW
};

// The 8 knight jumps: {dRow, dCol}
static const int KNIGHT_MOVES[8][2] = {
    { 2,  1}, { 2, -1},
    {-2,  1}, {-2, -1},
    { 1,  2}, { 1, -2},
    {-1,  2}, {-1, -2},
};

int moveToIndex(const Move& move, Piecetype promotionType) {
    int fromRow = move.from.row, fromCol = move.from.col;
    int toRow   = move.to.row,   toCol   = move.to.col;
    int dRow = toRow - fromRow;
    int dCol = toCol - fromCol;

    int moveType = -1;

    // --- Check underpromotions first ---
    // A promotion to knight, bishop, or rook gets its own encoding.
    // Queen promotion (or no promotion specified) falls through to queen-style.
    if (promotionType == Piecetype::Knight ||
        promotionType == Piecetype::Bishop ||
        promotionType == Piecetype::Rook) {

        // Direction: which way is the pawn going?
        int dir;
        if (dCol == 0)       dir = 0;  // straight
        else if (dCol == -1) dir = 1;  // capture left
        else                 dir = 2;  // capture right

        int piece;
        if (promotionType == Piecetype::Knight)      piece = 0;
        else if (promotionType == Piecetype::Bishop)  piece = 1;
        else                                          piece = 2; // Rook

        moveType = 64 + dir * 3 + piece;
    }

    // --- Check knight moves ---
    if (moveType < 0) {
        for (int i = 0; i < 8; i++) {
            if (dRow == KNIGHT_MOVES[i][0] && dCol == KNIGHT_MOVES[i][1]) {
                moveType = 56 + i;
                break;
            }
        }
    }

    // --- Queen-style moves (also covers queen promotions) ---
    if (moveType < 0) {
        // Find which direction this move goes
        int stepRow = (dRow > 0) ? 1 : (dRow < 0) ? -1 : 0;
        int stepCol = (dCol > 0) ? 1 : (dCol < 0) ? -1 : 0;
        int distance = std::max(std::abs(dRow), std::abs(dCol));

        for (int d = 0; d < 8; d++) {
            if (QUEEN_DIRS[d][0] == stepRow && QUEEN_DIRS[d][1] == stepCol) {
                moveType = d * 7 + (distance - 1);
                break;
            }
        }
    }

    if (moveType < 0) {
        throw std::runtime_error("Could not encode move");
    }

    // Final index: source square * 73 + move type
    return (fromRow * 8 + fromCol) * NUM_MOVE_TYPES + moveType;
}


Move indexToMove(int index, Piecetype& outPromotionType) {
    outPromotionType = Piecetype::None;

    // Decompose: which source square, which move type?
    int sourceSquare = index / NUM_MOVE_TYPES;
    int moveType     = index % NUM_MOVE_TYPES;

    int fromRow = sourceSquare / 8;
    int fromCol = sourceSquare % 8;
    int toRow, toCol;

    if (moveType < 56) {
        // Queen-style move
        int direction = moveType / 7;
        int distance  = (moveType % 7) + 1;
        toRow = fromRow + QUEEN_DIRS[direction][0] * distance;
        toCol = fromCol + QUEEN_DIRS[direction][1] * distance;

    } else if (moveType < 64) {
        // Knight move
        int knightIdx = moveType - 56;
        toRow = fromRow + KNIGHT_MOVES[knightIdx][0];
        toCol = fromCol + KNIGHT_MOVES[knightIdx][1];

    } else {
        // Underpromotion (indices 64-72)
        int promoIdx = moveType - 64;
        int dir   = promoIdx / 3;  // 0=straight, 1=left-capture, 2=right-capture
        int piece = promoIdx % 3;  // 0=knight, 1=bishop, 2=rook

        // Pawn direction: figure out from the row.
        // If pawn is on row 6, it's white going to row 7.
        // If pawn is on row 1, it's black going to row 0.
        int pawnDir = (fromRow == 6) ? 1 : -1;
        toRow = fromRow + pawnDir;

        if (dir == 0)      toCol = fromCol;
        else if (dir == 1) toCol = fromCol - 1;
        else               toCol = fromCol + 1;

        switch (piece) {
            case 0: outPromotionType = Piecetype::Knight; break;
            case 1: outPromotionType = Piecetype::Bishop; break;
            case 2: outPromotionType = Piecetype::Rook;   break;
        }
    }

    Move m;
    m.from = {fromRow, fromCol};
    m.to   = {toRow, toCol};
    return m;
}

} // namespace Encoding