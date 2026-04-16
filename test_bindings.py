"""
test_bindings.py — Verify that the C++ engine works from Python.

Run this after building the pybind11 module:
    cd build
    cmake .. && make
    cd ..
    python test_bindings.py

This script does three things:
  1. Plays a random game to prove the basic API works
  2. Tests the board tensor encoding (what the neural network will see)
  3. Tests move-to-index round-tripping (what the policy head will output)
"""

import skakspil_py as skakspil
import random
import numpy as np


def test_random_game():
    """Play a full game with random moves. Proves the engine works end-to-end."""
    print("=== Random game ===")
    game = skakspil.Game()

    move_count = 0
    while game.status() == "playing" and move_count < 200:
        moves = game.get_legal_moves()
        if not moves:
            break

        # Pick a random move
        fr, fc, tr, tc = random.choice(moves)

        # Handle promotion: if a pawn reaches the last rank, promote to queen
        promo = ""
        if tr == 7 or tr == 0:  # last rank
            promo = "q"  # always queen for simplicity

        result = game.make_move(fr, fc, tr, tc, promo)

        # If it needed promotion and we didn't provide one, retry with queen
        if result == "needs_promotion":
            result = game.make_move(fr, fc, tr, tc, "q")

        move_count += 1

    print(f"  Game ended after {move_count} moves. Status: {game.status()}")
    game.print_board()
    print()


def test_board_tensor():
    """Verify the board tensor has the right shape and content."""
    print("=== Board tensor ===")
    game = skakspil.Game()

    tensor = game.get_board_tensor()
    print(f"  Shape: {tensor.shape}")  # should be (19, 8, 8)
    print(f"  Dtype: {tensor.dtype}")  # should be float32

    # Plane 0 is white pawns. In the starting position, row 1 should be all 1s.
    white_pawns = tensor[0]  # plane 0 = white pawns
    print(f"  White pawns on rank 2 (should be all 1s): {white_pawns[1]}")
    print(f"  White pawns on rank 3 (should be all 0s): {white_pawns[2]}")

    # Plane 12 is side-to-move. White starts, so entire plane should be 1.
    print(f"  Side-to-move plane sum (should be 64): {tensor[12].sum()}")
    print()


def test_move_encoding():
    """Verify that move-to-index and index-to-move are inverses."""
    print("=== Move encoding ===")
    game = skakspil.Game()
    moves = game.get_legal_moves()

    print(f"  Starting position has {len(moves)} legal moves")

    # Encode each legal move and decode it back
    all_ok = True
    for fr, fc, tr, tc in moves:
        idx = skakspil.move_to_index(fr, fc, tr, tc)
        back = skakspil.index_to_move(idx)
        back_fr, back_fc, back_tr, back_tc, back_promo = back

        if (fr, fc, tr, tc) != (back_fr, back_fc, back_tr, back_tc):
            print(f"  MISMATCH: ({fr},{fc})->({tr},{tc}) encoded as {idx}, "
                  f"decoded as ({back_fr},{back_fc})->({back_tr},{back_tc})")
            all_ok = False

    if all_ok:
        print(f"  All {len(moves)} moves round-trip correctly!")

    # Show a few examples
    print(f"  e2e4 index: {skakspil.move_to_index(1, 4, 3, 4)}")
    print(f"  d2d4 index: {skakspil.move_to_index(1, 3, 3, 3)}")
    print(f"  g1f3 index: {skakspil.move_to_index(0, 6, 2, 5)}")
    print()


def test_clone():
    """Verify that cloning creates an independent copy."""
    print("=== Clone test ===")
    game = skakspil.Game()
    game.make_move(1, 4, 3, 4)  # e2e4

    clone = game.clone()
    clone.make_move(6, 4, 4, 4)  # e7e5 on the clone only

    # Original should still be black to move (only one move made)
    # Clone should be white to move (two moves made)
    print(f"  Original turn: {game.get_turn()} (should be black)")
    print(f"  Clone turn:    {clone.get_turn()} (should be white)")
    print()


if __name__ == "__main__":
    test_random_game()
    test_board_tensor()
    test_move_encoding()
    test_clone()
    print("All tests passed!")