"""
train.py — Self-play + training loop.

Usage:
    python train.py                          # start fresh
    python train.py --resume model_final.pt  # continue from saved model
    python train.py --resume checkpoint_cycle_5.pt  # continue from checkpoint

This is the AlphaZero cycle (simplified, no MCTS yet):
  1. SELF-PLAY:  The network plays games against itself.
  2. TRAIN:      Train the network on the recorded positions.
  3. REPEAT:     Play more games with the improved network.
"""

import sys
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add the build directory so Python can find skakspil_py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build", "Release"))

import skakspil_py as engine
from model import ChessNet, get_policy_probabilities


# =============================================================================
# CONFIGURATION — tweak these for real training
# =============================================================================

CONFIG = {
    # --- Training loop ---
    "num_cycles":       10,     # total self-play → train cycles (100+ for real training)
    "games_per_cycle":  50,     # games per cycle (200-500 for real training)
    "batch_size":       64,     # training batch size (128-256 for bigger GPU)
    "epochs":           5,      # training epochs per cycle (3-5 is fine)
    "learning_rate":    0.001,  # optimizer learning rate (drop to 0.0001 later)
    "weight_decay":     1e-4,   # L2 regularization (prevents overfitting)
    
    # --- Network size ---
    "num_filters":      64,     # network width (128-256 for stronger play)
    "num_blocks":       5,      # network depth (10-15 for stronger play)
    
    # --- Self-play ---
    "temperature":      1.0,    # move randomness (1.0 = exploratory, 0.1 = greedy)
    "temp_decay":       0.05,   # temperature drops by this much each cycle
    "temp_min":         0.5,    # temperature never goes below this
    "max_moves":        300,    # max moves per game before declaring draw
}


# =============================================================================
# SELF-PLAY
# =============================================================================

def get_legal_move_indices(game):
    """Convert the engine's legal moves to policy indices."""
    moves = game.get_legal_moves()
    indices = []
    for fr, fc, tr, tc in moves:
        is_promo = (tr == 7 or tr == 0)
        if is_promo:
            for p in ["q", "r", "b", "n"]:
                indices.append(engine.move_to_index(fr, fc, tr, tc, p))
        else:
            indices.append(engine.move_to_index(fr, fc, tr, tc))
    return indices, moves


def play_one_game(model, device, temperature=1.0, max_moves=300):
    """
    Play a single self-play game. Returns training data.
    
    Returns:
        List of (board_tensor, policy_target, outcome) tuples.
    """
    game = engine.Game()
    game_history = []
    
    move_count = 0
    while game.status() == "playing" and move_count < max_moves:
        board_tensor = game.get_board_tensor()
        current_turn = game.get_turn()
        
        legal_indices, legal_moves = get_legal_move_indices(game)
        if not legal_indices:
            break
        
        # Ask the network what it thinks
        probs, indices, value = get_policy_probabilities(
            model, board_tensor, legal_indices, device
        )
        
        # Apply temperature
        if temperature > 0:
            adjusted = np.power(probs, 1.0 / temperature)
            adjusted = adjusted / adjusted.sum()
        else:
            adjusted = np.zeros_like(probs)
            adjusted[np.argmax(probs)] = 1.0
        
        # Sample a move
        choice = np.random.choice(len(legal_indices), p=adjusted)
        chosen_index = legal_indices[choice]
        
        # Build policy target
        policy_target = np.zeros(engine.NUM_MOVE_INDICES, dtype=np.float32)
        policy_target[chosen_index] = 1.0
        
        game_history.append((board_tensor.copy(), policy_target, current_turn))
        
        # Play the move
        move_info = engine.index_to_move(chosen_index)
        fr, fc, tr, tc, promo = move_info
        
        result = game.make_move(fr, fc, tr, tc, promo if promo else "")
        if result == "needs_promotion":
            result = game.make_move(fr, fc, tr, tc, "q")
        
        if result not in ("success", "checkmate", "stalemate"):
            game_history.pop()
        
        move_count += 1
    
    # Determine outcome
    status = game.status()
    if status == "checkmate":
        loser = game.get_turn()
        outcome = -1.0 if loser == "white" else 1.0
    else:
        outcome = 0.0
    
    # Convert to training samples
    training_data = []
    for board_tensor, policy_target, turn in game_history:
        value_target = outcome if turn == "white" else -outcome
        training_data.append((board_tensor, policy_target, value_target))
    
    return training_data, outcome, move_count


def generate_self_play_data(model, device, num_games, temperature):
    """Play multiple games and collect all training data."""
    all_data = []
    results = {"white": 0, "black": 0, "draw": 0}
    
    for i in range(num_games):
        data, outcome, moves = play_one_game(model, device, temperature)
        all_data.extend(data)
        
        if outcome > 0:
            results["white"] += 1
        elif outcome < 0:
            results["black"] += 1
        else:
            results["draw"] += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Game {i+1}/{num_games} complete "
                  f"({moves} moves, {'W' if outcome > 0 else 'B' if outcome < 0 else 'D'})")
    
    print(f"  Results: W={results['white']} B={results['black']} D={results['draw']}")
    return all_data


# =============================================================================
# TRAINING
# =============================================================================

def train_on_data(model, optimizer, training_data, device, batch_size, epochs):
    """Train the network on self-play data."""
    model.train()
    random.shuffle(training_data)
    
    total_policy_loss = 0
    total_value_loss = 0
    num_batches = 0
    
    for epoch in range(epochs):
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            if len(batch) < 4:
                continue
            
            boards = torch.FloatTensor(np.array([d[0] for d in batch])).to(device)
            targets_policy = torch.FloatTensor(np.array([d[1] for d in batch])).to(device)
            targets_value = torch.FloatTensor(np.array([d[2] for d in batch])).unsqueeze(1).to(device)
            
            policy_logits, value = model(boards)
            
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(targets_policy * log_probs) / boards.size(0)
            value_loss = nn.MSELoss()(value, targets_value)
            loss = policy_loss + 1.0 * value_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_p = total_policy_loss / num_batches
            avg_v = total_value_loss / num_batches
            print(f"  Epoch {epoch+1}/{epochs} — "
                  f"policy_loss: {avg_p:.4f}, value_loss: {avg_v:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="AlphaZero chess training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint or model file to resume from")
    args = parser.parse_args()
    
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = ChessNet(
        num_planes=19,
        num_filters=CONFIG["num_filters"],
        num_blocks=CONFIG["num_blocks"],
        num_moves=engine.NUM_MOVE_INDICES
    ).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    start_cycle = 0
    
    # --- Resume from checkpoint if provided ---
    if args.resume:
        print(f"Loading weights from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        
        # Handle both full checkpoints and plain model weights
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "cycle" in checkpoint:
                start_cycle = checkpoint["cycle"]
            print(f"Resumed from cycle {start_cycle}")
        else:
            # Plain state dict (from model_final.pt)
            model.load_state_dict(checkpoint)
            print("Loaded model weights (starting cycle count from 0)")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Network has {num_params:,} parameters")
    
    # --- Training cycles ---
    total_cycles = start_cycle + CONFIG["num_cycles"]
    
    for cycle in range(start_cycle, total_cycles):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle+1}/{total_cycles}")
        print(f"{'='*60}")
        
        # Temperature decays over cycles
        temp = max(
            CONFIG["temp_min"],
            CONFIG["temperature"] - cycle * CONFIG["temp_decay"]
        )
        print(f"Temperature: {temp:.2f}")
        
        # --- Phase 1: Self-play ---
        print("\nGenerating self-play games...")
        training_data = generate_self_play_data(
            model, device,
            num_games=CONFIG["games_per_cycle"],
            temperature=temp
        )
        print(f"  Collected {len(training_data)} positions")
        
        # --- Phase 2: Train ---
        print("\nTraining...")
        train_on_data(
            model, optimizer, training_data, device,
            batch_size=CONFIG["batch_size"],
            epochs=CONFIG["epochs"]
        )
        
        # --- Save checkpoint ---
        checkpoint_path = f"checkpoint_cycle_{cycle+1}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cycle": cycle + 1,
            "config": CONFIG,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), "model_final.pt")
    print("\nTraining complete! Final model saved to model_final.pt")


if __name__ == "__main__":
    main()