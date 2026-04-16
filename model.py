"""
model.py — AlphaZero-style neural network for chess.

Architecture overview:
    Input:  Board tensor (19 planes of 8x8)
            ↓
    Backbone: Stack of residual blocks (convolutions)
            ↓
    ┌───────┴───────┐
    Policy Head     Value Head
    (4672 outputs)  (1 output)
    
    Policy Head: "What move should I play?"
        Outputs a probability for each of the 4672 possible move indices.
        During play, we mask illegal moves and sample from the rest.
    
    Value Head: "Who's winning from this position?"
        Outputs a single number from -1 (black wins) to +1 (white wins).
        This tells the network how good the current position is.

What's a residual block?
    Two convolution layers with a skip connection. The skip connection
    means the block learns "what to ADD to the input" rather than
    "what the output should be." This makes deep networks much easier
    to train — it's the key insight from the ResNet paper (2015).
    
    Input → Conv → ReLU → Conv → + Input → ReLU → Output
                                  ↑
                          (skip connection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    One residual block: two conv layers with a skip connection.
    
    The convolutions use 3x3 filters (same as image recognition networks).
    The input passes through two convolutions, then gets ADDED back to
    the original input. This "shortcut" is what makes training stable
    even with many layers stacked.
    """
    def __init__(self, num_filters):
        super().__init__()
        # First convolution: input → num_filters feature maps
        # padding=1 keeps the spatial size at 8x8
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(num_filters)  # normalizes activations (stabilizes training)
        
        # Second convolution
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        # Save input for the skip connection
        residual = x
        
        # First conv + normalize + activate
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second conv + normalize
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the original input back (the "residual" connection)
        out = out + residual
        
        # Final activation
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """
    The full AlphaZero network.
    
    Args:
        num_planes:  Number of input planes (19 for our encoding)
        num_filters: Width of the network (how many features each conv layer extracts)
        num_blocks:  Depth of the network (how many residual blocks to stack)
        num_moves:   Size of the policy output (4672 for chess)
    
    Start small (5 blocks, 64 filters) to iterate fast.
    AlphaZero used 19 blocks with 256 filters — we can scale up later.
    """
    def __init__(self, num_planes=19, num_filters=64, num_blocks=5, num_moves=4672):
        super().__init__()
        
        self.num_moves = num_moves
        
        # --- Initial convolution ---
        # Converts 19 input planes to num_filters feature maps
        self.initial_conv = nn.Conv2d(num_planes, num_filters, kernel_size=3, padding=1)
        self.initial_bn   = nn.BatchNorm2d(num_filters)
        
        # --- Residual backbone ---
        # Stack of residual blocks. Each one refines the features.
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])
        
        # --- Policy head ---
        # Narrows from num_filters to 2 feature maps, then flattens to 4672 outputs
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)  # 1x1 conv to reduce channels
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 8 * 8, num_moves)  # 128 → 4672
        
        # --- Value head ---
        # Narrows to 1 feature map, then through a hidden layer to 1 output
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(1 * 8 * 8, 64)   # 64 → 64 hidden
        self.value_fc2  = nn.Linear(64, 1)             # 64 → 1 output

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Board tensor, shape (batch_size, 19, 8, 8)
            
        Returns:
            policy_logits: Raw scores for each move, shape (batch_size, 4672)
                          (not yet softmaxed — we do that after masking illegal moves)
            value: Position evaluation, shape (batch_size, 1), range [-1, +1]
        """
        # Initial convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = F.relu(x)
        
        # Pass through all residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # --- Policy head ---
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)    # flatten: (batch, 2, 8, 8) → (batch, 128)
        policy_logits = self.policy_fc(p)  # (batch, 128) → (batch, 4672)
        
        # --- Value head ---
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)    # flatten: (batch, 1, 8, 8) → (batch, 64)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # tanh squashes to [-1, +1]
        
        return policy_logits, value


def get_policy_probabilities(model, board_tensor, legal_move_indices, device="cpu"):
    """
    Get move probabilities for a single position, masked to legal moves only.
    
    This is what you call during self-play to pick a move:
      1. Run the network to get raw scores for all 4672 moves
      2. Set illegal moves to -infinity (so they get 0 probability after softmax)
      3. Softmax the remaining scores to get probabilities
    
    Args:
        model:              The ChessNet
        board_tensor:       numpy array shape (19, 8, 8) from game.get_board_tensor()
        legal_move_indices: list of valid move indices (from move_to_index for each legal move)
        device:             "cuda" or "cpu"
        
    Returns:
        probs:   probability for each legal move (sums to 1)
        indices: the corresponding move indices
        value:   position evaluation [-1, +1]
    """
    model.eval()
    with torch.no_grad():
        # Convert numpy array to PyTorch tensor, add batch dimension
        x = torch.FloatTensor(board_tensor).unsqueeze(0).to(device)  # (1, 19, 8, 8)
        
        policy_logits, value = model(x)
        
        # Mask: set all illegal moves to -infinity
        mask = torch.full((4672,), float('-inf'), device=device)
        for idx in legal_move_indices:
            mask[idx] = 0.0
        
        masked_logits = policy_logits[0] + mask  # illegal moves → -inf
        probs = F.softmax(masked_logits, dim=0)  # softmax: -inf → 0, rest → probabilities
        
        # Extract only the legal move probabilities
        legal_probs = probs[legal_move_indices].cpu().numpy()
        value_scalar = value.item()
        
    return legal_probs, legal_move_indices, value_scalar