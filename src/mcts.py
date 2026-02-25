import math
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import chess
import numpy as np
import torch

# Inject project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.board_encoder import encode_board, move_to_policy_index

# Configure logging
logger = logging.getLogger(__name__)


class MCTSNode:
    """
    Represents a single node in the Monte Carlo Search Tree.
    Uses __slots__ to minimize memory footprint during massive searches.
    """
    __slots__ = ('P', 'N', 'W', 'Q', 'parent', 'children', 'is_expanded')

    def __init__(self, prob: float, parent: Optional['MCTSNode'] = None):
        self.P = prob          # Prior probability (Policy Head)
        self.N = 0             # Visit count
        self.W = 0.0           # Total accumulated value
        self.Q = 0.0           # Mean value (W / N)
        self.parent = parent
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False

    def is_leaf(self) -> bool:
        return not self.is_expanded

    def best_child(self, cpuct: float = 2.0) -> Tuple[Optional[chess.Move], Optional['MCTSNode']]:
        """
        Selects the child with the highest PUCT score.
        PUCT = Q(s,a) + cpuct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        best_score = -float("inf")
        best_move = None
        best_child = None

        # Precompute sqrt(N) for efficiency
        sqrt_n = math.sqrt(self.N)

        for move, child in self.children.items():
            # UCB formula variation (AlphaZero style)
            score = child.Q + cpuct * child.P * (sqrt_n / (1 + child.N))
            
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child


class MCTS:
    """
    Batched Monte Carlo Tree Search implementation.
    Designed for high-throughput GPU inference.
    """

    def __init__(
        self, 
        model: torch.nn.Module, 
        device: str = "cuda", 
        cpuct: float = 2.0, 
        batch_size: int = 8, 
        material_weight: float = 0.2, 
        discount: float = 0.90, 
        piece_values: Optional[Dict[int, int]] = None
    ):
        self.model = model
        self.device = torch.device(device)
        self.cpuct = cpuct
        self.batch_size = batch_size
        self.material_weight = material_weight
        self.discount = discount
        self.root: Optional[MCTSNode] = None
        
        # Standard piece values for material heuristic
        self.piece_values = piece_values or {
            chess.PAWN: 1, 
            chess.KNIGHT: 3, 
            chess.BISHOP: 3.2, 
            chess.ROOK: 5, 
            chess.QUEEN: 9
        }

    def search(self, board: chess.Board, num_simulations: int = 800) -> chess.Move:
        """
        Executes the MCTS search.
        
        Args:
            board: The current root board state.
            num_simulations: Total number of leaf evaluations to perform.
        """
        self.model.eval()

        # 1. Initialize Root
        # Note: In a production engine, we might reuse the subtree from the previous search.
        # For simplicity/safety, we rebuild here.
        self.root = MCTSNode(prob=1.0, parent=None)
        
        # Expand root immediately to valid moves
        root_policy, _ = self._predict_and_evaluate(board)
        self._expand_node(self.root, board, root_policy)
        
        # Add Dirichlet noise to root for exploration (standard AlphaZero practice)
        self._add_dirichlet_noise(self.root)

        # 2. Main Simulation Loop
        # We process in batches to maximize GPU utilization.
        num_batches = math.ceil(num_simulations / self.batch_size)

        for _ in range(num_batches):
            self._process_batch(board)

        # 3. Select Best Move
        # Robust selection: most visited node (N), not highest value (Q)
        if not self.root.children:
            # Fallback for no legal moves (checkmate/stalemate detected at root)
            return list(board.legal_moves)[0] if board.legal_moves else None

        best_move = max(self.root.children.items(), key=lambda item: item[1].N)[0]
        return best_move

    def _process_batch(self, root_board: chess.Board):
        """
        Runs one iteration of: Select -> Evaluate (Batch) -> Backpropagate.
        """
        leaves = []
        paths = []
        
        # --- Phase 1: Selection ---
        for _ in range(self.batch_size):
            node = self.root
            board = root_board.copy()
            path = [node]

            # Traverse tree until leaf
            while not node.is_leaf():
                move, child = node.best_child(self.cpuct)
                if child is None:
                    break # Should not happen if expanded correctly
                
                board.push(move)
                path.append(child)
                node = child

                # Check for repetition loops during tree traversal
                if board.is_repetition(2) or board.can_claim_draw():
                    break
            
            # Virtual Loss: discourages other threads/batch-items from picking this same node
            # before it's evaluated. Essential for batch diversity.
            node.N += 1
            node.W -= 1.0 
            node.Q = node.W / node.N

            leaves.append((node, board))
            paths.append(path)

        if not leaves:
            return

        # --- Phase 2: Evaluation (Batch) ---
        # Filter out terminal states that don't need NN inference
        inference_indices = []
        inference_states = []
        computed_values = [None] * len(leaves)
        computed_policies = [None] * len(leaves)

        for i, (leaf_node, leaf_board) in enumerate(leaves):
            outcome = leaf_board.outcome(claim_draw=True)
            
            if outcome:
                # Terminal state: Solve directly
                computed_values[i] = self._get_terminal_value(outcome, leaf_board.turn)
                computed_policies[i] = None # No policy for terminal state
            else:
                # Active state: Queue for NN
                inference_indices.append(i)
                inference_states.append(encode_board(leaf_board))

        # Run NN inference if needed
        if inference_states:
            tensor = torch.from_numpy(np.array(inference_states)).to(self.device)
            
            with torch.inference_mode():
                 if self.device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        policies, values = self.model(tensor)
                 else:
                    policies, values = self.model(tensor)

            policies = policies.cpu().numpy()
            values = values.cpu().numpy().flatten()

            # Map results back to the original batch indices
            for idx, pool_idx in enumerate(inference_indices):
                # Mix NN value with Material Heuristic
                _, board = leaves[pool_idx]
                nn_val = float(values[idx])
                mixed_val = self._mix_value_with_material(nn_val, board)
                
                computed_values[pool_idx] = mixed_val
                computed_policies[pool_idx] = policies[idx]

        # --- Phase 3: Backpropagation ---
        for i, (leaf_node, leaf_board) in enumerate(leaves):
            path = paths[i]
            value = computed_values[i]
            policy = computed_policies[i]

            # Revert Virtual Loss
            leaf_node.N -= 1
            leaf_node.W += 1.0
            leaf_node.Q = leaf_node.W / leaf_node.N if leaf_node.N > 0 else 0.0

            # Expand if not terminal
            if policy is not None:
                self._expand_node(leaf_node, leaf_board, policy)

            # Backpropagate value up the path
            # Value `v` is for the player whose turn it is at the leaf.
            # The parent node represents the player who *just moved*.
            # So, we flip the value at each step up the tree.
            
            # Apply discount for depth (prefer faster wins)
            depth = len(path) - 1
            discounted_val = value * (self.discount ** depth)
            
            # We start backprop. `discounted_val` is from the perspective of the 
            # player to move at the leaf.
            # The last node in `path` is the leaf.
            self._backprop_path(path, discounted_val)

    def _backprop_path(self, path: List[MCTSNode], leaf_value: float):
        """Propagates the value up the tree, flipping perspective at each step."""
        current_val = leaf_value
        
        # Reverse: Leaf -> Root
        for node in reversed(path):
            node.N += 1
            node.W += current_val
            node.Q = node.W / node.N
            
            # Flip for the parent (opponent)
            current_val = -current_val

    def _expand_node(self, node: MCTSNode, board: chess.Board, policy_logits: np.ndarray):
        """Expands a leaf node by creating children for all legal moves."""
        node.is_expanded = True
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return

        # 1. Mask illegal moves
        # 2. Apply Softmax
        policy_map = {}
        for move in legal_moves:
            idx = move_to_policy_index(move, board.turn)
            policy_map[move] = policy_logits[idx]

        # Numerical stability for softmax
        logits = np.array(list(policy_map.values()))
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)

        # Create child nodes
        for i, move in enumerate(policy_map.keys()):
            node.children[move] = MCTSNode(prob=float(probs[i]), parent=node)

    def _predict_and_evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """Runs single-instance inference (helper for root)."""
        state = encode_board(board)
        tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            policy, value = self.model(tensor)
            
        p = policy.squeeze(0).cpu().numpy()
        v = float(value.item())
        
        mixed_v = self._mix_value_with_material(v, board)
        return p, mixed_v

    def _mix_value_with_material(self, nn_value: float, board: chess.Board) -> float:
        """Blends Neural Network evaluation with a static Material Heuristic."""
        # Calculate material score (White relative)
        mat_score_white = 0.0
        for pt, val in self.piece_values.items():
            mat_score_white += len(board.pieces(pt, chess.WHITE)) * val
            mat_score_white -= len(board.pieces(pt, chess.BLACK)) * val
            
        # Normalize: 20 points of material advantage = 1.0 (Winning)
        mat_score = max(-1.0, min(1.0, mat_score_white / 20.0))
        
        # Convert to "Side to Move" perspective
        if board.turn == chess.BLACK:
            mat_score = -mat_score
            
        # Weighted blend
        return (1.0 - self.material_weight) * nn_value + (self.material_weight * mat_score)

    def _get_terminal_value(self, outcome: chess.Outcome, turn_at_leaf: chess.Color) -> float:
        """Returns the terminal value (1, -1, 0) from the perspective of turn_at_leaf."""
        if outcome.winner is None:
            return 0.0 # Draw
        
        if outcome.winner == turn_at_leaf:
            return 1.0 # Win
        else:
            return -1.0 # Loss

    def _add_dirichlet_noise(self, node: MCTSNode):
        """Adds exploration noise to the root node actions."""
        moves = list(node.children.keys())
        noise = np.random.dirichlet([0.3] * len(moves))
        
        for i, move in enumerate(moves):
            node.children[move].P = 0.75 * node.children[move].P + 0.25 * noise[i]