
import math
import sys
from pathlib import Path
import numpy as np
import torch
import chess

# Allow direct imports when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.board_encoder import encode_board

def _move_to_policy_index(move: chess.Move, turn: chess.Color) -> int:
    """Encode a ``chess.Move`` into a policy index."""
    from_sq = move.from_square
    to_sq = move.to_square
    if turn == chess.BLACK:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)
    return from_sq * 64 + to_sq

class MCTSNode:
    def __init__(self, prob: float, parent=None):
        self.P = prob          # Prior probability from NN policy
        self.N = 0             # Visit count
        self.W = 0.0           # Total action value
        self.Q = 0.0           # Mean action value (W/N)
        self.parent = parent
        self.children: dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False

    def is_leaf(self):
        return not self.is_expanded

    def best_child(self, cpuct=2.0):
        """Select the child with the highest PUCT score."""
        # U(s, a) = Q(s, a) + cpuct * P(s, a) * sqrt(sum(N)) / (1 + N(s, a))
        best_score = -float("inf")
        best_move = None
        best_node = None

        # Precompute sqrt(sum(N))
        # Note: simulation counts parent visits including the current one being searched
        sqrt_n = math.sqrt(self.N)

        for move, child in self.children.items():
            u = child.Q + cpuct * child.P * sqrt_n / (1 + child.N)
            if u > best_score:
                best_score = u
                best_move = move
                best_node = child
        
        return best_move, best_node

class MCTS:
    def __init__(self, model, device="cuda", cpuct=2.0, batch_size=8, material_weight=0.2, discount=0.90, piece_values=None):
        self.model = model
        self.device = device
        self.cpuct = cpuct
        self.batch_size = batch_size
        self.material_weight = material_weight
        self.discount = discount
        self.piece_values = piece_values or {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.2, chess.ROOK: 5, chess.QUEEN: 9}
        self.batch_size = batch_size
        self.root = None

    def search(self, board: chess.Board, num_simulations: int = 800):
        """Run MCTS search for the given board state."""
        self.model.eval()
        
        self.model.eval()
        
        # Reset root to ensure we don't use stale tree from previous move
        # (TODO: Implement proper subtree reuse based on move history)
        self.root = MCTSNode(1.0)
        policy, _ = self._predict_single(board)
        self._expand_node(self.root, board, policy)

        # Main simulation loop (batched)
        num_batches = math.ceil(num_simulations / self.batch_size)
        
        for _ in range(num_batches):
            leaves = []
            paths = []
            
            # 1. Select (gather a batch of leaves)
            for _ in range(self.batch_size):
                leaf, path, leaf_board = self._select_leaf(self.root, board.copy())
                
                if leaf_board.is_game_over():
                    # Terminal state: handle immediately (no NN needed)
                    # Apply discount just like regular backprop
                    v = self._get_terminal_value(leaf_board, leaf_board.turn)
                    depth = len(path) - 1
                    discount = self.discount ** depth
                    discounted_v = v * discount # result for leaf to move is v, no flip needed yet as _backprop flips
                    
                    self._backpropagate(path, discounted_v)
                else:
                    leaves.append((leaf, leaf_board))
                    paths.append(path)
                    
                    # Virtual Loss to discourage re-selection in this batch
                    # Move is determined by path[-1] (node) relative to path[-2] (parent)
                    # We usually apply virtual loss to the nodes in the path? 
                    # Simpler: just apply to the leaf node so it looks "bad/busy".
                    # Actually, standard VL is applied all the way down.
                    # Simplified: Leaf N+=1, W-=1 (assume loss)
                    leaf.N += 1
                    leaf.W -= 1.0 
                    leaf.Q = leaf.W / leaf.N
            
            if not leaves:
                continue

            # 2. Evaluate Batch
            states = []
            for _, b in leaves:
                states.append(encode_board(b))
            
            if not states:
                continue
                
            states_tensor = torch.from_numpy(np.array(states)).to(self.device)
            
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda"):
                        policy_batch, value_batch = self.model(states_tensor)
                else:
                    policy_batch, value_batch = self.model(states_tensor)
            
            policy_batch = policy_batch.cpu().numpy()
            value_batch = value_batch.cpu().numpy().flatten()
            
            # 3. Expand & Backpropagate
            for i, (leaf, leaf_board) in enumerate(leaves):
                path = paths[i]
                
                # Revert Virtual Loss on the leaf
                leaf.N -= 1
                leaf.W += 1.0
                leaf.Q = leaf.W / leaf.N if leaf.N > 0 else 0.0
                
                # Value from NN is for the player who just moved?
                # Usually NN(state) -> v is for player to move.
                # So value for parent (who moved) is -v.
                v = float(value_batch[i])

                # Blend with material score (prevent tactical blunders)
                # 80% NN, 20% Material
                mat_score = self._calculate_material_score(leaf_board)
                if leaf_board.turn == chess.BLACK:
                     mat_score = -mat_score
                
                v = 0.8 * v + 0.2 * mat_score
                
                # Expand
                self._expand_node(leaf, leaf_board, policy_batch[i])
                
                # Backpropagate (-v because leaf value is for opponent of parent)
                # Apply sharper discount to force faster mates
                depth = len(path) - 1
                discount = self.discount ** depth
                discounted_v = (-v) * discount
                
                self._backpropagate(path, discounted_v)

        # Return best move (most visited)
        best_n = -1
        best_move = None
        for move, child in self.root.children.items():
            if child.N > best_n:
                best_n = child.N
                best_move = move
        
        # Debug info
        if best_move and best_move in self.root.children:
            best_q = self.root.children[best_move].Q
            # print(f"MCTS Best: {best_move} N={best_n} Q={best_q:.2f}")

        return best_move

    def _select_leaf(self, root, board):
        node = root
        path = [node]
        
        while not node.is_leaf():
            move, child = node.best_child(self.cpuct)
            if child is None: # Should not happen unless terminal
                 break
            board.push(move)
            
            # Check for 3-fold repetition or 2-fold repetition (loop guard)
            if board.can_claim_draw() or board.is_repetition(2):
                # Loop breaker: treat as terminal leaf (will be evaluated as Draw 0.0)
                node = child
                path.append(node)
                break

            node = child
            path.append(node)
            
        return node, path, board

    def _expand_node(self, node, board, policy_logits):
        node.is_expanded = True
        
        # Softmax over legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return

        policy_map = {}
        for move in legal_moves:
            idx = _move_to_policy_index(move, board.turn)
            policy_map[move] = policy_logits[idx]
        
        # Normalize
        probs = np.array(list(policy_map.values()))
        probs = np.exp(probs - np.max(probs)) # Stable softmax
        probs /= np.sum(probs)
        
        for i, move in enumerate(policy_map.keys()):
            node.children[move] = MCTSNode(float(probs[i]), parent=node)

    def _backpropagate(self, path, value):
        # Value is from perspective of the player at the LEAF node.
        # We propagate up. At each step, we flip perspective.
        # Leaf (Player A to move) -> evaluated as V (for A).
        # Parent (Player B moved) -> value for B is -V.
        
        current_val = value
        # Traverse path in reverse (Leaf -> Root)
        for node in reversed(path):
            node.N += 1
            node.W += current_val
            node.Q = node.W / node.N
            current_val = -current_val # Flip for parent

    def _get_terminal_value(self, board, turn_at_leaf):
        # Result is 1-0, 0-1, 1/2-1/2.
        # We want value for 'turn_at_leaf'.
        outcome = board.outcome(claim_draw=True)
        if not outcome:
            return 0.0
        
        if outcome.winner is None:
            return 0.0
        
        if outcome.winner == turn_at_leaf:
            return 1.0
        else:
            return -1.0

    def _calculate_material_score(self, board):
        """Simple material heuristic in [-1, 1] range."""
        score = 0.0
        for piece_type, value in self.piece_values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        
        # Normalize: max material diff ~39 (all pieces). 
        # We want drastic response for hanging pieces.
        # Let's say +3 (Piece up) -> +0.3
        norm_score = score / 20.0
        return max(-1.0, min(1.0, norm_score))

    def _predict_single(self, board):
        """Helper for single-state prediction (root init)."""
        # ... existing prediction ...
        state = encode_board(board)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    p, v = self.model(state_tensor)
            else:
                 p, v = self.model(state_tensor)
        
        nn_val = float(v.item())
        
        # Blend with material: 80% NN, 20% Material
        # Note: Material score is White-relative. NN val is Side-to-move relative?
        # Check _nn_evaluate logic in engine.py:
        # "Value-head evaluation in [-1, 1] for side to move."
        # My _calculate_material_score is White-relative.
        mat_score = self._calculate_material_score(board)
        if board.turn == chess.BLACK:
            mat_score = -mat_score
            
        mixed_val = (1.0 - self.material_weight) * nn_val + self.material_weight * mat_score
        return p.squeeze(0).cpu().numpy(), mixed_val
