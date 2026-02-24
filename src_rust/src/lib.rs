use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray3, IntoPyArray};
use shakmaty::{fen::Fen, CastlingMode, Chess, Color, Position, Role, MoveList};
use std::str::FromStr;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

fn compute_hash(pos: &Chess) -> u64 {
    let mut hasher = DefaultHasher::new();
    pos.hash(&mut hasher);
    hasher.finish()
}

fn move_to_policy_index(m: &shakmaty::Move, turn: Color) -> usize {
    let mut from_sq = m.from().map(|sq| sq as usize).unwrap_or(0);
    let mut to_sq = m.to() as usize;
    if turn == Color::Black {
        from_sq ^= 56;
        to_sq ^= 56;
    }
    from_sq * 64 + to_sq
}

#[derive(Clone)]
struct Node {
    pos: Chess,
    hash: u64,
    parent: Option<usize>,
    action: Option<String>,
    children: Vec<usize>,
    
    visits: u32,
    value_sum: f32,
    prior: f32,
    
    expanded: bool,
    terminal: bool,
}

impl Node {
    fn new(pos: Chess, parent: Option<usize>, action: Option<String>, prior: f32) -> Self {
        let hash = compute_hash(&pos);
        Node {
            pos,
            hash,
            parent,
            action,
            children: Vec::new(),
            visits: 0,
            value_sum: 0.0,
            prior,
            expanded: false,
            terminal: false,
        }
    }
}

pub fn encode_rust_board<'py>(pos: &Chess, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let mut planes = ndarray::Array3::<f32>::zeros((18, 8, 8));
    
    let us = pos.turn();
    let them = us.other();
    let flip = us == Color::Black;

    let roles = [
        Role::Pawn, Role::Knight, Role::Bishop, 
        Role::Rook, Role::Queen, Role::King
    ];

    for (i, &role) in roles.iter().enumerate() {
        let mut our_bb = pos.board().by_piece(role.of(us)).0;
        let mut their_bb = pos.board().by_piece(role.of(them)).0;

        if flip {
            our_bb = our_bb.swap_bytes();
            their_bb = their_bb.swap_bytes();
        }

        let mut bb = our_bb;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            let r = sq / 8;
            let c = sq % 8;
            planes[[i, r, c]] = 1.0;
            bb &= bb - 1;
        }

        let mut bb = their_bb;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            let r = sq / 8;
            let c = sq % 8;
            planes[[i + 6, r, c]] = 1.0;
            bb &= bb - 1;
        }
    }

    let castling = pos.castles().castling_rights();
    let white_k = castling.contains(shakmaty::Square::H1);
    let white_q = castling.contains(shakmaty::Square::A1);
    let black_k = castling.contains(shakmaty::Square::H8);
    let black_q = castling.contains(shakmaty::Square::A8);

    let (our_k, our_q, their_k, their_q) = if us == Color::White {
        (white_k, white_q, black_k, black_q)
    } else {
        (black_k, black_q, white_k, white_q)
    };

    let halfmoves = pos.halfmoves() as f32;
    let p17 = (halfmoves / 100.0).min(1.0);

    for r in 0..8 {
        for c in 0..8 {
            planes[[12, r, c]] = 0.0;
            planes[[13, r, c]] = if our_k { 1.0 } else { 0.0 };
            planes[[14, r, c]] = if our_q { 1.0 } else { 0.0 };
            planes[[15, r, c]] = if their_k { 1.0 } else { 0.0 };
            planes[[16, r, c]] = if their_q { 1.0 } else { 0.0 };
            planes[[17, r, c]] = p17;
        }
    }

    // SAFETY: Translates the Rust `ndarray` to a Python numpy object dynamically.
    // `into_pyarray_bound` executes a zero-copy transfer of the heap-allocated Rust array into Python space.
    Ok(planes.into_pyarray_bound(py))
}

#[pyclass]
pub struct RustBoard {
    pos: Chess,
}

#[pymethods]
impl RustBoard {
    #[new]
    pub fn new(fen: &str) -> PyResult<Self> {
        let setup = Fen::from_str(fen)
            .map_err(|e| PyValueError::new_err(format!("Invalid FEN: {}", e)))?;
        let pos = setup.into_position(CastlingMode::Standard)
            .map_err(|e| PyValueError::new_err(format!("Failed to parse board: {}", e)))?;
        Ok(RustBoard { pos })
    }

    pub fn turn(&self) -> bool {
        self.pos.turn() == Color::White
    }

    pub fn legal_moves(&self) -> Vec<String> {
        let moves: MoveList = self.pos.legal_moves();
        moves.into_iter()
            .map(|m| m.to_uci(CastlingMode::Standard).to_string())
            .collect()
    }

    pub fn push_uci(&mut self, uci: &str) -> PyResult<()> {
        let moves: MoveList = self.pos.legal_moves();
        if let Some(m) = moves.into_iter().find(|m| m.to_uci(CastlingMode::Standard).to_string() == uci) {
            self.pos.play_unchecked(&m);
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("Illegal move: {}", uci)))
        }
    }

    pub fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        encode_rust_board(&self.pos, py)
    }
}

#[pyclass]
pub struct RustMCTS {
    nodes: Vec<Node>,
    root: usize,
    cpuct: f32,
    discount: f32,
}

#[pymethods]
impl RustMCTS {
    #[new]
    pub fn new(fen: &str, cpuct: Option<f32>, discount: Option<f32>) -> PyResult<Self> {
        let setup = Fen::from_str(fen)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid FEN: {}", e)))?;
        let pos = setup.into_position(CastlingMode::Standard)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to parse board: {}", e)))?;
        
        let root_node = Node::new(pos, None, None, 1.0);
        
        Ok(RustMCTS {
            nodes: vec![root_node],
            root: 0,
            cpuct: cpuct.unwrap_or(1.25),
            discount: discount.unwrap_or(0.99),
        })
    }

    pub fn select_leaves<'py>(&mut self, py: Python<'py>, batch_size: usize) -> PyResult<(Vec<Bound<'py, PyArray3<f32>>>, Vec<usize>)> {
        let mut leaves = Vec::new();
        let mut tensors = Vec::new();

        for _ in 0..batch_size {
            let mut curr = self.root;
            
            let mut depth = 0;
            while self.nodes[curr].expanded && !self.nodes[curr].terminal && depth < 200 {
                let parent_visits = self.nodes[curr].visits;
                let mut best_score = f32::NEG_INFINITY;
                let mut best_child = None;
                
                for &child_id in &self.nodes[curr].children {
                    let child = &self.nodes[child_id];
                    
                    let q = if child.visits > 0 { -(child.value_sum / child.visits as f32) } else { 0.0 };
                    let u = self.cpuct * child.prior * (parent_visits as f32).sqrt() / (1.0 + child.visits as f32);
                    
                    let score = q + u;
                    if score > best_score {
                        best_score = score;
                        best_child = Some(child_id);
                    }
                }
                
                if let Some(c) = best_child {
                    curr = c;
                    depth += 1;
                } else {
                    break;
                }
            }
            
            // Draw guard
            let mut rep_count = 1;
            let mut p_opt = self.nodes[curr].parent;
            while let Some(p) = p_opt {
                if self.nodes[p].hash == self.nodes[curr].hash {
                    rep_count += 1;
                }
                p_opt = self.nodes[p].parent;
            }
            
            if !self.nodes[curr].terminal {
                if rep_count >= 3 || self.nodes[curr].pos.halfmoves() >= 100 || self.nodes[curr].pos.is_insufficient_material() {
                    self.nodes[curr].terminal = true;
                } else if let Some(outcome) = self.nodes[curr].pos.outcome() {
                    self.nodes[curr].terminal = true;
                }
            }

            if self.nodes[curr].terminal {
                let mut v = 0.0;
                if let Some(outcome) = self.nodes[curr].pos.outcome() {
                    v = match outcome {
                        shakmaty::Outcome::Decisive { winner: w } => if w == self.nodes[curr].pos.turn() { 1.0 } else { -1.0 },
                        shakmaty::Outcome::Draw => 0.0,
                    };
                }
                
                let mut back_curr = curr;
                let mut current_val = v;
                self.nodes[back_curr].visits += 1;
                self.nodes[back_curr].value_sum += current_val;
                
                let mut path_depth = 0;
                while let Some(p) = self.nodes[back_curr].parent {
                    path_depth += 1;
                    current_val = -current_val;
                    let discounted = current_val * self.discount.powi(path_depth);
                    self.nodes[p].visits += 1;
                    self.nodes[p].value_sum += discounted;
                    back_curr = p;
                }
                continue;
            }

            // Virtual loss
            let mut back_curr = curr;
            self.nodes[back_curr].visits += 1;
            while let Some(p) = self.nodes[back_curr].parent {
                self.nodes[p].visits += 1;
                back_curr = p;
            }

            leaves.push(curr);
            let tensor = encode_rust_board(&self.nodes[curr].pos, py)?;
            tensors.push(tensor);
        }

        Ok((tensors, leaves))
    }

    pub fn backpropagate(&mut self, node_ids: Vec<usize>, values: Vec<f32>, policies: Vec<Vec<f32>>) -> PyResult<()> {
        for i in 0..node_ids.len() {
            let curr = node_ids[i];
            let raw_v = values[i];
            let policy_logits = &policies[i];
            
            if !self.nodes[curr].expanded && !self.nodes[curr].terminal {
                self.nodes[curr].expanded = true;
                
                let legal_moves = self.nodes[curr].pos.legal_moves();
                let turn = self.nodes[curr].pos.turn();
                
                let mut sum_exp = 0.0;
                let mut child_priors = Vec::new();
                let mut max_logit = f32::NEG_INFINITY;
                
                for m in &legal_moves {
                    let idx = move_to_policy_index(m, turn);
                    let logit = policy_logits[idx];
                    if logit > max_logit { max_logit = logit; }
                }
                
                for m in &legal_moves {
                    let idx = move_to_policy_index(m, turn);
                    let exp = (policy_logits[idx] - max_logit).exp();
                    child_priors.push(exp);
                    sum_exp += exp;
                }
                
                for (i, m) in legal_moves.clone().into_iter().enumerate() {
                    let prior = child_priors[i] / sum_exp;
                    let mut child_pos = self.nodes[curr].pos.clone();
                    child_pos.play_unchecked(&m);
                    
                    let child_node = Node::new(child_pos, Some(curr), Some(m.to_uci(CastlingMode::Standard).to_string()), prior);
                    let child_id = self.nodes.len();
                    self.nodes.push(child_node);
                    self.nodes[curr].children.push(child_id);
                }
            }
            
            let mut back_curr = curr;
            let mut current_val = raw_v;
            self.nodes[back_curr].value_sum += current_val;
            
            let mut path_depth = 0;
            while let Some(p) = self.nodes[back_curr].parent {
                path_depth += 1;
                current_val = -current_val;
                let discounted = current_val * self.discount.powi(path_depth);
                self.nodes[p].value_sum += discounted;
                back_curr = p;
            }
        }
        
        Ok(())
    }

    pub fn best_move(&self) -> String {
        let mut best_n = 0;
        let mut best_m = String::new();
        
        for &child_id in &self.nodes[self.root].children {
            let n = self.nodes[child_id].visits;
            if n > best_n {
                best_n = n;
                if let Some(action) = &self.nodes[child_id].action {
                    best_m = action.clone();
                }
            }
        }
        best_m
    }
}

#[pymodule]
fn chess_engine_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBoard>()?;
    m.add_class::<RustMCTS>()?;
    Ok(())
}
