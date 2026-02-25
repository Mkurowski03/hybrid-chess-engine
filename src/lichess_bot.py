#!/usr/bin/env python3
"""
Lichess Bot Interface for ChessNet-3070.
Handles connection to Lichess API, tournament matchmaking, and threaded game execution.
"""

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import berserk
import chess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import ChessEngine
from config import ModelConfig

# --- Configuration ---
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")
SYZYGY_PATH = os.getenv("SYZYGY_PATH")
EXPERIMENTS_FILE = Path("experiments.json")
RESULTS_FILE = Path("tournament_results.json")

# Default targets for tournament mode
TOURNAMENT_BOTS = [
    "Boris-Trapsky", "Humaia", "turkjs", "LeelaRogue", "TuroBot", "Elmichess", "Cimille"
]

# Lock for thread-safe file writes
RESULTS_LOCK = threading.Lock()


class GameSession:
    """
    Manages the lifecycle of a single chess game in its own thread.
    """
    def __init__(self, game_id: str, client: berserk.Client, profile: dict, engine: ChessEngine, bot_id: str):
        self.game_id = game_id
        self.client = client
        self.profile = profile
        self.engine = engine
        self.bot_id = bot_id
        self.board = chess.Board()
        self.is_white = True  # Will be updated on init
        self.opponent_name = "Unknown"
        self.opponent_rating = "?"

    def run(self):
        """Main game loop listening to the event stream."""
        logger.info(f"[{self.profile['name']}] Connected to game {self.game_id}")
        
        try:
            for event in self.client.bots.stream_game_state(self.game_id):
                if event['type'] == 'gameFull':
                    self._handle_full_state(event)
                elif event['type'] == 'gameState':
                    self._handle_state_update(event)
                elif event['type'] == 'chatLine':
                    pass
        except Exception:
            logger.exception(f"Game {self.game_id} crashed unexpectedly")
        finally:
            logger.info(f"Game {self.game_id} session ended.")

    def _handle_full_state(self, event: dict):
        """Initialize board and metadata from the full game state."""
        self.is_white = (event['white'].get('id') == self.bot_id)
        
        # Extract opponent info
        opp_color = 'black' if self.is_white else 'white'
        opp_data = event[opp_color]
        self.opponent_name = opp_data.get('name', 'Unknown')
        self.opponent_rating = opp_data.get('rating', '?')
        
        logger.info(f"Match: {self.profile['name']} vs {self.opponent_name} ({self.opponent_rating})")

        # Replay existing moves
        state = event['state']
        self._apply_moves(state.get('moves', ''))
        self._attempt_move()

    def _handle_state_update(self, state: dict):
        """Handle incremental updates."""
        # Check if game is over
        if state.get('status') != 'started':
            self._save_result(state)
            return

        self._apply_moves(state.get('moves', ''))
        self._attempt_move()

    def _apply_moves(self, moves_str: str):
        self.board.reset()
        if moves_str:
            for uci in moves_str.split():
                self.board.push(chess.Move.from_uci(uci))

    def _attempt_move(self):
        """Checks turn and asks engine for a move if it's our turn."""
        if self.board.is_game_over():
            return

        # Turn check
        is_my_turn = (self.board.turn == chess.WHITE and self.is_white) or \
                     (self.board.turn == chess.BLACK and not self.is_white)
        
        if not is_my_turn:
            return

        # Prepare Engine Arguments
        cfg = self.profile['config']
        sims = cfg.get('simulations', 800)
        
        # Parse piece values if custom
        p_vals = None
        if 'piece_values' in cfg:
            p_vals = {int(k): float(v) for k, v in cfg['piece_values'].items()}

        engine_kwargs = {
            'cpuct': cfg.get('cpuct', 2.0),
            'material_weight': cfg.get('material_weight', 0.2),
            'discount': cfg.get('discount', 0.90),
            'piece_values': p_vals,
            'book_strategy': cfg.get('book_strategy', 'best')
        }

        # Select Move
        logger.debug(f"Thinking... (Sims: {sims})")
        try:
            # We don't have exact clock times from the stream easily available in this loop structure
            # without parsing 'wtime'/'btime' from state updates. 
            # For simplicity in this bot wrapper, we rely on the engine's default time management 
            # or static allocation if wtime isn't passed.
            move = self.engine.select_move(self.board, simulations=sims, **engine_kwargs)
            
            self.client.bots.make_move(self.game_id, move.uci())
            logger.info(f"[{self.profile['name']}] Played {move.uci()}")
            
        except Exception:
            logger.exception(f"Engine failure in game {self.game_id}")

    def _save_result(self, state: dict):
        """Writes game result to JSON with thread safety."""
        winner = state.get('winner')
        status = state.get('status')
        
        if winner is None:
            outcome = "draw"
        elif (winner == 'white' and self.is_white) or (winner == 'black' and not self.is_white):
            outcome = "win"
        else:
            outcome = "loss"

        result_entry = {
            "timestamp": time.time(),
            "game_id": self.game_id,
            "profile": self.profile['name'],
            "opponent": self.opponent_name,
            "opponent_rating": self.opponent_rating,
            "outcome": outcome,
            "status": status,
            "moves": self.board.fullmove_number,
            "config": self.profile['config']
        }

        with RESULTS_LOCK:
            try:
                data = []
                if RESULTS_FILE.exists():
                    with open(RESULTS_FILE, 'r') as f:
                        data = json.load(f)
                
                data.append(result_entry)
                
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Result saved: {outcome.upper()} vs {self.opponent_name}")
            except Exception as e:
                logger.error(f"Failed to save result: {e}")


class LichessBot:
    def __init__(self, token: str):
        if not token:
            logger.critical("LICHESS_TOKEN is missing. Check your .env file.")
            sys.exit(1)

        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(self.session)
        
        # User Info
        try:
            account = self.client.account.get()
            self.bot_id = account['id']
            self.bot_name = account['username']
            logger.info(f"Authenticated as {self.bot_name} ({self.bot_id})")
        except berserk.exceptions.ResponseError as e:
            logger.critical(f"Authentication failed: {e}")
            sys.exit(1)

        # Resources
        self.profiles = self._load_profiles()
        self.engines: Dict[str, ChessEngine] = {}
        self.active_games: Dict[str, threading.Thread] = {}

    def _load_profiles(self) -> List[dict]:
        if not EXPERIMENTS_FILE.exists():
            logger.critical(f"{EXPERIMENTS_FILE} not found.")
            sys.exit(1)
            
        try:
            with open(EXPERIMENTS_FILE, 'r') as f:
                data = json.load(f)
                active = [p for p in data if p.get("active", False)]
                if not active:
                    logger.warning("No active profiles found in experiments.json")
                return active
        except Exception as e:
            logger.critical(f"Invalid experiments.json: {e}")
            sys.exit(1)

    def get_engine(self, model_path: str) -> ChessEngine:
        """Lazy-loads and caches engine instances to save VRAM."""
        if model_path not in self.engines:
            logger.info(f"Initializing engine: {model_path} (Syzygy={bool(SYZYGY_PATH)})")
            self.engines[model_path] = ChessEngine(
                checkpoint_path=model_path,
                syzygy_path=SYZYGY_PATH,
                device="cuda" if os.getenv("DEVICE", "cuda") == "cuda" else "cpu"
            )
        return self.engines[model_path]

    def run(self):
        """Main event loop."""
        logger.info("Listening for events...")
        try:
            for event in self.client.bots.stream_incoming_events():
                if event['type'] == 'challenge':
                    self._handle_challenge(event['challenge'])
                elif event['type'] == 'gameStart':
                    self._start_game(event['game']['id'])
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
        except Exception:
            logger.exception("Event stream connection lost")

    def _handle_challenge(self, challenge: dict):
        c_id = challenge['id']
        challenger = challenge['challenger']['name']
        logger.info(f"Challenge received from {challenger}")
        
        # Accept everything for now (add filters here if needed)
        try:
            self.client.bots.accept_challenge(c_id)
            logger.info(f"Accepted challenge {c_id}")
        except berserk.exceptions.ResponseError as e:
            logger.error(f"Could not accept challenge: {e}")

    def _start_game(self, game_id: str):
        """Spawns a new GameSession thread."""
        # Clean up finished threads
        self.active_games = {g: t for g, t in self.active_games.items() if t.is_alive()}

        if not self.profiles:
            logger.error("No profiles available to play!")
            return

        # 1. Smart Selection: Pick profile with fewest games against this opponent?
        # Since we don't know the opponent yet (stream hasn't started), pick Random for now.
        # The GameSession can refine logic if needed, but simple random is robust.
        profile = random.choice(self.profiles)
        
        # 2. Get Engine
        model_path = profile['config'].get('model_path', 'checkpoints/baseline/chessnet_epoch9.pt')
        engine = self.get_engine(model_path)

        # 3. Launch Thread
        session = GameSession(game_id, self.client, profile, engine, self.bot_id)
        t = threading.Thread(target=session.run, daemon=True, name=f"Game-{game_id}")
        t.start()
        
        self.active_games[game_id] = t

    def start_tournament_mode(self):
        """Background thread to actively seek games against target bots."""
        def seeker():
            logger.info("Tournament Seeker Started.")
            while True:
                current_load = len([t for t in self.active_games.values() if t.is_alive()])
                
                if current_load < 3:
                    try:
                        self._seek_game()
                    except Exception as e:
                        logger.error(f"Seek error: {e}")
                
                time.sleep(45) # Rate limit protection

        t = threading.Thread(target=seeker, daemon=True, name="TournamentSeeker")
        t.start()

    def _seek_game(self):
        """Determine next opponent and send challenge."""
        # Load history to balance matchups
        history = []
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, 'r') as f:
                history = json.load(f)

        # Count matches
        counts = {bot: 0 for bot in TOURNAMENT_BOTS}
        for h in history:
            opp = h.get('opponent')
            if opp in counts:
                counts[opp] += 1
        
        # Find underserved opponents
        # Target: 2 games per profile per opponent
        target_count = len(self.profiles) * 2
        candidates = [b for b, c in counts.items() if c < target_count]

        if not candidates:
            logger.info("Tournament quota met. Waiting for new profiles or manual games.")
            return

        opponent = random.choice(candidates)
        logger.info(f"Seeking match against {opponent} (10+0)...")
        
        try:
            self.client.challenges.create(
                opponent, 
                rated=True, 
                clock_limit=600, 
                clock_increment=0
            )
        except berserk.exceptions.ResponseError as e:
            logger.warning(f"Challenge to {opponent} failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChessNet Lichess Bot")
    parser.add_argument("--seek", action="store_true", help="Enable active tournament seeking")
    args = parser.parse_args()

    bot = LichessBot(LICHESS_TOKEN)

    if args.seek:
        bot.start_tournament_mode()

    bot.run()