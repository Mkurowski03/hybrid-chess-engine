import argparse
import os
import sys
import threading
import time
import json
import random

import berserk
import chess

from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine import ChessEngine
from config import ModelConfig

CHECKPOINT_PATH = "checkpoints/baseline/chessnet_epoch9.pt"
LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")
SYZYGY_PATH = os.getenv("SYZYGY_PATH")
EXPERIMENTS_FILE = "experiments.json"
RESULTS_FILE = "tournament_results.json"
DEVICE = "cuda"

if not LICHESS_TOKEN:
    print("Error: LICHESS_TOKEN environment variable not found.")
    print("Please set it in your terminal: $env:LICHESS_TOKEN='your_token'")
    sys.exit(1)

class LichessBot:
    def __init__(self, token):
        self.session = berserk.TokenSession(token)
        self.client = berserk.Client(self.session)
        
        try:
             self.bot_id = self.client.account.get()['id']
             self.bot_name = self.client.account.get()['username']
        except:
             print("Warning: Could not get bot account details. Token might be invalid or permissions missing.")
             self.bot_id = None
             self.bot_name = "UnknownBot"
             
        self.games = {} # game_id -> {'board': board, 'is_white': bool}
        self.profiles = self._load_profiles()
        self.engines = {} # model_path -> ChessEngine instance
        self.active_games = {} # game_id -> profile configuration

    def _load_profiles(self):
        try:
            with open(EXPERIMENTS_FILE, 'r') as f:
                data = json.load(f)
                active = [p for p in data if p.get("active", False)]
                if not active:
                    print(f"No active profiles found in {EXPERIMENTS_FILE}!")
                    sys.exit(1)
                return active
        except Exception as e:
            print(f"Error loading {EXPERIMENTS_FILE}: {e}. Ensure the file exists in the project root.")
            sys.exit(1)

    def _get_engine(self, model_path):
        """Lazy load and cache engines based on model path."""
        # Convert to absolute or project-relative if needed, assuming run from root
        if model_path not in self.engines:
            print(f"Loading engine globally for {model_path}...")
            try:
                self.engines[model_path] = ChessEngine(model_path, syzygy_path=SYZYGY_PATH, device=DEVICE)
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to load engine at {model_path}. {e}")
                sys.exit(1)
        return self.engines[model_path]

    def run(self):
        print(f"Connecting to Lichess as {self.bot_name}...")
        print(f"Loaded {len(self.profiles)} active tournament profiles.")
        
        # Generator for events
        for event in self.client.bots.stream_incoming_events():
            if event['type'] == 'challenge':
                self._handle_challenge(event['challenge'])
            elif event['type'] == 'gameStart':
                self._start_game_thread(event['game']['id'])

    def _handle_challenge(self, challenge):
        print(f"Challenge from {challenge['challenger']['name']}")
        try:
            self.client.bots.accept_challenge(challenge['id'])
            print(f"Accepted challenge {challenge['id']}")
        except Exception as e:
            print(f"Failed to accept: {e}")

    def _start_game_thread(self, game_id):
        # Temporary assignment to avoid None, will be reassigned smartly in _init_game
        profile = random.choice(self.profiles)
        self.active_games[game_id] = profile
        
        t = threading.Thread(target=self._play_game, args=(game_id,))
        t.daemon = True
        t.start()

    def _play_game(self, game_id):
        print(f"Starting tracking for game {game_id}")
        import traceback
        try:
            # Stream game state
            for event in self.client.bots.stream_game_state(game_id):
                if event['type'] == 'gameFull':
                    self._init_game(game_id, event)
                elif event['type'] == 'gameState':
                    self._handle_state_change(game_id, event)
                elif event['type'] == 'chatLine':
                    pass
        except Exception:
            print(f"Game {game_id} crashed:")
            traceback.print_exc()
        finally:
            if game_id in self.games:
                del self.games[game_id]
            if game_id in self.active_games:
                del self.active_games[game_id]
            print(f"Game {game_id} removed from tracking.")

    def _get_opponent_rating(self, event, my_color):
        """Extract opponent's rating from game state."""
        opp_color = 'black' if my_color == 'white' else 'white'
        try:
            return event[opp_color].get('rating', 'Unknown')
        except:
            return 'Unknown'

    def _init_game(self, game_id, event):
        board = chess.Board()
        
        # Check if my turn
        white_id = event['white'].get('id')
        is_white = (white_id == self.bot_id)
        my_color = 'white' if is_white else 'black'
        
        # Get opponent info for logging
        opp_color_key = 'black' if is_white else 'white'
        opp_name = event[opp_color_key].get('name', 'Unknown')
        opp_rating = self._get_opponent_rating(event, my_color)
        
        # Smart Profile Assignment: balance games per opponent
        try:
            results = []
            if os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE, 'r') as f:
                    results = json.load(f)
            # Count games per profile against THIS opponent
            counts = {p['name']: 0 for p in self.profiles}
            for r in results:
                if r.get('opponent') == opp_name and r.get('profile') in counts:
                    counts[r['profile']] += 1
            # Add currently active games
            for g_id, prof in self.active_games.items():
                if g_id in self.games and self.games[g_id].get('opp_name') == opp_name:
                    if prof['name'] in counts:
                        counts[prof['name']] += 1
            best_profile = min(self.profiles, key=lambda p: counts[p['name']])
            self.active_games[game_id] = best_profile
        except Exception as e:
            print(f"Matchmaking error, using random: {e}")
            
        profile_name = self.active_games[game_id]['name']
        print(f"[{profile_name}] Game {game_id} vs {opp_name} (Rating: {opp_rating})")
        
        self.games[game_id] = {'board': board, 'is_white': is_white, 'opp_name': opp_name, 'opp_rating': opp_rating}
        
        # Replay any existing moves
        state = event['state']
        self._apply_moves(game_id, state['moves'])
        self._attempt_move(game_id)

    def _handle_state_change(self, game_id, state):
        if game_id not in self.games:
            return
            
        # Check if game ended
        status = state.get('status')
        if status and status != 'started':
            self._record_result(game_id, state)
            del self.games[game_id]
            return

        self._apply_moves(game_id, state['moves'])
        self._attempt_move(game_id)

    def _apply_moves(self, game_id, moves_str):
        data = self.games[game_id]
        board = data['board']
        board.reset()
        if moves_str:
            for uci in moves_str.split():
                board.push(chess.Move.from_uci(uci))

    def _attempt_move(self, game_id):
        data = self.games[game_id]
        board = data['board']
        is_white = data['is_white']
        
        if board.is_game_over():
             return # Handled by status check
             
        # Check turn
        if board.turn == chess.WHITE and not is_white: return
        if board.turn == chess.BLACK and is_white: return
            
        profile = self.active_games[game_id]
        print(f"[{profile['name']}] Thinking on {game_id}...")
        
        try:
            # 1. Get profile config
            config = profile['config']
            engine = self._get_engine(config.get('model_path', CHECKPOINT_PATH))
            
            # 2. Extract specific parameters
            sims = config.get('simulations', 800)
            
            p_vals_raw = config.get('piece_values', None)
            p_vals = None
            if p_vals_raw:
                p_vals = {int(k): float(v) for k, v in p_vals_raw.items()}
                
            kwargs = {
                'cpuct': config.get('cpuct', 2.0),
                'material_weight': config.get('material_weight', 0.2),
                'discount': config.get('discount', 0.90),
                'piece_values': p_vals
            }

            
            # 3. Calculate move
            move = engine.select_move(board, simulations=sims, **kwargs)
            
            # 4. Push to remote
            self.client.bots.make_move(game_id, move.uci())
            print(f"[{profile['name']}] Moved {move.uci()} in {game_id}")
        except Exception as e:
            print(f"Error making move on {game_id}: {e}")

    def _record_result(self, game_id, state):
        """Save the game result to tournament_results.json."""
        if game_id not in self.active_games or game_id not in self.games: return
        
        profile = self.active_games[game_id]
        game_data = self.games[game_id]
        
        status = state.get('status')
        winner_color = state.get('winner') # 'white', 'black', or None
        
        # Determine outcome from bot's perspective
        if winner_color is None:
            outcome = "draw"
        elif (winner_color == 'white' and game_data['is_white']) or (winner_color == 'black' and not game_data['is_white']):
            outcome = "win"
        else:
            outcome = "loss"
            
        moves = state.get('moves', '').split()
        
        result_data = {
            "timestamp": time.time(),
            "game_id": game_id,
            "profile": profile['name'],
            "opponent": game_data['opp_name'],
            "opponent_rating": game_data['opp_rating'],
            "outcome": outcome,
            "status": status,
            "moves": len(moves),
            "config": profile['config']
        }
        
        try:
            results = []
            if os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE, 'r') as f:
                    results = json.load(f)
            results.append(result_data)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"[{profile['name']}] Result for {game_id} saved: {outcome.upper()} ({status})")
        except Exception as e:
            print(f"Error saving result for {game_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lichess Bot tournament mode")
    parser.add_argument("--seek", action="store_true", help="Auto-seek rated games")
    args = parser.parse_args()

    bot = LichessBot(LICHESS_TOKEN)

    if args.seek:
        print("Auto-Seek Enabled: Creating specialized 10-minute challenges...")
        TARGET_BOTS = [
            "Boris-Trapsky", "Humaia", "turkjs", "LeelaRogue", "TuroBot", "Elmichess", "Cimille"
        ]
        
        def seeker():
            while True:
                if len(bot.games) < 3: # Allow up to 3 concurrent games for testing
                    try:
                        results = []
                        if os.path.exists(RESULTS_FILE):
                            with open(RESULTS_FILE, 'r') as f:
                                results = json.load(f)
                        
                        # Count total games per opponent
                        counts = {b: 0 for b in TARGET_BOTS}
                        for r in results:
                            opp = r.get('opponent')
                            if opp in counts:
                                counts[opp] += 1
                        for g_id, g_data in bot.games.items():
                            opp = g_data.get('opp_name')
                            if opp in counts:
                                counts[opp] += 1
                                
                        needed_opponents = [b for b, c in counts.items() if c < len(bot.profiles) * 2]
                        
                        if not needed_opponents:
                            print("Tournament complete! All target match counts reached.")
                            break
                            
                        opponent = random.choice(needed_opponents)
                        print(f"Seeking: Challenging {opponent} (10+0)...")
                        bot.client.challenges.create(opponent, rated=True, clock_limit=600, clock_increment=0)
                    except Exception as e:
                        print(f"Seek failed: {e}")
                time.sleep(45) # Wait 45s between attempts
        
        t = threading.Thread(target=seeker, daemon=True)
        t.start()
        
    bot.run()
