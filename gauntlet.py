import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def find_executable(name: str, search_dirs: List[str]) -> Optional[str]:
    """Find the path to an executable within the specified search directories.

    Args:
        name (str): Name of the executable.
        search_dirs (List[str]): Base directories to search in.

    Returns:
        Optional[str]: Absolute path to the executable or None.
    """
    for d in search_dirs:
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
        path_exe = path + ".exe"
        if os.path.exists(path_exe):
            return path_exe
    return None

def main() -> None:
    """Run the engine tournament utilizing cutechess-cli."""
    root_dir = os.path.abspath(os.path.dirname(__file__))
    search_dirs = [
        root_dir,
        os.path.join(root_dir, "cutechess-1.4.0-win64")
    ]

    # Find executables
    stockfish_path = find_executable("stockfish", search_dirs)
    if not stockfish_path:
        logging.error("Could not find stockfish or stockfish.exe in root or cutechess folders.")
        sys.exit(1)
        
    cutechess_path = find_executable("cutechess-cli", search_dirs)
    if not cutechess_path:
        logging.error("Could not find cutechess-cli or cutechess-cli.exe in root or cutechess folders.")
        sys.exit(1)

    logging.info(f"Found Stockfish: {stockfish_path}")
    logging.info(f"Found cutechess: {cutechess_path}")

    # Define our engine configurations
    hybrid_bat = os.path.join(root_dir, "run_hybrid_beast.bat")
    
    engines = [
        f'-engine name="Hybrid_Beast_40k" cmd="{hybrid_bat}" proto=uci',
    ]

    # Define Stockfish opponent ladder
    opponents = [
        f'-engine name="Stockfish-2100" cmd="{stockfish_path}" proto=uci option.UCI_LimitStrength=true option.UCI_Elo=2100',
        f'-engine name="Stockfish-2300" cmd="{stockfish_path}" proto=uci option.UCI_LimitStrength=true option.UCI_Elo=2300',
        f'-engine name="Stockfish-2500" cmd="{stockfish_path}" proto=uci option.UCI_LimitStrength=true option.UCI_Elo=2500'
    ]

    # Tournament settings (Rapid time control)
    time_control = "-each tc=180+2.0" # 3 mins + 2.0s increment
    concurrency = "-concurrency 3"
    rounds = "-rounds 2" # 2 matches with -repeat gives 4 games per pairing
    
    # cutechess-cli tournament flags:
    # -recover: restart crashed engines
    # -repeat: play each opening twice (once as white, once as black)
    flags = "-tournament round-robin -recover -repeat -pgnout hybrid_beast_results.pgn"

    # Build the command string
    cmd_parts = [cutechess_path] + time_control.split() + concurrency.split() + rounds.split()
    
    # We must properly split engine definition strings so subprocess understands them
    # A cleaner way when using subprocess.Popen is to pass the whole string if shell=True
    
    tournament_cmd = f'"{cutechess_path}" {time_control} {concurrency} {rounds}'
    for eg in engines:
        tournament_cmd += f" {eg}"
    for opp in opponents:
        tournament_cmd += f" {opp}"
    tournament_cmd += f" {flags}"

    logging.info(f"Executing Gauntlet Command: {tournament_cmd}")
    logging.info("Tournament started! Live results will appear below...")

    # Run the tournament and stream output
    try:
        process = subprocess.Popen(
            tournament_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=True
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Print output directly to our console
                print(output.strip(), flush=True)
                
        return_code = process.poll()
        if return_code == 0:
            logging.info("Tournament Completed Successfully!")
        else:
            logging.error(f"Tournament exited with error code {return_code}")
            
    except KeyboardInterrupt:
        logging.warning("Tournament interrupted by user.")
        process.terminate()
    except Exception as e:
        logging.error(f"Error running tournament: {e}")

if __name__ == "__main__":
    main()
