import os
import sys
import subprocess
from pathlib import Path

def find_executable(name, search_dirs):
    for d in search_dirs:
        path = os.path.join(d, name)
        if os.path.exists(path):
            return path
        path_exe = path + ".exe"
        if os.path.exists(path_exe):
            return path_exe
    return None

def main():
    root_dir = os.path.abspath(os.path.dirname(__file__))
    search_dirs = [
        root_dir,
        os.path.join(root_dir, "cutechess-1.4.0-win64")
    ]

    # Find executables
    stockfish_path = find_executable("stockfish", search_dirs)
    if not stockfish_path:
        print("Error: Could not find stockfish or stockfish.exe in root or cutechess folders.")
        sys.exit(1)
        
    cutechess_path = find_executable("cutechess-cli", search_dirs)
    if not cutechess_path:
        print("Error: Could not find cutechess-cli or cutechess-cli.exe in root or cutechess folders.")
        sys.exit(1)

    print(f"Found Stockfish: {stockfish_path}")
    print(f"Found cutechess: {cutechess_path}")

    # Define our engine configurations
    champion_bat = os.path.join(root_dir, "run_champion_rapid.bat")
    sniper_bat = os.path.join(root_dir, "run_sniper_rapid.bat")
    solid_bat = os.path.join(root_dir, "run_solid_rapid.bat")
    
    engines = [
        f'-engine name="Champion_Rapid" cmd="{champion_bat}" proto=uci',
        f'-engine name="Sniper_Rapid" cmd="{sniper_bat}" proto=uci',
        f'-engine name="Solid_Rapid" cmd="{solid_bat}" proto=uci'
    ]

    # Define Stockfish opponent ladder
    opponents = [
        f'-engine name="Stockfish-1700" cmd="{stockfish_path}" proto=uci option.UCI_LimitStrength=true option.UCI_Elo=1700',
        f'-engine name="Stockfish-1900" cmd="{stockfish_path}" proto=uci option.UCI_LimitStrength=true option.UCI_Elo=1900',
        f'-engine name="Stockfish-2100" cmd="{stockfish_path}" proto=uci option.UCI_LimitStrength=true option.UCI_Elo=2100'
    ]

    # Tournament settings (Rapid time control)
    time_control = "-each tc=180+2.0" # 3 mins + 2.0s increment
    concurrency = "-concurrency 4"
    rounds = "-rounds 3" # 3 matches with -repeat gives 6 games per pairing
    
    # cutechess-cli tournament flags:
    # -recover: restart crashed engines
    # -repeat: play each opening twice (once as white, once as black)
    flags = "-tournament round-robin -recover -repeat -pgnout rapid_results.pgn"

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

    print("\n--- Executing Gauntlet Command ---")
    print(tournament_cmd)
    print("----------------------------------\n")
    print("Tournament started! Live results will appear below...\n")

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
                print(output.strip())
                
        return_code = process.poll()
        if return_code == 0:
            print("\nTournament Completed Successfully!")
        else:
            print(f"\nTournament exited with error code {return_code}")
            
    except KeyboardInterrupt:
        print("\nTournament interrupted by user.")
        process.terminate()
    except Exception as e:
        print(f"\nError running tournament: {e}")

if __name__ == "__main__":
    main()
