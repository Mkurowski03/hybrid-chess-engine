import os
import sys
import requests
from dotenv import load_dotenv

# Load .env
load_dotenv()

LICHESS_TOKEN = os.getenv("LICHESS_TOKEN")

if not LICHESS_TOKEN or "your_token" in LICHESS_TOKEN:
    print("Error: LICHESS_TOKEN not found in .env file.")
    print("Please edit .env and paste your token.")
    sys.exit(1)

print(f"Upgrading account using token: {LICHESS_TOKEN[:4]}...{LICHESS_TOKEN[-4:]}")

response = requests.post(
    "https://lichess.org/api/bot/account/upgrade",
    headers={"Authorization": f"Bearer {LICHESS_TOKEN}"}
)

if response.status_code == 200:
    print("✅ Success! Your account is now a BOT.")
    print("Log in to Lichess and check for the purple BOT tag.")
else:
    print(f"❌ Failed: {response.text}")
