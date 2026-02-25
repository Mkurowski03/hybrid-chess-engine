#!/usr/bin/env python3
"""
Lichess Account Upgrade Utility.

Upgrades a standard Lichess account to a BOT account.
Requires a valid LICHESS_TOKEN in the .env file.
"""

import logging
import os
import sys

import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

API_URL = "https://lichess.org/api/bot/account/upgrade"


def upgrade_to_bot(token: str) -> bool:
    """
    Sends the upgrade request to Lichess API.
    
    Args:
        token: Valid OAuth2 token with 'bot:play' scope.
        
    Returns:
        True if successful, False otherwise.
    """
    headers = {"Authorization": f"Bearer {token}"}
    
    # Mask token for safe logging
    masked_token = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "****"
    logger.info(f"Attempting upgrade with token: {masked_token}")

    try:
        response = requests.post(API_URL, headers=headers, timeout=10)
        
        if response.status_code == 200:
            logger.info("Success! Account upgraded to BOT status.")
            logger.info("Please log in to Lichess to verify the purple 'BOT' tag.")
            return True
            
        elif response.status_code == 401:
            logger.error("Authentication failed. Check if your token is valid and has 'bot:play' scope.")
        else:
            logger.error(f"Upgrade failed (Status {response.status_code}): {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during upgrade: {e}")

    return False


def main():
    # Load environment variables
    load_dotenv()
    
    token = os.getenv("LICHESS_TOKEN")

    # validation
    if not token or token.strip() == "" or "your_token" in token:
        logger.critical("LICHESS_TOKEN is missing or invalid in .env file.")
        logger.info("Steps to fix:")
        logger.info("1. Create a token at https://lichess.org/account/oauth/token")
        logger.info("2. Ensure 'Play as a bot' scope is selected.")
        logger.info("3. Paste it into your .env file as LICHESS_TOKEN=...")
        sys.exit(1)

    success = upgrade_to_bot(token)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()