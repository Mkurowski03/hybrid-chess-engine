import concurrent.futures
import logging
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Logging configuration - clean and professional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
DEST_DIR = Path("tablebases")
SYZYGY_URLS = [
    "http://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/",
    "http://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/"
]
MAX_WORKERS = 4  # Number of parallel download threads


def get_file_list(base_url):
    """
    Parses the directory listing using regex to avoid heavy bs4 dependency.
    Returns a list of (base_url, filename) tuples.
    """
    try:
        with urllib.request.urlopen(base_url) as response:
            html = response.read().decode('utf-8')
            
        # Regex to find .rtbw (WDL) and .rtbz (DTZ) files
        # Pattern matches: href="filename.rtbw" or href="filename.rtbz"
        pattern = r'href="([^"]+\.rtb[wz])"'
        files = re.findall(pattern, html)
        
        # Deduplicate and return full tuples
        return [(base_url, f) for f in set(files)]
        
    except urllib.error.URLError as e:
        logger.error(f"Failed to fetch file list from {base_url}: {e}")
        return []


def download_file(args):
    """Worker function for threading."""
    base_url, filename = args
    file_url = base_url + filename
    dest_path = DEST_DIR / filename

    # Skip if exists and is not empty
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return f"Skipped {filename} (already exists)"

    logger.info(f"Downloading: {filename}")
    
    # Simple retry logic
    attempts = 3
    for attempt in range(attempts):
        try:
            urllib.request.urlretrieve(file_url, dest_path)
            return f"Downloaded {filename}"
        except (urllib.error.URLError, ConnectionResetError) as e:
            if attempt == attempts - 1:
                logger.error(f"Failed to download {filename} after {attempts} attempts: {e}")
                return None
            time.sleep(1)  # Brief pause before retry
                
    return None


def main():
    if not DEST_DIR.exists():
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {DEST_DIR}")

    # 1. Gather all files to download
    logger.info("Fetching file lists...")
    all_files = []
    
    for url in SYZYGY_URLS:
        files = get_file_list(url)
        all_files.extend(files)
        logger.info(f"Found {len(files)} files at {url}")

    total_files = len(all_files)
    logger.info(f"Total files to process: {total_files}")

    # 2. Download in parallel
    logger.info(f"Starting download pool with {MAX_WORKERS} workers...")
    
    start_time = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(download_file, f): f for f in all_files}
        
        # Process as they complete
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += 1
            if result and "Downloaded" in result:
                # Only log actual downloads to avoid cluttering the output
                logger.info(f"[{completed}/{total_files}] {result}")

    duration = time.perf_counter() - start_time
    logger.info(f"All tasks finished in {duration:.2f}s.")
    logger.info("Syzygy tablebases ready.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user.")
        sys.exit(0)