import os
import urllib.request
from bs4 import BeautifulSoup
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def download_syzygy():
    urls = [
        "http://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/",
        "http://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/"
    ]
    target_dir = Path("tablebases")
    
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        
    files_to_download = []
    
    for base_url in urls:
        logging.info(f"Fetching tablebase list from {base_url}...")
        try:
            response = urllib.request.urlopen(base_url)
            html = response.read().decode('utf-8')
        except Exception as e:
            logging.error(f"Failed to fetch directory listing: {e}")
            continue

        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a')
        
        for link in links:
            href = link.get('href')
            if href and (href.endswith('.rtbw') or href.endswith('.rtbz')):
                files_to_download.append((base_url, href))
            
    total_files = len(files_to_download)
    logging.info(f"Found {total_files} tablebase files to download.")
    
    for i, (base_url, file_name) in enumerate(files_to_download, 1):
        file_url = base_url + file_name
        dest_path = target_dir / file_name
        
        if dest_path.exists() and dest_path.stat().st_size > 0:
            logging.info(f"[{i}/{total_files}] Skipping {file_name} (already exists)")
            continue
            
        logging.info(f"[{i}/{total_files}] Downloading {file_name}...")
        try:
            urllib.request.urlretrieve(file_url, dest_path)
        except Exception as e:
            logging.error(f"Failed to download {file_name}: {e}")

    logging.info("Syzygy tablebase download complete!")

if __name__ == "__main__":
    download_syzygy()
