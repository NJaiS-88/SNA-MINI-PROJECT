"""Download the SNAP Facebook dataset into data/facebook_combined.txt (run once)."""
from pathlib import Path
from src.data_loader import download_dataset

if __name__ == "__main__":
    download_dataset(Path("data") / "facebook_combined.txt")
