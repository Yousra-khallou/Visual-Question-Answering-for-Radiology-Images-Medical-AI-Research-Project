"""
Utility script for automated download of MedVQA N6 model weights.
Supports Hugging Face Hub, Google Drive (gdown), and direct local destination paths.
"""

import os
import argparse
import sys

DEFAULT_DEST = "model/best_n6.pth"

def download_from_hf(repo_id: str, filename: str = "best_n6.pth", destination: str = DEFAULT_DEST):
    """Download checkpoint from Hugging Face Model Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"📥 Downloading checkpoint from Hugging Face Hub ({repo_id}/{filename})...")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        import shutil
        shutil.copy(cached_path, destination)
        print(f"✅ Weights saved successfully to: {destination}")
        return True
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        return False

def download_from_gdrive(file_id: str, destination: str = DEFAULT_DEST):
    """Download checkpoint from Google Drive using gdown."""
    try:
        import gdown
        print(f"📥 Downloading checkpoint from Google Drive (ID: {file_id})...")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        print(f"✅ Weights saved successfully to: {destination}")
        return True
    except Exception as e:
        print(f"❌ Error downloading from Google Drive: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="MedVQA N6 Model Weights Downloader")
    parser.add_argument("--hf-repo", type=str, help="Hugging Face repo ID (e.g., username/medvqa-n6)")
    parser.add_argument("--gdrive-id", type=str, help="Shared Google Drive File ID")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST, help="Local destination file path")

    args = parser.parse_args()

    if args.hf_repo:
        download_from_hf(args.hf_repo, destination=args.dest)
    elif args.gdrive_id:
        download_from_gdrive(args.gdrive_id, destination=args.dest)
    else:
        print("💡 Usage:")
        print("  python -m src.utils.download_weights --gdrive-id <FILE_ID>")
        print("  python -m src.utils.download_weights --hf-repo <REPO_ID>")

if __name__ == "__main__":
    main()
