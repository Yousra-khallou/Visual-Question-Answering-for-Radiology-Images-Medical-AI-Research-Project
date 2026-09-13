"""
Script utilitaire pour le téléchargement automatique des poids MedVQA N6.
Supporte le téléchargement direct via URL, Hugging Face Hub, ou Google Drive.
"""

import os
import argparse
import sys

DEFAULT_DEST = "model/best_n6.pth"

def download_from_hf(repo_id: str, filename: str = "best_n6.pth", destination: str = DEFAULT_DEST):
    """Téléchargement depuis Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"📥 Téléchargement depuis Hugging Face Hub ({repo_id}/{filename})...")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        import shutil
        shutil.copy(cached_path, destination)
        print(f"✅ Poids enregistrés dans : {destination}")
        return True
    except Exception as e:
        print(f"❌ Erreur Hugging Face : {e}")
        return False

def download_from_gdrive(file_id: str, destination: str = DEFAULT_DEST):
    """Téléchargement depuis Google Drive via gdown."""
    try:
        import gdown
        print(f"📥 Téléchargement depuis Google Drive (ID: {file_id})...")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        print(f"✅ Poids enregistrés dans : {destination}")
        return True
    except Exception as e:
        print(f"❌ Erreur Google Drive : {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Téléchargement des poids du modèle MedVQA N6")
    parser.add_argument("--hf-repo", type=str, help="Identifiant du repo Hugging Face (ex: username/medvqa-n6)")
    parser.add_argument("--gdrive-id", type=str, help="ID du fichier Google Drive partagé")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST, help="Chemin de destination local")

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
