"""
Script de Download Automático dos Pesos do Modelo GLiNER Forense Fine-Tuned.
Baixa o checkpoint pré-treinado 'pytorch_model.bin' (1.15 GB) da Release do GitHub.
"""

import os
import sys
import urllib.request
import hashlib

sys.stdout.reconfigure(encoding='utf-8')

REPO = "guexmaxine/gemma-forensic-graph"
RELEASE_TAG = "v1.0.0"
FILENAME = "pytorch_model.bin"
DOWNLOAD_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{FILENAME}"

TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_PATH = os.path.join(TARGET_DIR, FILENAME)

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    downloaded_mb = count * block_size / (1024 * 1024)
    total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
    sys.stdout.write(f"\rProgresso: [{percent:3d}%] {downloaded_mb:7.1f} MB / {total_mb:7.1f} MB")
    sys.stdout.flush()

def main():
    if os.path.exists(TARGET_PATH):
        size_mb = os.path.getsize(TARGET_PATH) / (1024 * 1024)
        print(f"[OK] O arquivo de pesos '{FILENAME}' já existe localmente ({size_mb:.1f} MB).")
        print(f"Caminho: {TARGET_PATH}")
        return

    print("="*75)
    print("  DOWNLOAD DOS PESOS FINE-TUNED DO GLiNER FORENSE")
    print(f"  Origem: {DOWNLOAD_URL}")
    print(f"  Destino: {TARGET_PATH}")
    print("="*75)
    
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, TARGET_PATH, reporthook=download_progress)
        print(f"\n[SUCESSO] Pesos baixados com êxito em {TARGET_PATH}!")
    except Exception as e:
        print(f"\n[ERRO] Falha ao baixar pesos: {e}")
        print("Você também pode treinar o modelo localmente executando:")
        print("  python server/train_gliner_forensic.py")

if __name__ == "__main__":
    main()
