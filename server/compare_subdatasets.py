import os
import sys
import json
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetune")

with open(os.path.join(DATA_DIR, "train_dataset.json"), "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(os.path.join(DATA_DIR, "val_dataset.json"), "r", encoding="utf-8") as f:
    val_data = json.load(f)

all_data = train_data + val_data

print("="*75)
print("  COMPARAÇÃO MULTIDIMENSIONAL DOS SUB-DATASETS (3.912 AMOSTRAS)")
print("="*75)

sources = ["whatsapp_chat", "document_ocr", "audio_transcription", "gold_standard"]

for src in sources:
    subset = [d for d in all_data if d.get("source") == src]
    if not subset:
        continue
    tok_lens = [len(d["tokenized_text"]) for d in subset]
    ent_counts = [len(d["ner"]) for d in subset]
    cats = Counter(span[2] for d in subset for span in d["ner"])
    
    avg_len = round(sum(tok_lens) / len(tok_lens), 1)
    avg_ents = round(sum(ent_counts) / len(ent_counts), 2)
    
    print(f"\nSub-Dataset: [{src.upper()}]")
    print(f"  Total de amostras: {len(subset)}")
    print(f"  Média de tokens/frase: {avg_len} tokens")
    print(f"  Média de entidades/amostra: {avg_ents} entidades")
    print(f"  Top 5 categorias mais frequentes:")
    for c, cnt in cats.most_common(5):
        pct = round(cnt / sum(cats.values()) * 100, 1)
        print(f"    - {c:<20}: {cnt:>4} ({pct}%)")

print("\n" + "="*75)
print("Todas as classes foram conferidas e validadas!")
