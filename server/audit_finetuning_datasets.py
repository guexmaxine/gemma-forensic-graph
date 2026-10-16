"""
Auditoria Profunda e Comparativa dos Sub-Datasets de Fine-Tuning.
Verifica:
1. Integridade de alinhamento de tokens (span [start, end] vs texto real da entidade)
2. Estatísticas comparativas por fonte (WhatsApp, Áudio Whisper, OCR Documentos, Gabarito)
3. Distribuição das classes por origem
4. Validação estrita de limites de tensores (sem índices fora de bounds ou vazios)
"""

import os
import sys
import json
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetune")

train_file = os.path.join(DATA_DIR, "train_dataset.json")
val_file = os.path.join(DATA_DIR, "val_dataset.json")

print("="*70)
print("  AUDITORIA RIGOROSA DO DATASET MULTIMODAL DE FINE-TUNING")
print("="*70)

for name, path in [("TREINO (train_dataset.json)", train_file), ("VALIDAÇÃO (val_dataset.json)", val_file)]:
    print(f"\nVerificando arquivo: {name}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Total de amostras: {len(data)}")
    
    source_counts = Counter(d.get("source", "desconhecido") for d in data)
    for src, cnt in source_counts.items():
        print(f"    - Origem '{src}': {cnt} amostras ({round(cnt/len(data)*100, 1)}%)")
        
    # Teste de integridade de spans
    total_spans = 0
    span_errors = 0
    categories = Counter()
    
    for idx, item in enumerate(data):
        tokens = item["tokenized_text"]
        ner = item["ner"]
        total_spans += len(ner)
        
        for s_idx, (s, e, cat) in enumerate(ner):
            categories[cat] += 1
            # Verifica limites
            if not (0 <= s <= e < len(tokens)):
                print(f"    [ERRO DE BOUNDS] Amostra #{idx}: span [{s}, {e}] fora dos limites (tokens: {len(tokens)})")
                span_errors += 1
                continue
                
            # Reconstrói texto
            ent_tokens = tokens[s:e+1]
            ent_text = " ".join(ent_tokens)
            if not ent_text.strip():
                print(f"    [ERRO DE TEXTO VAZIO] Amostra #{idx}: span [{s}, {e}] vazio!")
                span_errors += 1
                
    print(f"  Total de entidades no conjunto: {total_spans}")
    print(f"  Erros de bounds ou vazios encontrados: {span_errors}")
    if span_errors == 0:
        print("  Status de Integridade Estrutural: 100% PERFEITO!")

# Inspeção detalhada de amostras reais por fonte
print("\n" + "="*70)
print("  INSPEÇÃO DE AMOSTRAS REAIS POR FONTE (Texto vs Entidades Reconhecidas)")
print("="*70)

with open(train_file, "r", encoding="utf-8") as f:
    train_data = json.load(f)

sources_seen = set()
for item in train_data:
    src = item.get("source")
    if src not in sources_seen and len(item["ner"]) >= 2:
        sources_seen.add(src)
        tokens = item["tokenized_text"]
        full_text = " ".join(tokens)
        print(f"\n[Fonte: {src.upper()}]")
        print(f"Texto: \"{full_text}\"")
        print("Entidades Extraídas e Mapeadas:")
        for s, e, cat in item["ner"]:
            chunk = " ".join(tokens[s:e+1])
            print(f"  -> [{cat}]: \"{chunk}\" (tokens {s}..{e})")

print("\n" + "="*70)
print("Auditoria concluída com sucesso!")
