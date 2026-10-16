"""
Higienizador e Validador Estrito para o Dataset de Fine-Tuning do GLiNER Forense.
Garante:
1. Eliminação absoluta de vazamento de dados (Data Leakage) com test_sample_gold.json
2. Eliminação de spans de ruído conversacional ("oi", "aqui", "blz", "bom dia", etc.)
3. Eliminação de spans sobrepostos (conflito de labels em spans compartilhados)
4. Remoção de stopwords isoladas ('o', 'a', 'de', 'para', etc.)
5. Validação estrita das 17 categorias oficiais
6. Limitação do tamanho de tokens (máximo 128 tokens para estabilidade)
"""

import os
import sys
import json
import re

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetune")
GOLD_FILE = os.path.join(BASE_DIR, "scratch", "test_sample_gold.json")

VALID_17_CATEGORIES = {
    "PESSOA", "ORGANIZACAO", "LOCAL", "DOCUMENTO", "PROCESSO",
    "EVENTO", "VEICULO", "IMOVEL", "VALOR_MONETARIO", "DATA_HORA",
    "EMAIL", "TELEFONE", "CPF_CNPJ", "RELACAO_FAMILIAR", "FUNCAO_CARGO",
    "TRIBUTO", "OUTRO"
}

STOPWORDS = {
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "para", "pra", "pro",
    "com", "sem", "e", "ou", "mas", "que", "se", "eu", "ele", "ela", "você", "vc",
    "meu", "minha", "seu", "sua", "esse", "essa", "este", "esta", "isso", "isto",
    "já", "ja", "tá", "ta", "vai", "vou", "foi", "ser", "tem", "ter"
}

NOISE_WORDS = {
    "oi", "olá", "ola", "opa", "aqui", "lá", "la", "bom dia", "boa noite", "boa tarde",
    "blz", "beleza", "ok", "valeu", "show", "sim", "não", "nao", "obrigado", "obrigada",
    "nada", "tudo", "mais", "menos", "cara", "mano", "pow", "quem", "dia", "noite", "tarde",
    "hoje", "ontem", "amanhã", "amanha"
}

# 1. Carregar Gabarito de Ouro para Blindagem Anti-Leakage
gold_hashes = set()
if os.path.exists(GOLD_FILE):
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
        for g in gold_data:
            # Normalização leve para match seguro
            norm_txt = re.sub(r'\s+', ' ', g["text"].strip().lower())
            gold_hashes.add(norm_txt)
    print(f"Carregadas {len(gold_hashes)} sentenças de gabarito para proteção anti-leakage.")

def sanitize_dataset(input_file: str, output_file: str):
    print(f"\nHigienizando: {os.path.basename(input_file)}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    initial_count = len(data)
    clean_data = []
    removed_spans = 0
    resolved_overlaps = 0
    leaked_removed = 0
    
    for item in data:
        tokens = item["tokenized_text"]
        ner = item["ner"]
        source = item.get("source", "whatsapp_chat")
        
        # 1. Blindagem de Vazamento de Dados (Data Leakage)
        full_text = " ".join(tokens).strip().lower()
        norm_full = re.sub(r'\s+', ' ', full_text)
        if any(g in norm_full or norm_full in g for g in gold_hashes):
            leaked_removed += 1
            continue
            
        # 2. Trunca sentenças longas para max 128 tokens
        if len(tokens) > 128:
            tokens = tokens[:128]
            ner = [s for s in ner if s[1] < 128]
            
        if not ner:
            continue
            
        # 3. Filtro de ruído, stopwords e saudações
        valid_spans = []
        for s, e, cat in ner:
            if cat not in VALID_17_CATEGORIES:
                cat = "OUTRO"
                
            chunk_tokens = [tokens[i].lower() for i in range(s, e + 1)]
            chunk_text = " ".join(chunk_tokens).strip()
            
            # Se for token único e stopword, descarta
            if len(chunk_tokens) == 1 and chunk_tokens[0] in STOPWORDS:
                removed_spans += 1
                continue
                
            # Se for ruído conversacional conhecido
            if chunk_text in NOISE_WORDS:
                removed_spans += 1
                continue
                
            # Se for pontuação pura ou vazio
            if not any(c.isalnum() for c in chunk_text):
                removed_spans += 1
                continue
                
            # Se for número com 1 dígito isolado sem ser valor
            if len(chunk_text) == 1 and chunk_text.isdigit() and cat != "VALOR_MONETARIO":
                removed_spans += 1
                continue
                
            valid_spans.append([s, e, cat])
            
        if not valid_spans:
            continue
            
        # 4. Resolução de Sobreposições (Non-overlapping, maior span ganha)
        valid_spans.sort(key=lambda x: (x[1] - x[0]), reverse=True)
        non_overlapping = []
        occupied_indices = set()
        
        for s, e, cat in valid_spans:
            span_indices = set(range(s, e + 1))
            if not span_indices.intersection(occupied_indices):
                non_overlapping.append([s, e, cat])
                occupied_indices.update(span_indices)
            else:
                resolved_overlaps += 1
                
        non_overlapping.sort(key=lambda x: x[0])
        
        if non_overlapping:
            clean_data.append({
                "tokenized_text": tokens,
                "ner": non_overlapping,
                "source": source
            })
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
        
    print(f"  Amostras iniciais: {initial_count} -> Amostras purificadas: {len(clean_data)}")
    print(f"  Spans de ruído/stopwords removidos: {removed_spans}")
    print(f"  Amostras com vazamento de dados removidas: {leaked_removed}")
    print(f"  Sobreposições conflitantes resolvidas: {resolved_overlaps}")
    return clean_data

train_in = os.path.join(DATA_DIR, "train_dataset.json")
val_in = os.path.join(DATA_DIR, "val_dataset.json")

train_clean = sanitize_dataset(train_in, train_in)
val_clean = sanitize_dataset(val_in, val_in)

print("\n" + "="*70)
print(f"DATASET TOTALMENTE HIGIENIZADO, AUDITADO E PRONTO:")
print(f"  Treino: {len(train_clean)} amostras")
print(f"  Validação: {len(val_clean)} amostras")
print("="*70)
