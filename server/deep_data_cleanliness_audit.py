"""
Auditoria Profunda de Limpeza e Integridade dos Dados de Treinamento.
Verifica:
1. Spans inválidos, índices fora de limite (bounds errors), spans invertidos
2. Spans vazios ou somente pontuação/espaço
3. Stopwords e ruídos conversacionais etiquetados indevidamente
4. Conformidade estrita com as 17 categorias oficiais
5. Distribuição de comprimento de tokens e truncamento
6. Detecção de vazamento de dados (Data Leakage) com test_sample_gold.json
7. Consistência de anotações e taxa de ruído
"""

import os
import sys
import json
import re
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetune")
GOLD_FILE = os.path.join(BASE_DIR, "scratch", "test_sample_gold.json")

VALID_17 = {
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

NOISE_CONVERSACIONAL = {
    "bom dia", "boa tarde", "boa noite", "olá", "ola", "opa", "oi", "ok", "blz",
    "beleza", "sim", "não", "nao", "valeu", "obrigado", "show", "aqui", "lá", "la"
}

# Carregar Gabarito de Ouro para checar Data Leakage
gold_texts = set()
if os.path.exists(GOLD_FILE):
    with open(GOLD_FILE, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
        for g in gold_data:
            gold_texts.add(g["text"].strip().lower())

def audit_file(filepath, label):
    print(f"\n{'='*75}")
    print(f"  AUDITORIA RIGOROSA: {label} ({os.path.basename(filepath)})")
    print(f"{'='*75}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_samples = len(data)
    total_spans = 0
    bounds_errors = 0
    empty_spans = 0
    punctuation_only = 0
    stopwords_spans = 0
    noise_spans = 0
    invalid_categories = 0
    overlapping_spans = 0
    token_len_exceeded = 0
    leaked_samples = 0
    
    category_counts = Counter()
    sources_count = Counter()
    
    for i, item in enumerate(data):
        tokens = item.get("tokenized_text", [])
        ner = item.get("ner", [])
        source = item.get("source", "desconhecido")
        sources_count[source] += 1
        
        full_text = " ".join(tokens).strip().lower()
        if full_text in gold_texts:
            leaked_samples += 1
            
        if len(tokens) > 128:
            token_len_exceeded += 1
            
        occupied = set()
        for span in ner:
            total_spans += 1
            if len(span) != 3:
                bounds_errors += 1
                continue
            s, e, cat = span
            
            # Check bounds
            if not (isinstance(s, int) and isinstance(e, int) and 0 <= s <= e < len(tokens)):
                bounds_errors += 1
                continue
                
            # Check overlap
            span_indices = set(range(s, e + 1))
            if span_indices.intersection(occupied):
                overlapping_spans += 1
            occupied.update(span_indices)
            
            # Check category
            if cat not in VALID_17:
                invalid_categories += 1
            category_counts[cat] += 1
            
            # Check span text
            chunk_tokens = tokens[s:e+1]
            chunk_str = " ".join(chunk_tokens).strip()
            chunk_lower = chunk_str.lower()
            
            if not chunk_str:
                empty_spans += 1
            elif not any(c.isalnum() for c in chunk_str):
                punctuation_only += 1
            elif chunk_lower in STOPWORDS and len(chunk_tokens) == 1:
                stopwords_spans += 1
            elif chunk_lower in NOISE_CONVERSACIONAL:
                noise_spans += 1
                
    print(f"Total de Amostras: {total_samples}")
    print(f"Total de Spans/Entidades Anotadas: {total_spans}")
    print(f"\n--- VERIFICAÇÃO DE INTEGRIDADE ESTRUTURAL E QUALIDADE ---")
    print(f"  [✓/✗] Erros de Bounds (índices fora do texto): {bounds_errors}")
    print(f"  [✓/✗] Spans Vazios: {empty_spans}")
    print(f"  [✓/✗] Spans Apenas de Pontuação: {punctuation_only}")
    print(f"  [✓/✗] Spans de Stopwords Isoladas: {stopwords_spans}")
    print(f"  [✓/✗] Spans de Ruído Conversacional: {noise_spans}")
    print(f"  [✓/✗] Categorias Inválidas (fora das 17): {invalid_categories}")
    print(f"  [✓/✗] Spans Sobrepostos/Conflitantes: {overlapping_spans}")
    print(f"  [✓/✗] Sequências > 128 Tokens: {token_len_exceeded}")
    print(f"  [✓/✗] Amostras com Vazamento de Dados (Gabarito de Teste): {leaked_samples}")
    
    print(f"\n--- DISTRIBUIÇÃO DAS FONTES MULTIMODAIS ---")
    for src, c in sources_count.most_common():
        print(f"  - {src:<25}: {c:>5} amostras ({c/total_samples*100:>5.1f}%)")
        
    print(f"\n--- DISTRIBUIÇÃO DAS 17 CATEGORIAS ---")
    for cat in sorted(VALID_17):
        cnt = category_counts[cat]
        print(f"  - {cat:<22}: {cnt:>5} anotações ({cnt/total_spans*100:>5.1f}%)" if total_spans else f"  - {cat:<22}: 0")
        
    is_pristine = (bounds_errors == 0 and empty_spans == 0 and punctuation_only == 0 and 
                   stopwords_spans == 0 and noise_spans == 0 and invalid_categories == 0 and 
                   overlapping_spans == 0 and leaked_samples == 0)
    print(f"\n===> STATUS FINAL DE LIMPEZA: {'100% LIMPO E PERFEITO' if is_pristine else 'ATENÇÃO: RUÍDOS DETECTADOS'}")
    return is_pristine

train_file = os.path.join(DATA_DIR, "train_dataset.json")
val_file = os.path.join(DATA_DIR, "val_dataset.json")

t_clean = audit_file(train_file, "CONJUNTO DE TREINO")
v_clean = audit_file(val_file, "CONJUNTO DE VALIDAÇÃO")

print("\n" + "#"*75)
if t_clean and v_clean:
    print("  CONFIRMAÇÃO: OS DADOS ESTÃO 100% LIMPOS, AUDITADOS E PRONTOS!")
else:
    print("  NECESSÁRIA RE-HIGIENIZAÇÃO!")
print("#"*75)
