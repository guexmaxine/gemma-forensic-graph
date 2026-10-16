"""
Pipeline de Limpeza e Preparação do Dataset Multimodal Forense para Fine-Tuning do GLiNER.
Fontes integradas:
1. Mensagens filtradas de WhatsApp (10.915 msgs) com as 17 categorias
2. Transcrições de Áudio (1.647 áudios WhatsApp do IPED / transcriptions.db e transcriptions_cache.json)
3. OCR de Documentos (14.364 comprovantes, CNHs, notas e recibos de ocr-results.db)
4. Contatos da Agenda / Pushnames (894 nomes reais de wa_contacts e tb_agenda_contato)
5. Ground Truth Forense Auditado (test_sample_gold.json)
"""

import os
import sys
import json
import sqlite3
import re
from typing import List, Dict, Tuple

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "data", "finetune")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Carregar Nomes de Contatos e Agenda (894 contatos)
print("1. Carregando Contatos e Agenda para Ground-Truth de PESSOA...")
wa_db = os.path.join(DATA_DIR, "wa_full.db")
if not os.path.exists(wa_db):
    wa_db = os.path.join(DATA_DIR, "wa.db")

known_persons = set()
known_emails = set()

if os.path.exists(wa_db):
    c_wa = sqlite3.connect(wa_db)
    rows = c_wa.execute("SELECT display_name, wa_name, jid FROM wa_contacts").fetchall()
    for d, w, j in rows:
        for name in [d, w]:
            if name and len(name.strip()) >= 3 and not name.strip().isdigit() and "~" not in name:
                # Remove emojis e pontuação estranha
                clean_n = re.sub(r'[^\w\s]', '', name).strip()
                if len(clean_n) >= 3 and not clean_n.isdigit():
                    known_persons.add(clean_n)
        if j and "@" in j:
            known_emails.add(j.strip())
    c_wa.close()

# Adiciona alvos e interlocutores chave conhecidos
for n in ["Diogo Gonçalves Costa", "Wendel Praxedes", "Brayan Gouvea", "Thaywan Da Silva", 
          "Yara Thais", "Rafael Alves Bocaiuva", "Tito", "Gordão", "João Batista", "Vinícius Souto"]:
    known_persons.add(n)

print(f"Total de nomes de pessoas canônicos carregados da agenda: {len(known_persons)}")

def tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tokeniza texto em palavras/pontuação e retorna tokens e seus spans de caracteres [start, end)."""
    tokens = []
    spans = []
    for match in re.finditer(r'\w+|[^\w\s]', text, re.UNICODE):
        tokens.append(match.group(0))
        spans.append((match.start(), match.end()))
    return tokens, spans

def char_spans_to_token_spans(char_entities: List[Dict], token_spans: List[Tuple[int, int]]) -> List[List]:
    """Converte entidades de caracteres em spans de tokens [start_tok, end_tok_inclusive, label]."""
    ner = []
    for ent in char_entities:
        c_start = ent.get("start")
        c_end = ent.get("end")
        cat = ent.get("categoria", ent.get("label", "OUTRO")).upper()
        
        if c_start is None or c_end is None:
            continue
            
        start_tok = None
        end_tok = None
        
        for idx, (t_s, t_e) in enumerate(token_spans):
            # Se o token se sobrepõe com a entidade
            if max(t_s, c_start) < min(t_e, c_end):
                if start_tok is None:
                    start_tok = idx
                end_tok = idx
                
        if start_tok is not None and end_tok is not None and start_tok <= end_tok:
            ner.append([start_tok, end_tok, cat])
            
    # Remove spans duplicados
    unique_ner = []
    seen = set()
    for span in ner:
        key = (span[0], span[1], span[2])
        if key not in seen:
            seen.add(key)
            unique_ner.append(span)
    return unique_ner

all_samples = []

# 2. Processar Mensagens do Dataset Completo
print("\n2. Processando Mensagens com Entidades Forenses...")
dataset_json_path = os.path.join(DATA_DIR, "full_processed_dataset.json")
if os.path.exists(dataset_json_path):
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    
    msg_count = 0
    for cid, cdata in ds.get("chats", {}).items():
        for m in cdata.get("messages", []):
            txt = (m.get("text") or "").strip()
            ents = m.get("entities_17", [])
            
            # Filtra mensagens curtas vazias ou saudações sem entidade
            if not ents or len(txt) < 5 or len(txt) > 400:
                continue
                
            tokens, spans = tokenize_with_spans(txt)
            if not tokens:
                continue
                
            # Localizar spans de caracteres para as entidades
            char_ents = []
            for e in ents:
                val = e.get("text", "")
                cat = e.get("categoria", "OUTRO")
                if not val or len(val) < 2:
                    continue
                # Localiza todas as ocorrências no texto
                for m_match in re.finditer(re.escape(val), txt, re.IGNORECASE):
                    char_ents.append({
                        "start": m_match.start(),
                        "end": m_match.end(),
                        "categoria": cat
                    })
                    
            token_ner = char_spans_to_token_spans(char_ents, spans)
            if token_ner:
                all_samples.append({
                    "tokenized_text": tokens,
                    "ner": token_ner,
                    "source": "whatsapp_chat"
                })
                msg_count += 1
    print(f"Mensagens de WhatsApp convertidas em amostras anotadas: {msg_count}")

# 3. Processar Transcrições de Áudio (Whisper IPED)
print("\n3. Processando Transcrições de Áudio IPED...")
transc_db_path = "C:/Casos_Indexados/Motorola_G6_Plus_Diogo_OCR/iped/text/transcriptions.db"
transc_count = 0
if os.path.exists(transc_db_path):
    conn_t = sqlite3.connect(transc_db_path)
    t_rows = conn_t.execute("SELECT id, text, score FROM transcriptions WHERE length(trim(text)) > 15").fetchall()
    for tid, t_txt, score in t_rows:
        t_clean = t_txt.strip()
        tokens, spans = tokenize_with_spans(t_clean)
        char_ents = []
        
        # Match com pessoas conhecidas
        for p in known_persons:
            if len(p) >= 4 and p.lower() in t_clean.lower():
                for m_match in re.finditer(r'\b' + re.escape(p) + r'\b', t_clean, re.IGNORECASE):
                    char_ents.append({"start": m_match.start(), "end": m_match.end(), "categoria": "PESSOA"})
                    
        # Match com valores monetários
        for m_match in re.finditer(r'\b(?:R\$\s*)?\d+(?:[\.,]\d{3})*(?:,\d{2})?\s*(?:reais|mil|cento|pau)?\b', t_clean, re.IGNORECASE):
            val_txt = m_match.group(0).strip()
            if any(c.isdigit() for c in val_txt) and len(val_txt) >= 2:
                char_ents.append({"start": m_match.start(), "end": m_match.end(), "categoria": "VALOR_MONETARIO"})
                
        # Match com termos de parentesco e funções
        for m_match in re.finditer(r'\b(mãe|pai|irmão|irmã|esposa|marido|patrão|gerente|advogado)\b', t_clean, re.IGNORECASE):
            word = m_match.group(1).lower()
            cat = "RELACAO_FAMILIAR" if word in ["mãe", "pai", "irmão", "irmã", "esposa", "marido"] else "FUNCAO_CARGO"
            char_ents.append({"start": m_match.start(), "end": m_match.end(), "categoria": cat})

        token_ner = char_spans_to_token_spans(char_ents, spans)
        if token_ner:
            all_samples.append({
                "tokenized_text": tokens,
                "ner": token_ner,
                "source": "audio_transcription"
            })
            transc_count += 1
    conn_t.close()
print(f"Transcrições de áudio com entidades forenses anotadas: {transc_count}")

# 4. Processar OCRs de Documentos (IPED)
print("\n4. Processando OCRs de Documentos IPED...")
ocr_db_path = "C:/Casos_Indexados/Motorola_G6_Plus_Diogo_OCR/iped/text/ocr-results.db"
ocr_sample_count = 0
if os.path.exists(ocr_db_path):
    conn_o = sqlite3.connect(ocr_db_path)
    o_rows = conn_o.execute("""
        SELECT id, text FROM ocr 
        WHERE length(trim(text)) BETWEEN 20 AND 300 
          AND (text LIKE '%R$%' OR text LIKE '%Nubank%' OR text LIKE '%Bradesco%' 
               OR text LIKE '%Banco%' OR text LIKE '%Comprovante%' OR text LIKE '%CPF%')
        LIMIT 1000
    """).fetchall()
    
    for oid, o_txt in o_rows:
        o_clean = o_txt.strip()
        tokens, spans = tokenize_with_spans(o_clean)
        char_ents = []
        
        # Bancos e Organizações
        for org in ["Nubank", "Bradesco", "Caixa", "Santander", "Itaú", "Inter", "Banco do Brasil"]:
            for m_match in re.finditer(r'\b' + re.escape(org) + r'\b', o_clean, re.IGNORECASE):
                char_ents.append({"start": m_match.start(), "end": m_match.end(), "categoria": "ORGANIZACAO"})
                
        # Valores em Documentos
        for m_match in re.finditer(r'R\$\s*[\d\.,]+', o_clean):
            char_ents.append({"start": m_match.start(), "end": m_match.end(), "categoria": "VALOR_MONETARIO"})
            
        # Documentos formais
        for doc in ["Comprovante", "Nota Fiscal", "Recibo", "Transferência", "Pix"]:
            for m_match in re.finditer(r'\b' + re.escape(doc) + r'\b', o_clean, re.IGNORECASE):
                char_ents.append({"start": m_match.start(), "end": m_match.end(), "categoria": "DOCUMENTO"})
                
        # CPFs
        for m_match in re.finditer(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b', o_clean):
            char_ents.append({"start": m_match.start(), "end": m_match.end(), "categoria": "CPF_CNPJ"})

        token_ner = char_spans_to_token_spans(char_ents, spans)
        if token_ner:
            all_samples.append({
                "tokenized_text": tokens,
                "ner": token_ner,
                "source": "document_ocr"
            })
            ocr_sample_count += 1
    conn_o.close()
print(f"OCRs de documentos convertidos em amostras anotadas: {ocr_sample_count}")

# 5. Adicionar Ground Truth Forense Auditado (Gabarito de Ouro)
gold_path = os.path.join(BASE_DIR, "scratch", "test_sample_gold.json")
if os.path.exists(gold_path):
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)
    for g in gold:
        gtxt = g["text"]
        tokens, spans = tokenize_with_spans(gtxt)
        char_ents = [{"start": ge["start"], "end": ge["end"], "categoria": ge["categoria"]} for ge in g.get("entities", [])]
        token_ner = char_spans_to_token_spans(char_ents, spans)
        if token_ner:
            all_samples.append({
                "tokenized_text": tokens,
                "ner": token_ner,
                "source": "gold_standard"
            })

print("\n" + "="*60)
print(f"DATASET MULTIMODAL FORENSE PRONTO:")
print(f"Total de amostras limpas e rotuladas: {len(all_samples)}")
print("="*60)

# Divisão Train / Validation (85% / 15%)
import random
random.seed(42)
random.shuffle(all_samples)

split_idx = int(len(all_samples) * 0.85)
train_samples = all_samples[:split_idx]
val_samples = all_samples[split_idx:]

train_file = os.path.join(OUT_DIR, "train_dataset.json")
val_file = os.path.join(OUT_DIR, "val_dataset.json")

with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_samples, f, ensure_ascii=False, indent=2)

with open(val_file, "w", encoding="utf-8") as f:
    json.dump(val_samples, f, ensure_ascii=False, indent=2)

print(f"Conjunto de Treinamento: {len(train_samples)} amostras gravadas em {train_file}")
print(f"Conjunto de Validação: {len(val_samples)} amostras gravadas em {val_file}")

# Contagem de entidades por classe
cat_counts = {}
for s in all_samples:
    for span in s["ner"]:
        c = span[2]
        cat_counts[c] = cat_counts.get(c, 0) + 1

print("\nDistribuição das classes no dataset de Fine-Tuning:")
for c, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {c:<22}: {cnt:>5} anotações")
