"""
Execução em Lote com Solução Híbrida Calibrada (GLiNER + Regex Validado + Lexicon Forense)
Processa 100% das mensagens de todas as conversas do caso Motorola (Diogo Gonçalves Costa).
"""

import os
import sys
import json
import time
import re
from typing import List, Dict, Any
from gliner import GLiNER

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "full_processed_dataset.json")

print("===================================================================")
print("  PROCESSAMENTO GLOBAL DE NER FORENSE HÍBRIDO (GLiNER + REGEX)")
print("===================================================================")

print(f"Carregando dataset de {DATASET_PATH}...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

total_chats = len(dataset["chats"])
total_msgs = sum(len(c["messages"]) for c in dataset["chats"].values())
print(f"Dataset carregado: {total_msgs} mensagens em {total_chats} conversas!")

print("Carregando modelo GLiNER Multilingual (urchade/gliner_multi-v2.1)...")
model = GLiNER.from_pretrained('urchade/gliner_multi-v2.1', local_files_only=True)
print("Modelo GLiNER carregado com sucesso na memória.")

class CalibratedHybridExtractor:
    RE_CPF = re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b')
    RE_CNPJ = re.compile(r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b')
    RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
    RE_PLACA = re.compile(r'\b[A-Z]{3}[0-9][0-9A-Z][0-9]{2}\b')
    RE_PROCESSO = re.compile(r'\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b')
    RE_TEL_BR = re.compile(r'(?:\(?\d{2}\)?\s*)?(?:9\d{4}[-\s]?\d{4}|\d{4}[-\s]?\d{4})\b')
    RE_VALOR = re.compile(r'(?:R\$\s*|pix\s+(?:de\s+)?|valor\s+(?:de\s+)?|limite\s+(?:pix\s+)?(?:de\s+)?|para\s+)(\d+(?:\.\d{3})*(?:,\d{2})?|\d+mil|\d+k)\b', re.IGNORECASE)
    RE_VALOR_SOLTO = re.compile(r'\b(\d+mil|\d+k|R\$\s*\d+[\d\.,]*)\b', re.IGNORECASE)
    RE_DATA = re.compile(r'\b(?:\d{1,2}/\d{1,2}/\d{2,4}|dia\s+\d{1,2}(?:\s+às\s+\d{1,2}(?::\d{2})?)?|\d+\s+(?:dias|meses|messes|semanas|anos)|ontem|amanhã|hoje)\b', re.IGNORECASE)
    
    NOISE_WORDS = {
        "boa noite", "bom dia", "boa tarde", "noite", "dia", "tarde", 
        "ola", "olá", "opa", "sim", "não", "blz", "show", "ele", "ela", 
        "você", "voce", "eu", "nós", "eles", "mim", "ti", "pessoa", "pessoas",
        "ok", "kkk", "valeu", "tmj", "obrigado", "obrigada", "blz?", "tudo bem"
    }
    
    LEX_FAMILIA = {"mãe", "mae", "pai", "irmão", "irmao", "irmã", "irma", "filho", "filha", "esposa", "marido", "primo", "prima", "tio", "tia"}
    LEX_CARGO = {"patrão", "patrao", "gerente", "advogado", "advogados", "delegado", "policial", "motorista", "entregador", "contador", "tatuador"}
    LEX_TRIBUTO = {"darf", "gps", "ipva", "iptu", "itbi", "iss", "taxa de custas", "taxa", "custas", "seguro desemprego", "seguro-desemprego"}
    LEX_DISPOSITIVO = {"motorola", "iphone", "samsung", "xiaomi", "moto g6 plus", "moto g", "celular", "aparelho", "imei", "tela", "display"}

    SEMANTIC_LABELS = [
        "pessoa", "organizacao", "local", "documento", "imovel",
        "veiculo", "relacao_familiar", "funcao_cargo", "tributo", "outro"
    ]

    @classmethod
    def extract_regex_only(cls, text: str) -> tuple[List[Dict[str, Any]], List[tuple]]:
        ents = []
        spans = []
        if not text:
            return ents, spans
            
        for m in cls.RE_CPF.finditer(text):
            ents.append({"text": m.group(0), "categoria": "CPF_CNPJ", "confidence": 0.99, "origem": "regex"})
            spans.append((m.start(), m.end()))
        for m in cls.RE_CNPJ.finditer(text):
            ents.append({"text": m.group(0), "categoria": "CPF_CNPJ", "confidence": 0.99, "origem": "regex"})
            spans.append((m.start(), m.end()))
        for m in cls.RE_EMAIL.finditer(text):
            ents.append({"text": m.group(0), "categoria": "EMAIL", "confidence": 0.99, "origem": "regex"})
            spans.append((m.start(), m.end()))
        for m in cls.RE_PROCESSO.finditer(text):
            ents.append({"text": m.group(0), "categoria": "PROCESSO", "confidence": 0.98, "origem": "regex"})
            spans.append((m.start(), m.end()))
        for m in cls.RE_PLACA.finditer(text):
            ents.append({"text": m.group(0), "categoria": "VEICULO", "confidence": 0.95, "origem": "regex"})
            spans.append((m.start(), m.end()))
        for m in cls.RE_VALOR.finditer(text):
            val = m.group(1) or m.group(0)
            ents.append({"text": val, "categoria": "VALOR_MONETARIO", "confidence": 0.95, "origem": "regex"})
            spans.append((m.start(), m.end()))
        for m in cls.RE_VALOR_SOLTO.finditer(text):
            val = m.group(1)
            # evita duplicar se já pego
            if not any(max(m.start(), s) < min(m.end(), e) for s, e in spans):
                ents.append({"text": val, "categoria": "VALOR_MONETARIO", "confidence": 0.90, "origem": "regex"})
                spans.append((m.start(), m.end()))
        for m in cls.RE_DATA.finditer(text):
            if not any(max(m.start(), s) < min(m.end(), e) for s, e in spans):
                ents.append({"text": m.group(0), "categoria": "DATA_HORA", "confidence": 0.88, "origem": "regex"})
                spans.append((m.start(), m.end()))
        for m in cls.RE_TEL_BR.finditer(text):
            t_txt = m.group(0)
            digits = re.sub(r'\D', '', t_txt)
            if len(digits) in (10, 11) and not any(max(m.start(), s) < min(m.end(), e) for s, e in spans):
                ents.append({"text": t_txt, "categoria": "TELEFONE", "confidence": 0.92, "origem": "regex"})
                spans.append((m.start(), m.end()))
                
        return ents, spans

# 1. Separar mensagens que necessitam de inferência semântica GLiNER vs. puramente estruturadas/triviais
print("Fase 1: Mapeando mensagens para extração híbrida...")
candidates_for_gliner = [] # lista de (chat_id, msg_idx, text, regex_ents, regex_spans)
total_msgs_processed = 0

for cid, cdata in dataset["chats"].items():
    for midx, m in enumerate(cdata["messages"]):
        total_msgs_processed += 1
        txt = (m.get("text") or "").strip()
        
        # Extrai regex primeiro
        r_ents, r_spans = CalibratedHybridExtractor.extract_regex_only(txt)
        
        # Filtro de relevância para GLiNER:
        # Se mensagem tem mais de 6 caracteres, tem letras e não é ruído trivial, envia para o GLiNER
        txt_lower = txt.lower()
        has_letters = any(c.isalpha() for c in txt)
        is_trivial = txt_lower in CalibratedHybridExtractor.NOISE_WORDS or len(txt) < 5 or not has_letters
        
        if not is_trivial:
            candidates_for_gliner.append({
                "chat_id": cid,
                "msg_idx": midx,
                "text": txt,
                "r_ents": r_ents,
                "r_spans": r_spans
            })
        else:
            # Apenas regex ents
            m["entities_17"] = r_ents

print(f"Total de mensagens: {total_msgs_processed}")
print(f"Mensagens selecionadas para inferência semântica profunda (GLiNER): {len(candidates_for_gliner)}")
print(f"Mensagens tratadas via Regex/Filtro direto (economia de tempo): {total_msgs_processed - len(candidates_for_gliner)}")

# 2. Execução em Lote com GLiNER (batch_size = 48)
BATCH_SIZE = 48
total_batches = (len(candidates_for_gliner) + BATCH_SIZE - 1) // BATCH_SIZE
print(f"\nFase 2: Executando inferência neural em {total_batches} lotes (tamanho={BATCH_SIZE})...")

t_start = time.time()
for b_idx in range(total_batches):
    start_i = b_idx * BATCH_SIZE
    end_i = min(start_i + BATCH_SIZE, len(candidates_for_gliner))
    batch_items = candidates_for_gliner[start_i:end_i]
    batch_texts = [item["text"] for item in batch_items]
    
    # Inferência no lote
    try:
        batch_preds = model.batch_predict_entities(batch_texts, CalibratedHybridExtractor.SEMANTIC_LABELS, threshold=0.30)
    except Exception as e:
        # fallback para predição individual
        batch_preds = [model.predict_entities(t, CalibratedHybridExtractor.SEMANTIC_LABELS, threshold=0.30) for t in batch_texts]
    
    # Processa e calibra predições
    for i, item in enumerate(batch_items):
        cid = item["chat_id"]
        midx = item["msg_idx"]
        txt = item["text"]
        ents = list(item["r_ents"])
        spans = list(item["r_spans"])
        
        preds = batch_preds[i]
        for ge in preds:
            g_start = ge["start"]
            g_end = ge["end"]
            g_text = ge["text"].strip()
            g_lower = g_text.lower()
            g_cat = ge["label"].upper()
            
            g_clean = re.sub(r'^(?:o|a|os|as|meu|minha|teu|tua|seu|sua)\s+', '', g_lower)
            if g_clean in CalibratedHybridExtractor.NOISE_WORDS or g_lower in CalibratedHybridExtractor.NOISE_WORDS:
                continue
            if len(g_clean) < 2:
                continue
                
            # Calibração léxica
            if g_clean in CalibratedHybridExtractor.LEX_FAMILIA:
                g_cat = "RELACAO_FAMILIAR"
            elif g_clean in CalibratedHybridExtractor.LEX_CARGO:
                g_cat = "FUNCAO_CARGO"
            elif g_clean in CalibratedHybridExtractor.LEX_TRIBUTO or "seguro desemprego" in g_lower:
                g_cat = "TRIBUTO"
            elif any(d in g_lower for d in CalibratedHybridExtractor.LEX_DISPOSITIVO):
                g_cat = "OUTRO"
                
            # Sobreposição
            if not any(max(g_start, s) < min(g_end, e) for s, e in spans):
                ents.append({
                    "text": g_text,
                    "categoria": g_cat,
                    "confidence": round(float(ge.get("score", 0.85)), 2),
                    "origem": "gliner_nlp"
                })
                spans.append((g_start, g_end))
                
        dataset["chats"][cid]["messages"][midx]["entities_17"] = ents

    if (b_idx + 1) % 10 == 0 or (b_idx + 1) == total_batches:
        elapsed = time.time() - t_start
        pct = round((b_idx + 1) / total_batches * 100, 1)
        print(f"  Progresso: Lote {b_idx + 1}/{total_batches} ({pct}%) | Tempo decorrido: {round(elapsed, 1)}s")

t_total = time.time() - t_start
print(f"\nInferência neural concluída em {round(t_total, 2)} segundos!")

# 3. Recalcular Estatísticas Globais da Taxonomia de 17 Categorias
print("\nFase 3: Consolidando métricas e salvando dataset atualizado...")
global_counters = {}
total_extracted_entities = 0

for cid, cdata in dataset["chats"].items():
    for m in cdata["messages"]:
        for e in m.get("entities_17", []):
            cat = e["categoria"]
            global_counters[cat] = global_counters.get(cat, 0) + 1
            total_extracted_entities += 1

dataset["metadata"]["total_mensagens"] = total_msgs_processed
dataset["metadata"]["total_entidades_17"] = total_extracted_entities
dataset["metadata"]["taxonomia_contadores"] = global_counters
dataset["metadata"]["modelo_extracao"] = "Solução Híbrida Calibrada (GLiNER Multilingual v2.1 + Regex Estrito)"
dataset["metadata"]["atualizado_em"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("\n" + "="*60)
print(f"  TAXONOMIA FORENSE DE 17 CATEGORIAS ATUALIZADA (GLiNER + REGEX)")
print("="*60)
print(f"Total de Mensagens: {total_msgs_processed}")
print(f"Total de Entidades Identificadas: {total_extracted_entities}")
print("-"*60)
for cat, count in sorted(global_counters.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat:<24}: {count:>6} ocorrências")
print("="*60)

# Salvar resumo em scratch/global_nlp_stats.json
with open(os.path.join(BASE_DIR, "scratch", "global_nlp_stats.json"), "w", encoding="utf-8") as f:
    json.dump({
        "total_mensagens": total_msgs_processed,
        "total_entidades": total_extracted_entities,
        "taxonomia": global_counters,
        "tempo_execucao_s": round(t_total, 2)
    }, f, ensure_ascii=False, indent=2)

print("Processamento global concluído com sucesso!")
