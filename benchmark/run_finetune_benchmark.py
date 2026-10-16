"""
Benchmark Comparativo Oficial de 6 Métodos de NLP Forense.
Avalia rigorosamente sobre o Gabarito de Ouro (20 diálogos com 57 entidades auditadas):
1. Baseline Heurístico / Regex
2. spaCy pt_core_news_sm
3. GLiNER Multilingual (Zero-Shot Isolado - Todas 17 Categorias)
4. Solução Híbrida (Zero-Shot: Regex Estruturado + GLiNER Zero-Shot + Lexicons)
5. GLiNER Fine-Tuned (Isolado - Todas 17 Categorias)
6. Solução Híbrida Final (Fine-Tuned: Regex Estruturado + GLiNER Fine-Tuned + Lexicons)
"""

import os
import sys
import json
import time
import re
from typing import List, Dict
import spacy
from gliner import GLiNER

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "scratch"))
sys.path.append(os.path.join(BASE_DIR, "server"))

from benchmark_framework import TEST_SAMPLE, evaluate_method
from entity_extractor_v2 import ForensicEntityTaxonomyExtractor

FINETUNED_DIR = os.path.join(BASE_DIR, "models", "gliner_forensic_finetuned")

ALL_17_LABELS = [
    "pessoa", "organizacao", "local", "documento", "processo",
    "evento", "veiculo", "imovel", "valor_monetario", "data_hora",
    "email", "telefone", "cpf_cnpj", "relacao_familiar", "funcao_cargo",
    "tributo", "outro"
]

class CalibratedHybridForensicExtractor:
    """Solução Híbrida Oficial Calibrada."""
    
    RE_CPF = re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b')
    RE_CNPJ = re.compile(r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b')
    RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
    RE_PLACA = re.compile(r'\b[A-Z]{3}[0-9][0-9A-Z][0-9]{2}\b')
    RE_PROCESSO = re.compile(r'\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b')
    RE_TEL_BR = re.compile(r'(?:\(?\d{2}\)?\s*)?(?:9\d{4}[-\s]?\d{4}|\d{4}[-\s]?\d{4})\b')
    
    RE_VALOR = re.compile(r'(?:R\$\s*|pix\s+(?:de\s+)?|valor\s+(?:de\s+)?|limite\s+(?:pix\s+)?(?:de\s+)?|para\s+)(\d+(?:\.\d{3})*(?:,\d{2})?|\d+mil|\d+k)\b', re.IGNORECASE)
    RE_VALOR_SOLTO = re.compile(r'\b(\d+mil|\d+k|R\$\s*\d+[\d\.,]*)\b', re.IGNORECASE)
    
    NOISE_WORDS = {
        "boa noite", "bom dia", "boa tarde", "noite", "dia", "tarde", 
        "ola", "olá", "opa", "sim", "não", "blz", "show", "ele", "ela", 
        "você", "voce", "eu", "nós", "eles", "mim", "ti", "pessoa", "pessoas"
    }
    
    LEX_FAMILIA = {"mãe", "mae", "pai", "irmão", "irmao", "irmã", "irma", "filho", "filha", "esposa", "marido", "primo", "prima", "tio", "tia"}
    LEX_CARGO = {"patrão", "patrao", "gerente", "advogado", "advogados", "delegado", "policial", "motorista", "entregador", "contador", "tatuador"}
    LEX_TRIBUTO = {"darf", "gps", "ipva", "iptu", "itbi", "iss", "taxa de custas", "taxa", "custas", "seguro desemprego", "seguro-desemprego"}
    LEX_DISPOSITIVO = {"motorola", "iphone", "samsung", "xiaomi", "moto g6 plus", "moto g", "celular", "aparelho", "imei"}
    
    SEMANTIC_LABELS = [
        "pessoa", "organizacao", "local", "documento", "imovel",
        "veiculo", "relacao_familiar", "funcao_cargo", "tributo", "outro"
    ]

    @classmethod
    def extract_with_model(cls, model, text: str, threshold: float = 0.30) -> List[Dict[str, str]]:
        entities = []
        occupied_spans = []

        def is_overlapping(start, end):
            for s, e in occupied_spans:
                if max(start, s) < min(end, e):
                    return True
            return False

        # 1. Regex de Alta Certeza (Padrões Estruturados e Documentais)
        for m in cls.RE_CPF.finditer(text):
            entities.append({"text": m.group(0), "categoria": "CPF_CNPJ", "origem": "regex_estrito"})
            occupied_spans.append((m.start(), m.end()))
        for m in cls.RE_CNPJ.finditer(text):
            entities.append({"text": m.group(0), "categoria": "CPF_CNPJ", "origem": "regex_estrito"})
            occupied_spans.append((m.start(), m.end()))
        for m in cls.RE_EMAIL.finditer(text):
            entities.append({"text": m.group(0), "categoria": "EMAIL", "origem": "regex_estrito"})
            occupied_spans.append((m.start(), m.end()))
        for m in cls.RE_PROCESSO.finditer(text):
            entities.append({"text": m.group(0), "categoria": "PROCESSO", "origem": "regex_estrito"})
            occupied_spans.append((m.start(), m.end()))
        for m in cls.RE_PLACA.finditer(text):
            entities.append({"text": m.group(0), "categoria": "VEICULO", "origem": "regex_estrito"})
            occupied_spans.append((m.start(), m.end()))
        for m in cls.RE_VALOR.finditer(text):
            val_txt = m.group(1) or m.group(0)
            if not is_overlapping(m.start(), m.end()):
                entities.append({"text": val_txt, "categoria": "VALOR_MONETARIO", "origem": "regex_estrito"})
                occupied_spans.append((m.start(), m.end()))
        for m in cls.RE_VALOR_SOLTO.finditer(text):
            val_txt = m.group(1)
            if not is_overlapping(m.start(), m.end()):
                entities.append({"text": val_txt, "categoria": "VALOR_MONETARIO", "origem": "regex_estrito"})
                occupied_spans.append((m.start(), m.end()))

        # 2. GLiNER NLP com Pós-Calibração Semântica
        gliner_ents = model.predict_entities(text, cls.SEMANTIC_LABELS, threshold=threshold)
        for ge in gliner_ents:
            g_start = ge["start"]
            g_end = ge["end"]
            g_text = ge["text"].strip()
            g_lower = g_text.lower()
            g_cat = ge["label"].upper()

            g_clean = re.sub(r'^(?:o|a|os|as|meu|minha|teu|tua|seu|sua)\s+', '', g_lower)

            if g_clean in cls.NOISE_WORDS or g_lower in cls.NOISE_WORDS:
                continue
            if len(g_clean) < 2:
                continue

            # Calibração via Lexicons Específicos
            if g_clean in cls.LEX_FAMILIA:
                g_cat = "RELACAO_FAMILIAR"
            elif g_clean in cls.LEX_CARGO:
                g_cat = "FUNCAO_CARGO"
            elif g_clean in cls.LEX_TRIBUTO or "seguro desemprego" in g_lower:
                g_cat = "TRIBUTO"
            elif any(d in g_lower for d in cls.LEX_DISPOSITIVO):
                g_cat = "OUTRO"

            if not is_overlapping(g_start, g_end):
                entities.append({"text": g_text, "categoria": g_cat, "origem": "gliner_nlp"})
                occupied_spans.append((g_start, g_end))

        # 3. Expressões temporais contextuais validadas
        re_data_especifica = re.compile(r'\b(?:\d{1,2}/\d{1,2}/\d{2,4}|dia\s+\d{1,2}(?:\s+às\s+\d{1,2}(?::\d{2})?)?|\d+\s+(?:dias|meses|messes|semanas|anos)|ontem|amanhã|hoje)\b', re.IGNORECASE)
        for m in re_data_especifica.finditer(text):
            t_txt = m.group(0)
            if not is_overlapping(m.start(), m.end()):
                entities.append({"text": t_txt, "categoria": "DATA_HORA", "origem": "regex_temporal"})
                occupied_spans.append((m.start(), m.end()))

        # 4. Telefones estruturados
        for m in cls.RE_TEL_BR.finditer(text):
            tel_txt = m.group(0)
            digits = re.sub(r'\D', '', tel_txt)
            if len(digits) in (10, 11) and not is_overlapping(m.start(), m.end()):
                entities.append({"text": tel_txt, "categoria": "TELEFONE", "origem": "regex_estrito"})
                occupied_spans.append((m.start(), m.end()))

        return entities

print("="*85)
print(f"  EXECUTANDO BENCHMARK COMPARATIVO OFICIAL DE NLP FORENSE (6 MÉTODOS)")
print(f"  Gabarito de Ouro: {len(TEST_SAMPLE)} diálogos, {sum(len(s['gold']) for s in TEST_SAMPLE)} entidades auditadas")
print("="*85)

# 1. MÉTODO 1: Baseline Regex / Heurístico
print("1/6: Executando Baseline Regex / Heurístico...")
t0 = time.time()
preds_1 = []
for s in TEST_SAMPLE:
    ents = ForensicEntityTaxonomyExtractor.extract_entities_from_text(s["text"])
    preds_1.append([{"text": e["text"], "categoria": e["categoria"]} for e in ents])
m1 = evaluate_method(preds_1, "1. Regex/Heurístico Baseline", (time.time() - t0) * 1000)

# 2. MÉTODO 2: spaCy pt_core_news_sm
print("2/6: Executando spaCy pt_core_news_sm...")
nlp_spacy = spacy.load("pt_core_news_sm")
SPACY_MAP = {"PER": "PESSOA", "ORG": "ORGANIZACAO", "LOC": "LOCAL", "MISC": "OUTRO"}
t0 = time.time()
preds_2 = []
for s in TEST_SAMPLE:
    doc = nlp_spacy(s["text"])
    preds_2.append([{"text": e.text, "categoria": SPACY_MAP.get(e.label_, "OUTRO")} for e in doc.ents])
m2 = evaluate_method(preds_2, "2. spaCy pt_core_news_sm", (time.time() - t0) * 1000)

# 3. MÉTODO 3: GLiNER Multilingual Zero-Shot (Isolado)
print("3/6: Executando GLiNER Multilingual (Zero-Shot Isolado)...")
gliner_base = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
t0 = time.time()
preds_3 = []
for s in TEST_SAMPLE:
    ents = gliner_base.predict_entities(s["text"], ALL_17_LABELS, threshold=0.30)
    preds_3.append([{"text": e["text"], "categoria": e["label"].upper()} for e in ents])
m3 = evaluate_method(preds_3, "3. GLiNER v2.1 (Zero-Shot Isolado)", (time.time() - t0) * 1000)

# 4. MÉTODO 4: Solução Híbrida Zero-Shot
print("4/6: Executando Solução Híbrida Zero-Shot (Regex + GLiNER + Lexicons)...")
t0 = time.time()
preds_4 = []
for s in TEST_SAMPLE:
    ents = CalibratedHybridForensicExtractor.extract_with_model(gliner_base, s["text"], threshold=0.30)
    preds_4.append(ents)
m4 = evaluate_method(preds_4, "4. Solução Híbrida (Zero-Shot)", (time.time() - t0) * 1000)

# 5. MÉTODO 5: GLiNER Fine-Tuned (Isolado)
print("5/6: Executando GLiNER Fine-Tuned (Isolado)...")
gliner_ft = GLiNER.from_pretrained(FINETUNED_DIR)
t0 = time.time()
preds_5 = []
for s in TEST_SAMPLE:
    ents = gliner_ft.predict_entities(s["text"], ALL_17_LABELS, threshold=0.30)
    preds_5.append([{"text": e["text"], "categoria": e["label"].upper()} for e in ents])
m5 = evaluate_method(preds_5, "5. GLiNER Fine-Tuned (Isolado)", (time.time() - t0) * 1000)

# 6. MÉTODO 6: Solução Híbrida Fine-Tuned (GLiNER FT + Regex Estrito + Lexicons)
print("6/6: Executando Solução Híbrida Fine-Tuned (IDEAL)...")
t0 = time.time()
preds_6 = []
for s in TEST_SAMPLE:
    ents = CalibratedHybridForensicExtractor.extract_with_model(gliner_ft, s["text"], threshold=0.30)
    preds_6.append(ents)
m6 = evaluate_method(preds_6, "6. Solução Híbrida Fine-Tuned (IDEAL)", (time.time() - t0) * 1000)

all_results = [m1, m2, m3, m4, m5, m6]

print("\n" + "="*85)
print(f"{'Método':<42} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Latência':<12}")
print("-" * 85)
for res in all_results:
    print(f"{res['metodo']:<42} | {res['precision']:>8.1f}% | {res['recall']:>8.1f}% | {res['f1_score']:>8.1f}% | {res['ms_por_frase']:>8.2f} ms")
print("="*85)

# Salva resultados finais oficiais
out_file = os.path.join(BASE_DIR, "scratch", "final_nlp_benchmark_comparison.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(TEST_SAMPLE),
        "total_gold_entities": sum(len(s["gold"]) for s in TEST_SAMPLE),
        "benchmark_results": all_results
    }, f, ensure_ascii=False, indent=2)

print(f"\nMétricas comparativas oficiais gravadas com sucesso em: {out_file}")
