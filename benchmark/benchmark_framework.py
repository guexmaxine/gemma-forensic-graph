"""
Benchmark Comparativo de NER Forense:
1. Baseline Regex / Heurístico Puro
2. spaCy pt_core_news_sm
3. GLiNER Multilingual (urchade/gliner_multi-v2.1)
4. Solução Híbrida Ideal (GLiNER + Regex Estrito + Filtro Anti-Falso Positivo)
"""

import time
import re
import json
from typing import List, Dict, Any

# 1. Amostra com Ground Truth (Gabarito Manual Auditado)
TEST_SAMPLE = [
    {
        "id": 1,
        "text": "148.028.237-54 Wendel praxedes Pix : Nubank",
        "gold": [
            {"text": "148.028.237-54", "categoria": "CPF_CNPJ"},
            {"text": "Wendel praxedes", "categoria": "PESSOA"},
            {"text": "Nubank", "categoria": "ORGANIZACAO"}
        ]
    },
    {
        "id": 2,
        "text": "Chave pix (27) 992234022 nubank Brayan Gouvea Andrade",
        "gold": [
            {"text": "(27) 992234022", "categoria": "TELEFONE"},
            {"text": "nubank", "categoria": "ORGANIZACAO"},
            {"text": "Brayan Gouvea Andrade", "categoria": "PESSOA"}
        ]
    },
    {
        "id": 3,
        "text": "Rafael Alves Bocaiuva 27998569389 Pix Bradesco",
        "gold": [
            {"text": "Rafael Alves Bocaiuva", "categoria": "PESSOA"},
            {"text": "27998569389", "categoria": "TELEFONE"},
            {"text": "Bradesco", "categoria": "ORGANIZACAO"}
        ]
    },
    {
        "id": 4,
        "text": "Assino a carteira de todos na minha empresa! Com 5 dias dou baixa E daqui do banco mesmo Eu libero o seguro desemprego",
        "gold": [
            {"text": "carteira", "categoria": "DOCUMENTO"},
            {"text": "5 dias", "categoria": "DATA_HORA"},
            {"text": "banco", "categoria": "ORGANIZACAO"},
            {"text": "seguro desemprego", "categoria": "TRIBUTO"}
        ]
    },
    {
        "id": 5,
        "text": "50 pessoas 50mil por mês No caso a pessoa fica recebendo por 4 messes",
        "gold": [
            {"text": "50mil", "categoria": "VALOR_MONETARIO"},
            {"text": "4 messes", "categoria": "DATA_HORA"}
        ]
    },
    {
        "id": 6,
        "text": "Boa noite, tudo bem por aí?",
        "gold": []  # "noite" em "Boa noite" é saudação, NÃO é entidade forense!
    },
    {
        "id": 7,
        "text": "estarei em casa só dia 13 às 15:30",
        "gold": [
            {"text": "dia 13 às 15:30", "categoria": "DATA_HORA"}
        ]
    },
    {
        "id": 8,
        "text": "Thaywan Da Silva Lemos Chaves 141.207.447-95 31/03/1998 Branco 21540-020 Rua dos topázios, 434 casa 2 - Rocha Miranda",
        "gold": [
            {"text": "Thaywan Da Silva Lemos Chaves", "categoria": "PESSOA"},
            {"text": "141.207.447-95", "categoria": "CPF_CNPJ"},
            {"text": "31/03/1998", "categoria": "DATA_HORA"},
            {"text": "Rua dos topázios, 434", "categoria": "LOCAL"},
            {"text": "Rocha Miranda", "categoria": "LOCAL"}
        ]
    },
    {
        "id": 9,
        "text": "Yara Thais Lauriano Honorato 225.292.777-10 18/11/2003 Rua Mariana Do Nascimento, Nº 91",
        "gold": [
            {"text": "Yara Thais Lauriano Honorato", "categoria": "PESSOA"},
            {"text": "225.292.777-10", "categoria": "CPF_CNPJ"},
            {"text": "18/11/2003", "categoria": "DATA_HORA"},
            {"text": "Rua Mariana Do Nascimento, Nº 91", "categoria": "LOCAL"}
        ]
    },
    {
        "id": 10,
        "text": "Fiz o pix de 1500 pra conta do João Batista no Nubank em Vitória ontem",
        "gold": [
            {"text": "1500", "categoria": "VALOR_MONETARIO"},
            {"text": "João Batista", "categoria": "PESSOA"},
            {"text": "Nubank", "categoria": "ORGANIZACAO"},
            {"text": "Vitória", "categoria": "LOCAL"},
            {"text": "ontem", "categoria": "DATA_HORA"}
        ]
    },
    {
        "id": 11,
        "text": "O celular Motorola Moto G6 Plus de Diogo foi guardado no galpão de Cariacica",
        "gold": [
            {"text": "Motorola Moto G6 Plus", "categoria": "OUTRO"},
            {"text": "Diogo", "categoria": "PESSOA"},
            {"text": "galpão", "categoria": "IMOVEL"},
            {"text": "Cariacica", "categoria": "LOCAL"}
        ]
    },
    {
        "id": 12,
        "text": "Consta no processo criminal nº 0012450-88.2022.8.08.0024 da 2ª Vara Criminal",
        "gold": [
            {"text": "0012450-88.2022.8.08.0024", "categoria": "PROCESSO"},
            {"text": "2ª Vara Criminal", "categoria": "ORGANIZACAO"}
        ]
    },
    {
        "id": 13,
        "text": "Ele veio com o Corolla prata placa RQX4A12 entregar a encomenda",
        "gold": [
            {"text": "Corolla", "categoria": "VEICULO"},
            {"text": "RQX4A12", "categoria": "VEICULO"}
        ]
    },
    {
        "id": 14,
        "text": "Minha mãe e meu irmão já transferiram os R$ 2.300",
        "gold": [
            {"text": "mãe", "categoria": "RELACAO_FAMILIAR"},
            {"text": "irmão", "categoria": "RELACAO_FAMILIAR"},
            {"text": "R$ 2.300", "categoria": "VALOR_MONETARIO"}
        ]
    },
    {
        "id": 15,
        "text": "O patrão mandou o gerente do posto recolher o malote",
        "gold": [
            {"text": "patrão", "categoria": "FUNCAO_CARGO"},
            {"text": "gerente", "categoria": "FUNCAO_CARGO"},
            {"text": "posto", "categoria": "IMOVEL"}
        ]
    },
    {
        "id": 16,
        "text": "Pagou a guia do DARF e a taxa de custas do cartório?",
        "gold": [
            {"text": "DARF", "categoria": "TRIBUTO"},
            {"text": "taxa de custas", "categoria": "TRIBUTO"},
            {"text": "cartório", "categoria": "ORGANIZACAO"}
        ]
    },
    {
        "id": 17,
        "text": "Já aumenta o limite pix de todas Para 200mil",
        "gold": [
            {"text": "200mil", "categoria": "VALOR_MONETARIO"}
        ]
    },
    {
        "id": 18,
        "text": "Manda o comprovante pro email wendelpraxedes180@gmail.com",
        "gold": [
            {"text": "comprovante", "categoria": "DOCUMENTO"},
            {"text": "wendelpraxedes180@gmail.com", "categoria": "EMAIL"}
        ]
    },
    {
        "id": 19,
        "text": "A reunião com os advogados do escritório Infortec foi remarcada para amanhã",
        "gold": [
            {"text": "advogados", "categoria": "FUNCAO_CARGO"},
            {"text": "Infortec", "categoria": "ORGANIZACAO"},
            {"text": "amanhã", "categoria": "DATA_HORA"}
        ]
    },
    {
        "id": 20,
        "text": "Tito falou que o Gordão da van vai levar o dinheiro",
        "gold": [
            {"text": "Tito", "categoria": "PESSOA"},
            {"text": "Gordão", "categoria": "PESSOA"},
            {"text": "van", "categoria": "VEICULO"},
            {"text": "dinheiro", "categoria": "VALOR_MONETARIO"}
        ]
    }
]

print(f"Total de frases de teste: {len(TEST_SAMPLE)}")
total_gold = sum(len(s["gold"]) for s in TEST_SAMPLE)
print(f"Total de entidades Gold Standard: {total_gold}")

# 2. Avaliador de Precisão, Recall e F1
def evaluate_method(predictions: List[List[Dict[str, str]]], method_name: str, exec_time_ms: float):
    tp = 0
    fp = 0
    fn = 0
    
    for i, s in enumerate(TEST_SAMPLE):
        gold_items = s["gold"]
        pred_items = predictions[i]
        
        # Match por sobreposição de texto ou equivalência de categoria
        matched_gold = set()
        matched_pred = set()
        
        for p_idx, p in enumerate(pred_items):
            p_text = p["text"].lower().strip()
            p_cat = p["categoria"].upper().strip()
            
            found = False
            for g_idx, g in enumerate(gold_items):
                if g_idx in matched_gold:
                    continue
                g_text = g["text"].lower().strip()
                g_cat = g["categoria"].upper().strip()
                
                # Critério de match: texto contido/igual e categoria compatível
                if (p_text in g_text or g_text in p_text) and (p_cat == g_cat or p_cat == "OUTRO" or g_cat == "OUTRO"):
                    matched_gold.add(g_idx)
                    matched_pred.add(p_idx)
                    tp += 1
                    found = True
                    break
            if not found:
                fp += 1
                
        fn += (len(gold_items) - len(matched_gold))
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "metodo": method_name,
        "tempo_ms": round(exec_time_ms, 2),
        "ms_por_frase": round(exec_time_ms / len(TEST_SAMPLE), 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "f1_score": round(f1 * 100, 1)
    }

# Salva amostra para outros scripts
with open('scratch/test_sample_gold.json', 'w', encoding='utf-8') as f:
    json.dump(TEST_SAMPLE, f, ensure_ascii=False, indent=2)
print("Amostra Gold Standard salva em scratch/test_sample_gold.json")
