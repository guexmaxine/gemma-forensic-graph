"""
Extrator Especializado de 17 Categorias de Entidades Forenses.
Categorias estritas:
1. PESSOA
2. ORGANIZACAO
3. LOCAL
4. DOCUMENTO
5. PROCESSO
6. EVENTO
7. VEICULO
8. IMOVEL
9. VALOR_MONETARIO
10. DATA_HORA
11. EMAIL
12. TELEFONE
13. CPF_CNPJ
14. RELACAO_FAMILIAR
15. FUNCAO_CARGO
16. TRIBUTO
17. OUTRO
"""

import re
import datetime
from typing import List, Dict, Any, Tuple

# Dicionários e Padrões Regex Forenses de Alta Precisão
REGEX_CPF = re.compile(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b')
REGEX_CNPJ = re.compile(r'\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b')
REGEX_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
REGEX_PHONE = re.compile(r'(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\s?)?\d{4}[-\s]?\d{4}\b')
REGEX_MONEY = re.compile(r'(?:R\$\s*|BRL\s*|\bvalor(?:\s*de)?\s*)(\d{1,3}(?:\.\d{3})*(?:,\d{2})?|\d+(?:[.,]\d{2})?)|(?:\b)(\d{2,6})\s*(?:reais|conto|pila|mil\s*reais|\bk\b)', re.IGNORECASE)
REGEX_DATE_TIME = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}h(?:\d{2})?|ontem|hoje|amanh[aã]|ter[cç]a|quarta|quinta|sexta|s[aá]bado|domingo|segunda|semana\s*passada|madrugada|noite|tarde|manh[aã])\b', re.IGNORECASE)
REGEX_VEICULO = re.compile(r'\b(van|moto|motoboy|carro|caminh[aã]o|ve[ií]culo|placa\s*[A-Z]{3}-?\d[A-Z0-9]\d{2}|gol|corolla|hilux|civic|onix)\b', re.IGNORECASE)
REGEX_PROCESSO = re.compile(r'\b(processo|autos|inqu[eé]rito|ip\s*n[ºo]?|mando|mandado|vara|gaeco|of[ií]cio|audi[eê]ncia)\b', re.IGNORECASE)
REGEX_DOCUMENTO = re.compile(r'\b(comprovante|recibo|nota\s*fiscal|nf|danfe|print|cnh|rg|cpf|contrato|documento|foto\s*do\s*imei|laudo)\b', re.IGNORECASE)
REGEX_IMOVEL = re.compile(r'\b(im[oó]vel|galp[aã]o|apartamento|apto|casa|sede|loja\s*f[ií]sica|armaz[eé]m|dep[oó]sito|ponto\s*comercial)\b', re.IGNORECASE)
REGEX_TRIBUTO = re.compile(r'\b(imposto|tributo|icms|iss|darf|taxa|multa|iof|receita\s*federal|alf[aâ]ndega)\b', re.IGNORECASE)
REGEX_RELACAO_FAMILIAR = re.compile(r'\b(m[aã]e|pai|irm[aã]o|irm[aã]|esposa|marido|filho|filha|sobrinho|sobrinha|cunhado|cunhada|tio|tia|primo|prima|namorada|namorado)\b', re.IGNORECASE)
REGEX_FUNCAO_CARGO = re.compile(r'\b(contador|advogado|adv|policial|delegado|chefe|patr[aã]o|gerente|operador|fornecedor|laranja|mandante|subordinado|cliente|vendedor|motoboy|piloto)\b', re.IGNORECASE)

KNOWN_ORGS = [
    "infortec", "apple", "nubank", "caixa econômica", "santander", "banco do brasil",
    "itaú", "bradesco", "mercado livre", "cdtvv", "polícia civil", "polícia federal",
    "gaeco", "loja do celular", "jack celulares atacado", "whatsapp"
]

KNOWN_LOCAIS = [
    "miami", "vitória", "vila velha", "serra", "cariacica", "são paulo", "belo horizonte",
    "posto", "galpão", "loja", "obra", "porto", "aeroporto", "rua", "avenida", "bairro"
]

class ForensicEntityTaxonomyExtractor:
    """Extrai e classifica termos rigorosamente nas 17 categorias de entidades forenses."""

    @staticmethod
    def extract_entities_from_text(text: str, anchor_time: str = None) -> List[Dict[str, Any]]:
        entities = []
        found_spans = []

        def add_entity(val: str, category: str, start: int, end: int, confidence: float = 0.95, extra: Dict = None):
            # Evita sobreposição direta
            val_clean = val.strip(" .,;:()[]{}")
            if len(val_clean) < 2:
                return
            for s, e in found_spans:
                if (start >= s and start < e) or (end > s and end <= e):
                    return
            found_spans.append((start, end))
            ent = {
                "text": val_clean,
                "categoria": category,
                "start": start,
                "end": end,
                "confidence": confidence,
                "metadata": extra or {}
            }
            entities.append(ent)

        # 1. CPF_CNPJ
        for m in REGEX_CNPJ.finditer(text):
            add_entity(m.group(), "CPF_CNPJ", m.start(), m.end(), 1.0, {"tipo": "CNPJ"})
        for m in REGEX_CPF.finditer(text):
            add_entity(m.group(), "CPF_CNPJ", m.start(), m.end(), 1.0, {"tipo": "CPF"})

        # 2. EMAIL
        for m in REGEX_EMAIL.finditer(text):
            add_entity(m.group(), "EMAIL", m.start(), m.end(), 1.0)

        # 3. VALOR_MONETARIO
        for m in REGEX_MONEY.finditer(text):
            raw = m.group(1) or m.group(2)
            add_entity(m.group(), "VALOR_MONETARIO", m.start(), m.end(), 0.98, {"valor_bruto": raw})

        # 4. TELEFONE
        for m in REGEX_PHONE.finditer(text):
            digits = re.sub(r'\D', '', m.group())
            if len(digits) in [8, 9, 10, 11, 12, 13]:
                add_entity(m.group(), "TELEFONE", m.start(), m.end(), 0.92, {"digits": digits})

        # 5. DATA_HORA
        for m in REGEX_DATE_TIME.finditer(text):
            add_entity(m.group(), "DATA_HORA", m.start(), m.end(), 0.90, {"ancora": anchor_time})

        # 6. VEICULO
        for m in REGEX_VEICULO.finditer(text):
            add_entity(m.group(), "VEICULO", m.start(), m.end(), 0.92)

        # 7. PROCESSO
        for m in REGEX_PROCESSO.finditer(text):
            add_entity(m.group(), "PROCESSO", m.start(), m.end(), 0.90)

        # 8. DOCUMENTO
        for m in REGEX_DOCUMENTO.finditer(text):
            add_entity(m.group(), "DOCUMENTO", m.start(), m.end(), 0.92)

        # 9. IMOVEL
        for m in REGEX_IMOVEL.finditer(text):
            add_entity(m.group(), "IMOVEL", m.start(), m.end(), 0.90)

        # 10. TRIBUTO
        for m in REGEX_TRIBUTO.finditer(text):
            add_entity(m.group(), "TRIBUTO", m.start(), m.end(), 0.90)

        # 11. RELACAO_FAMILIAR
        for m in REGEX_RELACAO_FAMILIAR.finditer(text):
            add_entity(m.group(), "RELACAO_FAMILIAR", m.start(), m.end(), 0.94)

        # 12. FUNCAO_CARGO
        for m in REGEX_FUNCAO_CARGO.finditer(text):
            add_entity(m.group(), "FUNCAO_CARGO", m.start(), m.end(), 0.92)

        # 13. ORGANIZACAO
        t_low = text.lower()
        for org in KNOWN_ORGS:
            idx = 0
            while True:
                pos = t_low.find(org, idx)
                if pos == -1:
                    break
                orig_text = text[pos : pos + len(org)]
                add_entity(orig_text, "ORGANIZACAO", pos, pos + len(org), 0.95)
                idx = pos + len(org)

        # 14. LOCAL
        for loc in KNOWN_LOCAIS:
            m = re.search(r'\b' + re.escape(loc) + r'\b', text, re.IGNORECASE)
            if m:
                add_entity(m.group(), "LOCAL", m.start(), m.end(), 0.90)

        # 15. PESSOA (Nomes Próprios e Vulgos Conhecidos da Investigação)
        KNOWN_PESSOAS = [
            "Diogo Gonçalves Costa", "Diogo Gonçalves", "Diogo",
            "João Batista de Almeida", "João Batista", "João do Posto", "Gordão", "Gordo da van", "Gordo",
            "Roberto Carlos Vieira", "Roberto Carlos", "Berto Infortec", "Berto",
            "Wendel Santos Silva", "Wendel", "El Cuervo",
            "Tito", "Ceará 02", "Ceará", "Xaropinho", "Ty", "Fabian",
            "Vinícius Souto", "Vinícius", "Ana Amorim", "Karolyne Coelho",
            "Lenimar Zaché", "Jorge", "Sapo Boi", "Alemão", "Beto", "Patrão"
        ]
        for pes in KNOWN_PESSOAS:
            m = re.search(r'\b' + re.escape(pes) + r'\b', text, re.IGNORECASE)
            if m:
                add_entity(m.group(), "PESSOA", m.start(), m.end(), 0.96)

        # 16. OUTRO (Dispositivos, IMEIs, Produtos, Cripto)
        REGEX_IMEI = re.compile(r'\b(imei\s*[:\s]?\s*\d{14,17}|iphone\s*(?:11|12|13|14|15|x|xr|xs|8|7|6)(?:\s*(?:pro|max|plus|gold))?)\b', re.IGNORECASE)
        for m in REGEX_IMEI.finditer(text):
            add_entity(m.group(), "OUTRO", m.start(), m.end(), 0.95, {"subcategoria": "DISPOSITIVO_HARDWARE"})

        # Ordenar por ordem de aparição no texto
        entities.sort(key=lambda x: x["start"])
        return entities
