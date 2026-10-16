"""
Pipeline Multi-Agentes de Inteligência Forense com Taxonomia Estrita de 17 Entidades.
Categorias:
PESSOA, ORGANIZACAO, LOCAL, DOCUMENTO, PROCESSO, EVENTO, VEICULO, IMOVEL,
VALOR_MONETARIO, DATA_HORA, EMAIL, TELEFONE, CPF_CNPJ, RELACAO_FAMILIAR,
FUNCAO_CARGO, TRIBUTO, OUTRO
"""

import json
import re
import urllib.request
import urllib.error
import datetime
from typing import Dict, List, Any, Optional

from entity_extractor_v2 import ForensicEntityTaxonomyExtractor

SYSTEM_PROMPT_FORENSE = """Você é o Agente de Inteligência Forense e Análise Criminal do GAECO.
Sua função é analisar a janela de mensagens de WhatsApp e extrair rigorosamente as seguintes 17 CATEGORIAS DE ENTIDADES:
1. PESSOA (nomes de investigados, terceiros, apelidos nominais)
2. ORGANIZACAO (empresas, bancos, facções, órgãos públicos)
3. LOCAL (cidades, postos, bairros, locais de entrega)
4. DOCUMENTO (comprovantes, notas fiscais, prints, contratos)
5. PROCESSO (números de autos, inquéritos, mandados)
6. EVENTO (transações, entregas, reuniões, blitzes, prisões)
7. VEICULO (vans, motos, carros, placas)
8. IMOVEL (galpões, casas, lojas físicas, depósitos)
9. VALOR_MONETARIO (valores em R$, quantias, depósitos)
10. DATA_HORA (datas e expressões temporais ancoradas)
11. EMAIL (endereços de e-mail)
12. TELEFONE (números de telefone e WhatsApp)
13. CPF_CNPJ (CPFs e CNPJs)
14. RELACAO_FAMILIAR (mãe, pai, irmão, esposa, filho)
15. FUNCAO_CARGO (contador, motoboy, chefe, patrão, advogado, vendedor)
16. TRIBUTO (ICMS, DARF, taxas, multas)
17. OUTRO (dispositivos, IMEIs, tokens cripto, cargas)

Retorne estritamente em JSON:
{
  "entidades": [
    {"nome_ou_termo": string, "categoria": "PESSOA|ORGANIZACAO|LOCAL|DOCUMENTO|PROCESSO|EVENTO|VEICULO|IMOVEL|VALOR_MONETARIO|DATA_HORA|EMAIL|TELEFONE|CPF_CNPJ|RELACAO_FAMILIAR|FUNCAO_CARGO|TRIBUTO|OUTRO", "papel_contexto": string}
  ],
  "eventos_transacoes": [
    {"tipo": string, "valor": number|null, "tempo_ancorado": string, "descricao": string}
  ],
  "tags_criminais": ["#lavagem_ocultacao", "#fraude_golpe", "#receptacao_celulares", ...]
}"""

class ForensicAgentPipeline:
    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", model: str = "llama3.2:3b"):
        self.ollama_url = ollama_url
        self.model = model

    def query_llm(self, prompt: str, timeout: int = 45) -> Optional[Dict[str, Any]]:
        url = f"{self.ollama_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FORENSE},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return json.loads(data["message"]["content"])
        except Exception:
            return None

    def rule_based_fallback(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica a taxonomia estrita de 17 entidades diretamente via analisador determinístico."""
        text = chunk["full_text"]
        raw_entities = ForensicEntityTaxonomyExtractor.extract_entities_from_text(text, anchor_time=chunk["start_time"])
        
        entidades = []
        for e in raw_entities:
            entidades.append({
                "nome_ou_termo": e["text"],
                "categoria": e["categoria"],
                "papel_contexto": e.get("metadata", {}).get("subcategoria", e["categoria"]),
                "confidence": e["confidence"]
            })

        # Adicionar tags criminais
        t_low = text.lower()
        tags = []
        if any(w in t_low for w in ["pix", "transfer", "caiu", "comprovante", "pagamento"]):
            tags.append("#lavagem_ocultacao")
        if any(w in t_low for w in ["golpe", "golpista", "bloquear", "imei", "sigiloso"]):
            tags.append("#fraude_golpe")
        if any(w in t_low for w in ["iphone", "gold", "peca", "aparelho", "atacado"]):
            tags.append("#receptacao_celulares")
        if any(w in t_low for w in ["laranja", "conta do joao", "passa a conta"]):
            tags.append("#laranja_potencial")

        eventos = []
        for e in raw_entities:
            if e["categoria"] == "VALOR_MONETARIO":
                raw_val = e.get("metadata", {}).get("valor_bruto", e["text"])
                clean_num = re.sub(r'[^\d,.]', '', raw_val).replace(",", ".")
                try:
                    eventos.append({
                        "tipo": "TRANSACAO_FINANCEIRA",
                        "valor": float(clean_num),
                        "tempo_ancorado": chunk["start_time"],
                        "descricao": f"Transação financeira identificada: {e['text']}"
                    })
                except Exception:
                    pass

        return {
            "entidades": entidades,
            "eventos_transacoes": eventos,
            "tags_criminais": tags or ["#comunicacao_operacional"],
            "engine": "Taxonomia Estrita 17 Categorias (Deterministic High-Precision)"
        }

    def process_session_chunk(self, chunk: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
        # Extração de base determinística de 17 categorias
        base_result = self.rule_based_fallback(chunk)
        
        if use_llm:
            prompt = (
                f"Analise a sessão conversacional a seguir (Início: {chunk['start_time']} | Chat: '{chunk['chat_title']}'):\n\n"
                f"{chunk['full_text']}\n\n"
                f"Extraia estritamente conforme as 17 categorias de entidades forenses."
            )
            llm_res = self.query_llm(prompt)
            if llm_res and "entidades" in llm_res:
                # Enriquece entidades da base com as descobertas do modelo
                seen = set(e["nome_ou_termo"].lower() for e in base_result["entidades"])
                for le in llm_res.get("entidades", []):
                    term = le.get("nome_ou_termo") or le.get("nome") or le.get("entidade")
                    cat = le.get("categoria", "OUTRO")
                    if term and term.lower() not in seen:
                        seen.add(term.lower())
                        base_result["entidades"].append({
                            "nome_ou_termo": term,
                            "categoria": cat,
                            "papel_contexto": le.get("papel_contexto") or le.get("papel", cat),
                            "confidence": 0.92
                        })
                base_result["engine"] = f"Híbrido (Taxonomia 17 + {self.model})"

        base_result["chunk_id"] = chunk["chunk_id"]
        base_result["chat_id"] = chunk["chat_id"]
        base_result["start_time"] = chunk["start_time"]
        return base_result
