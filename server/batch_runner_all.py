"""
Processador em Lote Completo de Todo o Caso Motorola ("Rode em Tudo").
Varre todos os chats e mensagens reais da extração, extrai as 17 categorias de entidades,
identifica pontes residuais cross-chat, calcula cruzamentos SIMBA/IPED e gera o grafo global.
"""

import os
import sys
import json
import time
import sqlite3
import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MSGSTORE_FULL = os.path.join(DATA_DIR, "msgstore_full.db")
WA_FULL = os.path.join(DATA_DIR, "wa_full.db")
MSGSTORE_PATH = MSGSTORE_FULL if os.path.exists(MSGSTORE_FULL) else os.path.join(DATA_DIR, "msgstore.db")
WA_PATH = WA_FULL if os.path.exists(WA_FULL) else os.path.join(DATA_DIR, "wa.db")
CROSS_DATA_PATH = os.path.join(DATA_DIR, "cross_case_data.json")
OUTPUT_CACHE_PATH = os.path.join(DATA_DIR, "full_processed_dataset.json")

from chunker import ConversationChunker
from agents import ForensicAgentPipeline
from cross_case_matcher import CrossCaseMatcher
from knowledge_graph import KnowledgeGraph
from langgraph_pipeline import LangGraphPipeline
from entity_extractor_v2 import ForensicEntityTaxonomyExtractor

def run_all_chats_pipeline():
    start_time = time.time()
    print("==================================================================")
    print("  INICIANDO PROCESSAMENTO COMPLETO DE TODOS OS CHATS MOTOROLA")
    print("  Taxonomia Estrita: 17 Categorias de Entidades Forenses")
    print("==================================================================")

    chunker = ConversationChunker(MSGSTORE_PATH, WA_PATH)
    matcher = CrossCaseMatcher(CROSS_DATA_PATH)
    pipeline_agent = ForensicAgentPipeline()
    graph = KnowledgeGraph()

    # 1. Carregar nós canônicos
    for susp in matcher.canonical_suspects:
        graph.add_node(
            node_id=susp["id"],
            label=susp["nome"],
            entity_type="PESSOA" if susp["tipo"] == "PESSOA_FISICA" else "ORGANIZACAO",
            status="VERIFIED",
            canonical_id=susp["id"],
            properties={
                "cpf_cnpj": susp.get("cpf") or susp.get("cnpj"),
                "vulgo": susp.get("vulgo"),
                "papel": susp.get("papel_presumido"),
                "telefones": susp.get("telefones", [])
            },
            tags=["#alvo_identificado", "#base_relacional"],
            cases=susp.get("casos", ["Base SIMBA/RIF"])
        )

    # 2. Carregar todas as conversas e mensagens
    all_chats = chunker.get_all_active_chats()
    print(f"Total de conversas ativas no aparelho: {len(all_chats)}")

    all_chats_index = {}
    for c in all_chats:
        msgs = chunker.load_chat_messages(c["chat_id"])
        if msgs:
            all_chats_index[c["chat_id"]] = msgs

    langgraph = LangGraphPipeline(all_chats, all_chats_index, matcher, pipeline_agent)

    processed_chats_data = {}
    all_suggestions = []
    total_msgs = 0
    total_entities_extracted = 0
    category_counts = {}

    for idx, c in enumerate(all_chats):
        cid = c["chat_id"]
        msgs = all_chats_index.get(cid, [])
        if not msgs:
            continue

        total_msgs += len(msgs)
        contact_label = c["name"] or c["jid"]
        node_jid = f"TEL_{c['jid']}"

        # Adiciona nó do interlocutor
        graph.add_node(
            node_id=node_jid,
            label=contact_label,
            entity_type="TELEFONE",
            status="UNRESOLVED",
            properties={"jid": c["jid"], "msg_count": len(msgs)},
            cases=["Motorola Moto G6 Plus"]
        )

        graph.add_edge(
            source_id="CAN_001",
            target_id=node_jid,
            relationship="FALOU_COM",
            confidence=1.0,
            properties={"chat_title": contact_label, "total_mensagens": len(msgs)}
        )

        # Executa LangGraph
        lg_res = langgraph.process_chat(cid, contact_label, msgs, use_llm=False)
        
        # Chunks de sessão
        chunks = chunker.chunk_messages(msgs, max_inactivity_minutes=30)
        enriched_chunks = []

        for ch in chunks:
            analysis = pipeline_agent.rule_based_fallback(ch)
            ch["analysis"] = analysis

            # Anota entidades em cada mensagem
            for m in ch["messages"]:
                m_ents = ForensicEntityTaxonomyExtractor.extract_entities_from_text(m["text"], anchor_time=m["timestamp"])
                m["entities_17"] = m_ents
                for me in m_ents:
                    cat = me["categoria"]
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    total_entities_extracted += 1

                    # Nós no grafo para entidades de alto valor
                    if cat in ["PESSOA", "ORGANIZACAO", "LOCAL", "VEICULO", "DOCUMENTO"]:
                        ent_node_id = f"{cat}_{abs(hash(me['text'])) % 1000000}"
                        graph.add_node(
                            node_id=ent_node_id,
                            label=me["text"],
                            entity_type=cat,
                            status="UNRESOLVED",
                            tags=[f"#{cat.lower()}"]
                        )
                        graph.add_edge(
                            source_id=node_jid,
                            target_id=ent_node_id,
                            relationship=f"MENCIONOU_{cat}",
                            confidence=0.88
                        )

            enriched_chunks.append(ch)

        # Resíduos e pontes cross-chat
        for bridge in lg_res.get("cross_chat_bridges", []):
            target_node = f"TEL_{bridge['target_chat_title']}"
            graph.add_edge(
                source_id=node_jid,
                target_id=target_node,
                relationship="PONTE_RESIDUO_CROSS_CHAT",
                confidence=bridge["confidence"],
                properties={"explicacao": bridge["explanation"]}
            )
            all_suggestions.append({
                "id": f"bridge_{cid}_{bridge['target_chat_id']}",
                "ghost_id": node_jid,
                "ghost_label": f"{contact_label} &harr; {bridge['target_chat_title']}",
                "canonical_id": target_node,
                "canonical_name": bridge["target_chat_title"],
                "canonical_cpf": "Ponte Cruzada Entre Chats",
                "confidence": bridge["confidence"],
                "reasons": [bridge["explanation"]],
                "heuristic": "RESIDUAL_CROSS_CHAT"
            })

        # Cruzamentos SIMBA / Outros IPEDs
        for s in lg_res.get("cross_case_links", []):
            s["chat_title"] = contact_label
            if not any(x["id"] == s["id"] for x in all_suggestions):
                all_suggestions.append(s)

        processed_chats_data[cid] = {
            "chat_info": c,
            "messages": msgs,
            "chunks": enriched_chunks,
            "analysis": lg_res
        }

        if (idx + 1) % 15 == 0 or (idx + 1) == len(all_chats):
            print(f"  Progresso: {idx + 1}/{len(all_chats)} chats processados ({total_msgs} mensagens)...")

    elapsed = time.time() - start_time

    # Salvar cache enriquecido
    final_output = {
        "metadata": {
            "device": "Motorola Moto G6 Plus (XT1926-9)",
            "alvo": "Diogo Gonçalves Costa",
            "total_conversas": len(processed_chats_data),
            "total_mensagens": total_msgs,
            "total_entidades_17": total_entities_extracted,
            "tempo_processamento_s": round(elapsed, 2),
            "taxonomia_contadores": category_counts,
            "gerado_em": datetime.datetime.now().isoformat()
        },
        "chats": processed_chats_data,
        "suggestions": all_suggestions,
        "graph": graph.export_graph_json()
    }

    with open(OUTPUT_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\n==================================================================")
    print(f"  PROCESSAMENTO CONCLUÍDO COM SUCESSO EM {elapsed:.2f} SEGUNDOS!")
    print(f"  Total Mensagens: {total_msgs} em {len(processed_chats_data)} conversas")
    print(f"  Total Entidades Extraídas (17 categorias): {total_entities_extracted}")
    print(f"  Sugestões e Pontes Descobertas: {len(all_suggestions)}")
    print(f"  Cache salvo em: {OUTPUT_CACHE_PATH}")
    print("==================================================================")

if __name__ == "__main__":
    run_all_chats_pipeline()
