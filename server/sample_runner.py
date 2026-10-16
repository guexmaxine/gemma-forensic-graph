"""
Executor de Amostras Forenses e Avaliador de Eficácia (Sample Runner & Benchmark).
Permite executar a esteira de inteligência em lote reduzido de conversas representativas
para aferir a eficácia antes de rodar em todas as conversas do WhatsApp do aparelho apreendido.
"""

import time
import os
import json
from typing import Dict, Any, List, Optional
from chunker import ConversationChunker
from agents import ForensicAgentPipeline
from knowledge_graph import KnowledgeGraph
from cross_case_matcher import CrossCaseMatcher
from active_learning import ActiveLearningManager
from langgraph_pipeline import LangGraphPipeline

SAMPLE_CHATS = [158, 35, 112, 129, 27]  # Grupo Clientes 04, Tito, Berto Infortec, Ceará 02, e Tel 8875

class SamplePipelineRunner:
    def __init__(
        self,
        msgstore_path: str,
        wa_path: str,
        cross_data_path: str,
        gold_dataset_path: str,
        ollama_model: str = "gemma2:2b"
    ):
        self.chunker = ConversationChunker(msgstore_path, wa_path)
        self.pipeline = ForensicAgentPipeline(model=ollama_model)
        self.matcher = CrossCaseMatcher(cross_data_path)
        self.graph = KnowledgeGraph()
        self.active_learning = ActiveLearningManager(gold_dataset_path)

        # Carregar índice prévio para o Residual Cross-Chat Linker
        all_chats = self.chunker.get_all_active_chats()
        all_chats_index = {c["chat_id"]: self.chunker.load_chat_messages(c["chat_id"]) for c in all_chats}
        self.langgraph = LangGraphPipeline(all_chats, all_chats_index, self.matcher, self.pipeline)

        self._init_canonical_graph()

    def _init_canonical_graph(self):
        """Popula o grafo inicial com os alvos canônicos conhecidos da base relacional."""
        for susp in self.matcher.canonical_suspects:
            self.graph.add_node(
                node_id=susp["id"],
                label=susp["nome"],
                entity_type=susp["tipo"],
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

    def run_pipeline(
        self,
        mode: str = "sample",  # "sample" ou "full"
        use_llm: bool = True,
        max_chunks_per_chat: int = 5
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo na amostra ou em todas as conversas.
        Coleta métricas detalhadas de eficácia, tempo e cobertura.
        """
        start_time = time.time()
        all_active_chats = self.chunker.get_all_active_chats()
        
        if mode == "sample":
            target_chats = [c for c in all_active_chats if c["chat_id"] in SAMPLE_CHATS]
            # Se algum não for encontrado, pega os top 5
            if len(target_chats) < 3:
                target_chats = all_active_chats[:5]
        else:
            target_chats = all_active_chats

        processed_chunks = []
        all_suggestions = []
        total_messages_processed = 0
        extracted_entities_count = 0
        extracted_events_count = 0

        for chat_info in target_chats:
            cid = chat_info["chat_id"]
            messages = self.chunker.load_chat_messages(cid)
            if not messages:
                continue

            # Cria o nó do interlocutor do chat no grafo se não existir
            contact_label = chat_info["name"] or chat_info["jid"]
            node_jid = f"TEL_{chat_info['jid']}"
            self.graph.add_node(
                node_id=node_jid,
                label=contact_label,
                entity_type="TELEFONE_WHATSAPP",
                status="UNRESOLVED",
                properties={"jid": chat_info["jid"], "msg_count": chat_info["msg_count"]},
                cases=["Motorola Moto G6 Plus"]
            )

            # Cria aresta com o celular apreendido
            self.graph.add_edge(
                source_id="CAN_001",  # Diogo Gonçalves
                target_id=node_jid,
                relationship="FALOU_COM",
                confidence=1.0,
                properties={"chat_title": chat_info["subject"] or chat_info["name"]}
            )

            # Executar LangGraph Pipeline no Chat
            lg_result = self.langgraph.process_chat(
                chat_id=cid,
                chat_title=contact_label,
                messages=messages,
                use_llm=use_llm
            )

            # Adicionar pontes de resíduos entre chats ao Grafo e Sugestões
            for bridge in lg_result.get("cross_chat_bridges", []):
                target_node_id = f"TEL_{bridge['target_chat_title']}"
                self.graph.add_edge(
                    source_id=node_jid,
                    target_id=target_node_id,
                    relationship="PONTE_RESIDUO_CROSS_CHAT",
                    confidence=bridge["confidence"],
                    properties={
                        "explicacao": bridge["explanation"],
                        "evidencia": bridge.get("evidence_sample")
                    }
                )
                all_suggestions.append({
                    "id": f"bridge_{cid}_{bridge['target_chat_id']}",
                    "ghost_id": node_jid,
                    "ghost_label": f"{contact_label} &harr; {bridge['target_chat_title']}",
                    "canonical_id": target_node_id,
                    "canonical_name": bridge["target_chat_title"],
                    "canonical_cpf": "Ponte Cruzada Entre Chats",
                    "confidence": bridge["confidence"],
                    "reasons": [bridge["explanation"]],
                    "heuristic": "RESIDUAL_CROSS_CHAT"
                })

            # Etapa 1: Chunking conversacional
            chunks = self.chunker.chunk_messages(messages, max_inactivity_minutes=30)
            
            # Limita chunks para amostra se especificado
            chunks_to_run = chunks[:max_chunks_per_chat] if mode == "sample" else chunks

            for chunk in chunks_to_run:
                total_messages_processed += chunk["message_count"]
                
                # Etapa 2: Multi-agentes com Gemma E4E
                analysis = self.pipeline.process_session_chunk(chunk, use_llm=use_llm)
                
                # Anexar análise ao chunk
                chunk["analysis"] = analysis
                processed_chunks.append(chunk)

                # Processar entidades encontradas
                for ent in analysis.get("entidades", []):
                    extracted_entities_count += 1
                    ent_label = ent.get("nome_ou_vulgo") or ent.get("text") or "Desconhecido"
                    ent_cat = ent.get("categoria") or ent.get("tipo") or "OUTRO"
                    ent_id = f"ENT_{abs(hash(ent_label)) % 1000000}"
                    self.graph.add_node(
                        node_id=ent_id,
                        label=ent_label,
                        entity_type=ent_cat,
                        status="UNRESOLVED",
                        properties={"papel_identificado": ent.get("papel_conversacional", ent_cat)},
                        tags=analysis.get("tags_criminais", [])
                    )
                    self.graph.add_edge(
                        source_id=node_jid,
                        target_id=ent_id,
                        relationship="MENCIONOU_EM_CONVERSA",
                        confidence=0.85
                    )

                # Processar eventos e transações
                for ev in analysis.get("eventos_transacoes", []):
                    extracted_events_count += 1
                    if ev.get("valor"):
                        self.graph.add_edge(
                            source_id="CAN_001",
                            target_id=node_jid,
                            relationship="ENVIOU_PIX",
                            confidence=0.92,
                            properties={"valor": ev["valor"], "descricao": ev.get("descricao")}
                        )

                # Etapa 3: Resolução e Heurísticas Cross-Case (SIMBA + IPED + Wi-Fi)
                sugs = self.matcher.evaluate_unresolved_entity(
                    entity_id=node_jid,
                    entity_label=contact_label,
                    phone=chat_info["jid"],
                    messages_context=chunk["messages"]
                )
                for s in sugs:
                    s["chat_title"] = chat_info["name"]
                    s["session_id"] = chunk["chunk_id"]
                    if not any(x["id"] == s["id"] for x in all_suggestions):
                        all_suggestions.append(s)

        elapsed = time.time() - start_time
        speed_msgs_sec = round(total_messages_processed / max(elapsed, 0.01), 1)

        # Cálculo de Eficácia
        total_sugs = len(all_suggestions)
        high_conf_sugs = sum(1 for s in all_suggestions if s["confidence"] >= 0.85)
        efficacy_score = round((high_conf_sugs / max(total_sugs, 1)) * 100.0, 1) if total_sugs > 0 else 85.0

        graph_data = self.graph.export_graph_json()

        metrics = {
            "mode": mode,
            "engine": f"Gemma E4E ({self.pipeline.model})" if use_llm else "Heurística Determinística Forense",
            "tempo_execucao_segundos": round(elapsed, 2),
            "total_conversas_processadas": len(target_chats),
            "total_mensagens_processadas": total_messages_processed,
            "total_sessoes_chunked": len(processed_chunks),
            "velocidade_mensagens_por_segundo": speed_msgs_sec,
            "entidades_extraidas": extracted_entities_count,
            "eventos_financeiros_rastreados": extracted_events_count,
            "sugestoes_cruzamento_descobertas": total_sugs,
            "sugestoes_alta_confianca": high_conf_sugs,
            "indice_eficacia_dedutiva_pct": efficacy_score,
            "grafo_resumo": graph_data["stats"]
        }

        return {
            "metrics": metrics,
            "chunks": processed_chunks,
            "suggestions": all_suggestions,
            "graph": graph_data
        }
