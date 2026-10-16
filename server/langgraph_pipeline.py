"""
Pipeline Multi-Agentes com Arquitetura de Grafo de Estados (LangGraph / StateGraph Pattern).
Coordena 5 agentes especializados para o caso Motorola:
1. SessionChunkerAgent (Chunking de sessão de 30 min + triggers)
2. ContextualNERAgent (Extração de entidades e papéis na conversa)
3. TemporalTransactionAgent (Extração de transações e âncoras temporais)
4. CrossCaseLinkerAgent (Cruzamento com SIMBA, Wi-Fi e outros casos IPED)
5. ResidualCrossChatLinkerAgent (Mapeador de resíduos: conecta citações órfãs entre os 117 chats)
"""

import json
import re
import datetime
from typing import Dict, List, Any, Optional

class LangGraphForensicState:
    """Estado tipado da investigação compartilhado entre os agentes da esteira."""
    def __init__(self, chat_id: int, chat_title: str, messages: List[Dict[str, Any]]):
        self.chat_id = chat_id
        self.chat_title = chat_title
        self.messages = messages
        self.sessions: List[Dict[str, Any]] = []
        self.extracted_entities: List[Dict[str, Any]] = []
        self.extracted_transactions: List[Dict[str, Any]] = []
        self.cross_case_links: List[Dict[str, Any]] = []
        self.residual_mentions: List[Dict[str, Any]] = []
        self.cross_chat_bridges: List[Dict[str, Any]] = []
        self.tags: List[str] = []
        self.step_history: List[str] = []

class ResidualCrossChatLinkerAgent:
    """
    Agente Especializado em Resíduos Conversacionais (Cross-Chat Linker).
    Identifica menções órfãs ou citações de terceiros em um chat e busca sistematicamente
    suas correspondências nos outros chats do caso, construindo as pontes operacionais.
    """
    def __init__(self, all_chats_metadata: List[Dict[str, Any]], all_chats_index: Dict[int, List[Dict[str, Any]]]):
        self.all_chats_metadata = all_chats_metadata
        self.all_chats_index = all_chats_index

    def execute(self, state: LangGraphForensicState) -> LangGraphForensicState:
        state.step_history.append("ResidualCrossChatLinkerAgent")
        
        # 1. Identificar entidades candidatas a resíduo
        potential_residuals = []
        for ent in state.extracted_entities:
            name = ent.get("nome") or ent.get("nome_ou_vulgo")
            if not name:
                continue
            if any(ign in name.lower() for ign in ["você", "diogo", "cliente", "vendedor", "apple", "eu"]):
                continue
            potential_residuals.append(name.strip())

        # 2. Varrer os outros chats da investigação
        for res_name in set(potential_residuals):
            norm_res = res_name.lower()
            
            for chat_meta in self.all_chats_metadata:
                other_cid = chat_meta["chat_id"]
                if str(other_cid) == str(state.chat_id):
                    continue

                other_title = (chat_meta.get("name") or chat_meta.get("subject") or "").strip()
                other_jid = (chat_meta.get("jid") or "").strip()
                
                # Caso A: A entidade é o próprio interlocutor do outro chat
                is_interlocutor = (
                    norm_res == other_title.lower() or 
                    norm_res in other_title.lower() or 
                    (other_jid and norm_res in other_jid.lower())
                )
                
                # Caso B: Menções no corpo das mensagens do outro chat
                other_msgs = self.all_chats_index.get(other_cid, [])
                matching_msgs = [om for om in other_msgs if norm_res in om.get("text", "").lower()]

                if is_interlocutor or matching_msgs:
                    bridge = {
                        "residual_entity": res_name,
                        "source_chat_id": state.chat_id,
                        "source_chat_title": state.chat_title,
                        "target_chat_id": other_cid,
                        "target_chat_title": other_title or other_jid,
                        "is_interlocutor_bridge": is_interlocutor,
                        "matches_count": len(matching_msgs) + (1 if is_interlocutor else 0),
                        "evidence_sample": matching_msgs[0].get("text", "")[:120] if matching_msgs else f"Interlocutor direto: {other_title}",
                        "confidence": 0.96 if is_interlocutor else (0.85 if len(matching_msgs) > 1 else 0.75),
                        "relationship_type": "CANAL_PRIVADO_COM_MEMBRO_DE_GRUPO" if is_interlocutor else "RESIDUO_MUTUO_EM_CONVERSAS",
                        "explanation": (
                            f"Ponte de Resíduo: O interlocutor '{res_name}' ativo no grupo '{state.chat_title}' "
                            + (f"possui chat privado 1-a-1 com o alvo no Chat {other_cid} ('{other_title}')." if is_interlocutor else 
                               f"reaparece em {len(matching_msgs)} mensagens no Chat '{other_title}'.")
                        )
                    }
                    state.cross_chat_bridges.append(bridge)

        return state

class LangGraphPipeline:
    """Grafo de Execução Multi-Agentes Orquestrado."""
    def __init__(self, all_chats_metadata: List[Dict[str, Any]], all_chats_index: Dict[int, List[Dict[str, Any]]], matcher=None, pipeline_agent=None):
        self.all_chats_metadata = all_chats_metadata
        self.all_chats_index = all_chats_index
        self.matcher = matcher
        self.pipeline_agent = pipeline_agent
        self.residual_agent = ResidualCrossChatLinkerAgent(all_chats_metadata, all_chats_index)

    def process_chat(self, chat_id: int, chat_title: str, messages: List[Dict[str, Any]], use_llm: bool = True) -> Dict[str, Any]:
        state = LangGraphForensicState(chat_id, chat_title, messages)
        
        # Nó 1: Chunking de Sessão
        from chunker import ConversationChunker
        # Agrupa em sessões de 30 min
        sessions = []
        if messages:
            curr_sess = [messages[0]]
            for i in range(1, len(messages)):
                prev = messages[i-1]
                curr = messages[i]
                diff_min = (curr["raw_timestamp"] - prev["raw_timestamp"]) / 60000.0
                if diff_min > 30:
                    sessions.append(curr_sess)
                    curr_sess = [curr]
                else:
                    curr_sess.append(curr)
            if curr_sess:
                sessions.append(curr_sess)
        state.sessions = sessions
        state.step_history.append("SessionChunkerAgent")

        # Nó 2 & 3: Contextual NER & Temporal Transactions
        for s_idx, sess in enumerate(sessions):
            full_text = "\n".join([f"[{m['timestamp']}] {m['sender_name']}: {m['text']}" for m in sess])
            chunk_dict = {
                "chunk_id": f"chunk_{chat_id}_{s_idx+1:02d}",
                "chat_id": chat_id,
                "chat_title": chat_title,
                "start_time": sess[0]["timestamp"],
                "end_time": sess[-1]["timestamp"],
                "message_count": len(sess),
                "messages": sess,
                "full_text": full_text
            }
            
            # Executa extração (com LLM ou Heurística Inteligente)
            if self.pipeline_agent:
                analysis = self.pipeline_agent.process_session_chunk(chunk_dict, use_llm=use_llm)
                state.extracted_entities.extend(analysis.get("entidades", []))
                state.extracted_transactions.extend(analysis.get("eventos_transacoes", []))
                state.tags.extend(analysis.get("tags_criminais", []))

        state.step_history.append("ContextualNERAgent")
        state.step_history.append("TemporalTransactionAgent")

        # Nó 4: Cross-Case Linker (SIMBA Pix, Contatos de outros IPEDs, Wi-Fi)
        if self.matcher:
            first_phone = messages[0]["sender_jid"] if messages else ""
            sugs = self.matcher.evaluate_unresolved_entity(
                entity_id=f"TEL_{first_phone}",
                entity_label=chat_title,
                phone=first_phone,
                messages_context=messages
            )
            state.cross_case_links = sugs
            state.step_history.append("CrossCaseLinkerAgent")

        # Nó 5: Residual Cross-Chat Linker (O pulo do gato para o resíduo dos outros chats)
        state = self.residual_agent.execute(state)

        return {
            "chat_id": state.chat_id,
            "chat_title": state.chat_title,
            "sessions_count": len(state.sessions),
            "entities": state.extracted_entities,
            "transactions": state.extracted_transactions,
            "cross_case_links": state.cross_case_links,
            "cross_chat_bridges": state.cross_chat_bridges,
            "tags": list(set(state.tags)),
            "graph_steps": state.step_history
        }
