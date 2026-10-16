"""
Módulo de Chunking e Pré-Processamento Conversacional (Etapa 1 do Pipeline).
Agrupa o fluxo bruto do WhatsApp em Janelas de Sessão Conversacional baseadas em tempo
(inatividade > 30 min ou janela deslizante de 40 msgs com overlap de 5).
"""

import sqlite3
import datetime
import re
from typing import List, Dict, Any, Optional

TRIGGER_REGEX = re.compile(
    r'(r\$|\bpix\b|\btransfer\b|\bcomprovante\b|\bdeposito\b|\bimei\b|\biphone\b|\bnota\b|\bcarga\b|\bpeça\b|\bposto\b|\bencontro\b|\blocal\b|\bpatr[aã]o\b|\bchefe\b|\bgordo\b|\bbeto\b|\balem[aã]o\b|\bmenino\b|\bconta\b|\bbanco\b|\bcaixa\b|\bagencia\b|\bcpf\b|\bcnpj\b|\bchave\b|\bgolpe\b|\bbloque\b|\bpolicia\b|\badv\b|\boficio\b)',
    re.IGNORECASE
)

class ConversationChunker:
    def __init__(self, msgstore_path: str, wa_path: str):
        self.msgstore_path = msgstore_path
        self.wa_path = wa_path

    def load_chat_messages(self, chat_id: int) -> List[Dict[str, Any]]:
        """Carrega e ordena as mensagens de um chat específico com os contatos vinculados."""
        conn = sqlite3.connect(self.msgstore_path)
        conn.execute("ATTACH DATABASE ? AS wa", (self.wa_path,))
        cur = conn.cursor()

        cur.execute("""
            SELECT 
                m._id,
                m.chat_row_id,
                m.from_me,
                m.timestamp,
                datetime(m.timestamp/1000, 'unixepoch') as iso_time,
                m.text_data,
                j.raw_string as sender_jid,
                coalesce(w.display_name, w.wa_name, j.raw_string) as sender_name,
                c.subject as chat_title
            FROM message m
            JOIN chat c ON m.chat_row_id = c._id
            LEFT JOIN jid j ON (
                CASE 
                    WHEN m.sender_jid_row_id > 0 THEN m.sender_jid_row_id 
                    ELSE c.jid_row_id 
                END
            ) = j._id
            LEFT JOIN wa.wa_contacts w ON w.jid = j.raw_string
            WHERE m.chat_row_id = ? AND m.text_data IS NOT NULL AND length(trim(m.text_data)) > 0
            ORDER BY m.timestamp ASC
        """, (chat_id,))

        messages = []
        for r in cur.fetchall():
            messages.append({
                "id": str(r[0]),
                "chat_id": r[1],
                "from_me": bool(r[2]),
                "raw_timestamp": r[3],
                "timestamp": r[4],
                "text": r[5].strip(),
                "sender_jid": r[6],
                "sender_name": "Você (Diogo)" if r[2] else (r[7] or "Interlocutor"),
                "chat_title": r[8] or r[6]
            })
        conn.close()
        return messages

    def get_all_active_chats(self) -> List[Dict[str, Any]]:
        """Lista todos os chats que possuem mensagens de texto."""
        conn = sqlite3.connect(self.msgstore_path)
        conn.execute("ATTACH DATABASE ? AS wa", (self.wa_path,))
        cur = conn.cursor()

        cur.execute("""
            SELECT 
                c._id,
                j.raw_string,
                c.subject,
                coalesce(w.display_name, w.wa_name, c.subject, j.raw_string) as contact_name,
                count(m._id) as msg_count,
                min(datetime(m.timestamp/1000, 'unixepoch')) as first_msg,
                max(datetime(m.timestamp/1000, 'unixepoch')) as last_msg
            FROM chat c
            JOIN jid j ON c.jid_row_id = j._id
            LEFT JOIN wa.wa_contacts w ON w.jid = j.raw_string
            LEFT JOIN message m ON m.chat_row_id = c._id AND m.text_data IS NOT NULL
            GROUP BY c._id
            HAVING msg_count > 0
            ORDER BY msg_count DESC
        """)

        chats = []
        for r in cur.fetchall():
            chats.append({
                "chat_id": r[0],
                "jid": r[1],
                "subject": r[2],
                "name": r[3],
                "msg_count": r[4],
                "first_date": r[5],
                "last_date": r[6]
            })
        conn.close()
        return chats

    def chunk_messages(
        self,
        messages: List[Dict[str, Any]],
        max_inactivity_minutes: int = 30,
        max_chunk_size: int = 40,
        overlap: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Agrupa mensagens em sessões conversacionais:
        1. Se intervalo entre msg N e N+1 > max_inactivity_minutes -> quebra de sessão.
        2. Se sessão > max_chunk_size -> divide em blocos deslizantes com overlap.
        """
        if not messages:
            return []

        sessions: List[List[Dict[str, Any]]] = []
        current_session: List[Dict[str, Any]] = [messages[0]]

        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]

            prev_ts = prev_msg["raw_timestamp"] / 1000.0
            curr_ts = curr_msg["raw_timestamp"] / 1000.0

            time_diff_min = (curr_ts - prev_ts) / 60.0

            if time_diff_min > max_inactivity_minutes:
                sessions.append(current_session)
                current_session = [curr_msg]
            else:
                current_session.append(curr_msg)

        if current_session:
            sessions.append(current_session)

        # Aplicar janela deslizante se a sessão exceder max_chunk_size
        chunks: List[Dict[str, Any]] = []
        chunk_idx = 1

        for sess in sessions:
            if len(sess) <= max_chunk_size:
                chunks.append(self._build_chunk_dict(sess, chunk_idx))
                chunk_idx += 1
            else:
                # Divide com overlap
                start = 0
                while start < len(sess):
                    sub = sess[start : start + max_chunk_size]
                    chunks.append(self._build_chunk_dict(sub, chunk_idx))
                    chunk_idx += 1
                    if start + max_chunk_size >= len(sess):
                        break
                    start += (max_chunk_size - overlap)

        return chunks

    def _build_chunk_dict(self, msg_list: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
        """Constrói o objeto de sessão com metadados e tags de gatilho."""
        start_time = msg_list[0]["timestamp"]
        end_time = msg_list[-1]["timestamp"]
        full_text = "\n".join([f"[{m['timestamp']}] {m['sender_name']}: {m['text']}" for m in msg_list])
        
        # Filtro de relevância forense
        triggers = TRIGGER_REGEX.findall(full_text)
        relevance_score = min(1.0, len(triggers) * 0.2 + (0.3 if len(msg_list) > 3 else 0.1))

        return {
            "chunk_id": f"chunk_{index:03d}",
            "chat_id": msg_list[0]["chat_id"],
            "chat_title": msg_list[0]["chat_title"],
            "start_time": start_time,
            "end_time": end_time,
            "message_count": len(msg_list),
            "messages": msg_list,
            "full_text": full_text,
            "has_triggers": len(triggers) > 0,
            "trigger_keywords": list(set([t.lower() for t in triggers])),
            "relevance_score": round(relevance_score, 2)
        }
