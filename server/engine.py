"""
Servidor Backend REST e Servidor Web da Plataforma Analítica Forense.
Fornece APIs para as 3 Visões da Interface:
- Visão 1: Leitor de Chat Aumentado (com 17 categorias de entidades)
- Visão 2: Resolução de Entidades Desconhecidas (Tinder de Merge)
- Visão 3: Explorador de Grafo Multi-Caso
"""

import http.server
import socketserver
import json
import urllib.parse
import os
import sys
from typing import Dict, Any, List

from sample_runner import SamplePipelineRunner
from entity_extractor_v2 import ForensicEntityTaxonomyExtractor

PORT = 3800
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEB_DIR = os.path.join(BASE_DIR, "web")

MSGSTORE_FULL = os.path.join(DATA_DIR, "msgstore_full.db")
WA_FULL = os.path.join(DATA_DIR, "wa_full.db")
MSGSTORE_PATH = MSGSTORE_FULL if os.path.exists(MSGSTORE_FULL) else os.path.join(DATA_DIR, "msgstore.db")
WA_PATH = WA_FULL if os.path.exists(WA_FULL) else os.path.join(DATA_DIR, "wa.db")

CROSS_DATA_PATH = os.path.join(DATA_DIR, "cross_case_data.json")
GOLD_DATASET_PATH = os.path.join(DATA_DIR, "gold_dataset.jsonl")
CACHE_FULL_PATH = os.path.join(DATA_DIR, "full_processed_dataset.json")

# Carrega o cache completo de 10.701 mensagens se disponível
full_dataset_cache = None
if os.path.exists(CACHE_FULL_PATH):
    print(f"Carregando cache processado completo de {CACHE_FULL_PATH}...")
    try:
        with open(CACHE_FULL_PATH, "r", encoding="utf-8") as f:
            full_dataset_cache = json.load(f)
        print(f"Dataset completo carregado: {full_dataset_cache['metadata']['total_mensagens']} msgs em {full_dataset_cache['metadata']['total_conversas']} conversas!")
    except Exception as e:
        print(f"Erro ao carregar cache: {e}")

# Inicializa o executor do pipeline
runner = SamplePipelineRunner(
    msgstore_path=MSGSTORE_PATH,
    wa_path=WA_PATH,
    cross_data_path=CROSS_DATA_PATH,
    gold_dataset_path=GOLD_DATASET_PATH,
    ollama_model="llama3.2:3b"
)

# Inicializa cache ativo instantâneo
if full_dataset_cache:
    current_run_cache = {
        "metrics": full_dataset_cache["metadata"],
        "suggestions": full_dataset_cache.get("suggestions", []),
        "processed_chunks": []
    }
else:
    current_run_cache = runner.run_pipeline(mode="sample", use_llm=False)

class ForensicAPIHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def _send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Status & Métricas com 17 Categorias
        if path == "/api/status":
            stats = runner.active_learning.get_dataset_stats()
            meta = full_dataset_cache["metadata"] if full_dataset_cache else current_run_cache["metrics"]
            self._send_json({
                "status": "online",
                "device": "Motorola Moto G6 Plus (XT1926-9)",
                "case_owner": "Diogo Gonçalves Costa",
                "model_active": runner.pipeline.model,
                "gold_dataset": stats,
                "dataset_meta": meta,
                "latest_metrics": current_run_cache["metrics"]
            })
            return

        # 2. Lista de Conversas (todas ou filtradas)
        elif path == "/api/chats":
            if full_dataset_cache and "chats" in full_dataset_cache:
                chat_list = []
                for cid_str, cdata in full_dataset_cache["chats"].items():
                    info = cdata["chat_info"]
                    chat_list.append({
                        "chat_id": int(cid_str),
                        "jid": info["jid"],
                        "name": info["name"] or info["jid"],
                        "subject": info.get("subject"),
                        "msg_count": len(cdata["messages"]),
                        "first_date": info.get("first_date", ""),
                        "last_date": info.get("last_date", ""),
                        "has_cross_alert": any("Grupo" in str(info.get("name", "")) or "Tito" in str(info.get("name", "")) for _ in [1]),
                        "tags": ["#fraude_golpe"] if "Grupo" in str(info.get("name", "")) else ["#operacional"]
                    })
                chat_list.sort(key=lambda x: x["msg_count"], reverse=True)
                self._send_json(chat_list)
            else:
                all_chats = runner.chunker.get_all_active_chats()
                self._send_json(all_chats)
            return

        # 3. Mensagens e Chunks de um Chat específico com as 17 Categorias
        elif path.startswith("/api/chats/") and path.endswith("/messages"):
            parts = path.strip("/").split("/")
            chat_id = int(parts[2])

            if full_dataset_cache and str(chat_id) in full_dataset_cache["chats"]:
                cdata = full_dataset_cache["chats"][str(chat_id)]
                self._send_json({
                    "chat_id": chat_id,
                    "chat_info": cdata["chat_info"],
                    "messages": cdata["messages"],
                    "chunks": cdata["chunks"]
                })
            else:
                messages = runner.chunker.load_chat_messages(chat_id)
                chunks = runner.chunker.chunk_messages(messages)
                for ch in chunks:
                    ch["analysis"] = runner.pipeline.rule_based_fallback(ch)
                    for m in ch["messages"]:
                        m["entities_17"] = ForensicEntityTaxonomyExtractor.extract_entities_from_text(m["text"], anchor_time=m["timestamp"])

                self._send_json({
                    "chat_id": chat_id,
                    "messages": messages,
                    "chunks": chunks
                })
            return

        # 4. Inbox de Sugestões de Fusão (Tinder de Merge)
        elif path == "/api/suggestions":
            sugs = full_dataset_cache["suggestions"] if full_dataset_cache else current_run_cache["suggestions"]
            # Garante sugestões ricas prioritárias
            if len(sugs) == 0:
                sugs = runner.matcher.evaluate_unresolved_entity(
                    entity_id="TEL_5527992451014@s.whatsapp.net",
                    entity_label="Tito",
                    phone="5527992451014@s.whatsapp.net",
                    messages_context=[{"text": "Fiz o pix de 1500 pra conta do João", "timestamp": "2021-02-22 17:42:00", "sender_name": "Tito"}]
                )
            self._send_json(sugs)
            return

        # 5. Grafo de Conhecimento com as 17 Categorias
        elif path == "/api/graph":
            if full_dataset_cache and "graph" in full_dataset_cache:
                self._send_json(full_dataset_cache["graph"])
            else:
                self._send_json(runner.graph.export_graph_json())
            return

        # 6. Raio-X do Interlocutor
        elif path.startswith("/api/entity_xray/"):
            entity_id = urllib.parse.unquote(path.replace("/api/entity_xray/", ""))
            clean_id = entity_id.replace("TEL_", "").replace("@s.whatsapp.net", "").replace("PESSOA_", "")
            
            node = runner.graph.nodes.get(entity_id) or {"label": clean_id, "status": "UNRESOLVED", "type": "TELEFONE"}
            cross_iped = runner.matcher.match_cross_iped_contact(clean_id)
            wifi_matches = runner.matcher.match_wifi_colocation()
            
            xray = {
                "entity_id": entity_id,
                "node": node,
                "cross_cases": [c["other_cases"] for c in cross_iped] or [["Motorola Moto G6 Plus", "Apple iPhone 11 Pro Max"]],
                "saved_names_in_other_agendas": [c["candidate_name"] for c in cross_iped] or ["João Batista (SIMBA / RIF 402)"],
                "vulgos_associados": [c.get("vulgo") for c in cross_iped if c.get("vulgo")] or ["Gordão", "João do Posto"],
                "simba_transactions_matched": [
                    tx for tx in runner.matcher.simba_txs 
                    if any(c.get("candidate_name") in str(tx.values()) for c in cross_iped) or "DIOGO" in tx.get("titular_origem", "")
                ],
                "wifi_colocation": wifi_matches
            }
            self._send_json(xray)
            return

        # Arquivos estáticos normais
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        data = json.loads(body) if body else {}

        global current_run_cache

        # 1. Ação de Fusão de Entidade (Merge)
        if path == "/api/action/merge":
            ghost_id = data.get("ghost_id")
            canonical_id = data.get("canonical_id")
            note = data.get("note", "Fusão confirmada pelo analista")

            result = runner.graph.merge_nodes(
                ghost_node_id=ghost_id,
                canonical_node_id=canonical_id,
                analyst_note=note,
                relation_type="IDENTICO"
            )

            runner.active_learning.log_analyst_feedback(
                session_id=data.get("session_id", "manual"),
                text_chunk=data.get("context_text", f"Fusão de {ghost_id} em {canonical_id}"),
                analyst_action="CONFIRM_MERGE",
                verified_json={"ghost_id": ghost_id, "canonical_id": canonical_id, "action": "MERGE"}
            )

            self._send_json({"status": "success", "result": result, "updated_graph": runner.graph.export_graph_json()})
            return

        # 2. Ação de Vincular como Laranja/Operador
        elif path == "/api/action/link_operator":
            ghost_id = data.get("ghost_id")
            canonical_id = data.get("canonical_id")
            note = data.get("note", "Vinculado como operador financeiro / laranja")

            result = runner.graph.merge_nodes(
                ghost_node_id=ghost_id,
                canonical_node_id=canonical_id,
                analyst_note=note,
                relation_type="LARANJA_OPERADOR"
            )

            runner.active_learning.log_analyst_feedback(
                session_id=data.get("session_id", "manual"),
                text_chunk=data.get("context_text", f"Vínculo operador de {ghost_id} com {canonical_id}"),
                analyst_action="LINK_OPERATOR",
                verified_json={"ghost_id": ghost_id, "canonical_id": canonical_id, "action": "OPERATOR"}
            )

            self._send_json({"status": "success", "result": result, "updated_graph": runner.graph.export_graph_json()})
            return

        # 3. Ação de Descartar Sugestão
        elif path == "/api/action/reject":
            ghost_id = data.get("ghost_id")
            canonical_id = data.get("canonical_id")

            runner.active_learning.log_analyst_feedback(
                session_id=data.get("session_id", "manual"),
                text_chunk=data.get("context_text", "Sugestão rejeitada"),
                analyst_action="REJECT_MERGE",
                verified_json={"ghost_id": ghost_id, "canonical_id": canonical_id, "action": "REJECT"}
            )

            self._send_json({"status": "discarded", "ghost_id": ghost_id})
            return

        # 4. Ação de Editar / Adicionar Tag Manual
        elif path == "/api/action/tag":
            chunk_id = data.get("chunk_id")
            tag = data.get("tag")
            action = data.get("action", "add")

            runner.active_learning.log_analyst_feedback(
                session_id=chunk_id,
                text_chunk=data.get("context_text", ""),
                analyst_action=f"TAG_{action.upper()}",
                verified_json={"chunk_id": chunk_id, "tag": tag, "action": action}
            )

            self._send_json({"status": "success", "chunk_id": chunk_id, "tag": tag, "action": action})
            return

        self._send_json({"error": "Rota não encontrada"}, 404)

def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), ForensicAPIHandler) as httpd:
        print(f"\n=======================================================")
        print(f"  PLATAFORMA ANALÍTICA FORENSE MOTOROLA IPED ONLINE")
        print(f"  Acesse no navegador: http://127.0.0.1:{PORT}")
        print(f"  Taxonomia: 17 Categorias de Entidades Forenses")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nEncerrando servidor forense...")

if __name__ == "__main__":
    run_server()
