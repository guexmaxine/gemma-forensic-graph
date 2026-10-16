"""
Módulo de Grafo de Conhecimento Forense (Property Knowledge Graph).
Gerencia entidades Canônicas (Verificadas) e Fantasmas/Candidatas (Não Resolvidas),
além de arestas direcionadas, tipadas e com pesos de confiança.
"""

import json
import uuid
import datetime
from typing import Dict, List, Any, Optional

class KnowledgeGraph:
    def __init__(self):
        # Nós: id -> Dict
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # Arestas: List[Dict]
        self.edges: List[Dict[str, Any]] = []
        # Histórico de Fusões (Merge Audit Trail)
        self.merge_history: List[Dict[str, Any]] = []

    def add_node(
        self,
        node_id: str,
        label: str,
        entity_type: str,
        status: str = "UNRESOLVED",  # VERIFIED ou UNRESOLVED
        canonical_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        cases: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Adiciona ou atualiza um nó no grafo."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if properties:
                node["properties"].update(properties)
            if tags:
                node["tags"] = list(set(node["tags"] + tags))
            if cases:
                node["cases"] = list(set(node["cases"] + cases))
            return node

        node = {
            "id": node_id,
            "label": label,
            "entity_type": entity_type,  # PESSOA_FISICA, EMPRESA, ALIAS_VULGO, TELEFONE, JID, PIX, CONTA, WIFI
            "status": status,            # VERIFIED, UNRESOLVED, MERGED
            "canonical_id": canonical_id or (node_id if status == "VERIFIED" else None),
            "properties": properties or {},
            "tags": tags or [],
            "cases": cases or ["Motorola Moto G6 Plus"],
            "created_at": datetime.datetime.now().isoformat()
        }
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship: str,  # ENVIOU_PIX, FALOU_COM, ORDENOU_PARA, MENCIONOU, CO_LOCALIZADO_COM, SOCIO_DE
        confidence: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """Adiciona uma aresta direcionada entre dois nós."""
        # Garante que os nós existam
        if source_id not in self.nodes:
            self.add_node(source_id, source_id, "IDENTIFICADOR_DIGITAL", "UNRESOLVED")
        if target_id not in self.nodes:
            self.add_node(target_id, target_id, "IDENTIFICADOR_DIGITAL", "UNRESOLVED")

        # Verifica se já existe aresta equivalente
        for edge in self.edges:
            if (
                edge["source"] == source_id and
                edge["target"] == target_id and
                edge["relationship"] == relationship
            ):
                # Atualiza peso ou propriedades
                edge["weight"] = edge.get("weight", 1) + 1
                edge["confidence"] = max(edge["confidence"], confidence)
                if properties:
                    edge["properties"].update(properties)
                return edge

        edge = {
            "id": f"edge_{uuid.uuid4().hex[:8]}",
            "source": source_id,
            "target": target_id,
            "relationship": relationship,
            "weight": 1,
            "confidence": round(confidence, 2),
            "timestamp": timestamp or datetime.datetime.now().isoformat(),
            "properties": properties or {}
        }
        self.edges.append(edge)
        return edge

    def merge_nodes(
        self,
        ghost_node_id: str,
        canonical_node_id: str,
        analyst_note: str = "",
        relation_type: str = "IDENTICO"  # IDENTICO ou LARANJA_OPERADOR
    ) -> Dict[str, Any]:
        """
        Funde uma entidade desconhecida/fantasma em uma entidade canônica.
        O nó fantasma aponta para a canônica e todas as arestas são herdadas.
        """
        if ghost_node_id not in self.nodes or canonical_node_id not in self.nodes:
            raise ValueError(f"Nós {ghost_node_id} ou {canonical_node_id} não encontrados no grafo.")

        ghost = self.nodes[ghost_node_id]
        canonical = self.nodes[canonical_node_id]

        # Auditoria da fusão
        record = {
            "ghost_id": ghost_node_id,
            "ghost_label": ghost["label"],
            "canonical_id": canonical_node_id,
            "canonical_label": canonical["label"],
            "relation_type": relation_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "note": analyst_note
        }
        self.merge_history.append(record)

        if relation_type == "IDENTICO":
            # Atualiza status do fantasma
            ghost["status"] = "MERGED"
            ghost["canonical_id"] = canonical_node_id

            # Adiciona apelidos / vulgos à canônica
            if "aliases" not in canonical["properties"]:
                canonical["properties"]["aliases"] = []
            if ghost["label"] not in canonical["properties"]["aliases"]:
                canonical["properties"]["aliases"].append(ghost["label"])

            # Mescla tags e casos
            canonical["tags"] = list(set(canonical["tags"] + ghost["tags"]))
            canonical["cases"] = list(set(canonical["cases"] + ghost["cases"]))

            # Redireciona arestas
            for edge in self.edges:
                if edge["source"] == ghost_node_id:
                    edge["source"] = canonical_node_id
                if edge["target"] == ghost_node_id:
                    edge["target"] = canonical_node_id
        else:
            # Vínculo como Laranja / Operador
            self.add_edge(
                source_id=ghost_node_id,
                target_id=canonical_node_id,
                relationship="OPERADOR_OU_LARANJA",
                confidence=0.95,
                properties={"vinculado_por_analista": True, "nota": analyst_note}
            )

        return {
            "status": "success",
            "merged": ghost_node_id,
            "into": canonical_node_id,
            "relation_type": relation_type
        }

    def calculate_centrality(self) -> Dict[str, int]:
        """Calcula o grau de centralidade (grau total de conexões) para ordenar triagem."""
        degrees = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            if edge["source"] in degrees:
                degrees[edge["source"]] += 1
            if edge["target"] in degrees:
                degrees[edge["target"]] += 1
        return degrees

    def export_graph_json(self) -> Dict[str, Any]:
        """Exporta dados para visualização D3 / Vis.js."""
        degrees = self.calculate_centrality()
        nodes_list = []
        for nid, data in self.nodes.items():
            nodes_list.append({
                "id": nid,
                "label": data["label"],
                "type": data["entity_type"],
                "status": data["status"],
                "canonical_id": data.get("canonical_id"),
                "degree": degrees.get(nid, 0),
                "tags": data.get("tags", []),
                "cases": data.get("cases", []),
                "properties": data.get("properties", {})
            })

        return {
            "nodes": nodes_list,
            "edges": self.edges,
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "verified_nodes": sum(1 for n in self.nodes.values() if n["status"] == "VERIFIED"),
                "unresolved_nodes": sum(1 for n in self.nodes.values() if n["status"] == "UNRESOLVED"),
                "merges_count": len(self.merge_history)
            }
        }
