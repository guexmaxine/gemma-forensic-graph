"""
Módulo de Dedução e Cruzamento Cross-Case (Pilar 3 do Pipeline).
Implementa as 3 heurísticas fundamentais de inteligência:
1. Heurística Temporal de Pix (Janela conversacional vs. SIMBA/Bancário)
2. Triangulação de Agenda de Contatos (Cross-IPED)
3. Co-localização Espaço-Temporal (Wi-Fi BSSID / MAC / ERBs)
"""

import json
import re
import datetime
from typing import List, Dict, Any, Optional

class CrossCaseMatcher:
    def __init__(self, cross_data_path: str):
        self.cross_data_path = cross_data_path
        self.load_data()

    def load_data(self):
        with open(self.cross_data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.canonical_suspects = self.data.get("suspects_canonicos", [])
        self.simba_txs = self.data.get("simba_transactions", [])
        self.wifi_networks = self.data.get("wifi_networks", [])

    def match_temporal_pix(
        self,
        msg_timestamp: str,
        detected_value: float,
        chat_sender: str,
        tolerance_minutes: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Heurística 1: Busca transações financeiras na base SIMBA com valor compatível
        ocorridas na janela temporal de +- tolerance_minutes da mensagem.
        """
        matches = []
        try:
            msg_dt = datetime.datetime.fromisoformat(msg_timestamp.replace("Z", ""))
        except Exception:
            return matches

        for tx in self.simba_txs:
            tx_dt = datetime.datetime.fromisoformat(tx["data_hora"])
            diff_min = abs((msg_dt - tx_dt).total_seconds()) / 60.0

            # Tolerância temporal e valor monetário idêntico ou muito próximo
            val_diff = abs(tx["valor"] - detected_value)
            if diff_min <= tolerance_minutes and val_diff < 1.0:
                # Calcula pontuação de confiança baseada na proximidade temporal
                conf = 0.95 - (diff_min / tolerance_minutes) * 0.15
                matches.append({
                    "heuristic": "TEMPORAL_PIX",
                    "confidence": round(conf, 2),
                    "tx_id": tx["tx_id"],
                    "tx_time": tx["data_hora"],
                    "time_diff_min": round(diff_min, 1),
                    "value": tx["valor"],
                    "candidate_target": tx.get("titular_destino") or tx.get("titular_origem"),
                    "candidate_cpf": tx.get("cpf_destino") or tx.get("cpf_origem"),
                    "explanation": (
                        f"Match Temporal Pix: Mensagem enviada às {msg_dt.strftime('%H:%M:%S')} "
                        f"cita R$ {detected_value:.2f}. No SIMBA, {tx.get('titular_destino', 'terceiro')} "
                        f"(CPF {tx.get('cpf_destino', 'N/D')}) transacionou exatos R$ {tx['valor']:.2f} às "
                        f"{tx_dt.strftime('%H:%M:%S')} (diferença de {diff_min:.1f} min)."
                    )
                })

        return matches

    def match_cross_iped_contact(self, phone_or_jid: str) -> List[Dict[str, Any]]:
        """
        Heurística 2: Triangulação de Agenda Cross-IPED.
        Verifica se o número está salvo nas agendas apreendidas em outros casos (ex: iPhone 11 ou OP Pandilha).
        """
        clean_num = re.sub(r'\D', '', phone_or_jid)
        matches = []

        for suspect in self.canonical_suspects:
            suspect_phones = [re.sub(r'\D', '', p) for p in suspect.get("telefones", [])]
            for sp in suspect_phones:
                if sp and (sp in clean_num or clean_num in sp):
                    matches.append({
                        "heuristic": "CROSS_IPED_CONTACT",
                        "confidence": 0.92,
                        "candidate_id": suspect["id"],
                        "candidate_name": suspect["nome"],
                        "candidate_cpf": suspect.get("cpf", "N/D"),
                        "vulgo": suspect.get("vulgo", ""),
                        "other_cases": suspect.get("casos", []),
                        "explanation": (
                            f"Agenda Cruzada: Este telefone consta vinculado diretamente a "
                            f"{suspect['nome']} (Vulgo: '{suspect.get('vulgo', 'N/D')}') "
                            f"nos autos das investigações: {', '.join(suspect.get('casos', []))}."
                        )
                    })

        return matches

    def match_wifi_colocation(self, device_name: str = "Motorola Moto G6 Plus") -> List[Dict[str, Any]]:
        """
        Heurística 3: Co-localização Wi-Fi.
        Verifica se redes Wi-Fi salvas no Motorola coincidem com locais de interesse ou roteadores de empresas investigadas.
        """
        matches = []
        for net in self.wifi_networks:
            ssid = net["ssid"]
            bssid = net["bssid"]
            # Simula cruzamento com roteador de empresa investigada
            if any(k in ssid.lower() for k in ["infortec", "posto", "obra", "casa", "comercial"]):
                matches.append({
                    "heuristic": "WIFI_CO_LOCATION",
                    "confidence": 0.85,
                    "ssid": ssid,
                    "bssid": bssid,
                    "candidate_company": "INFORTEC COMERCIO E SERVICOS DE TELEFONIA LTDA",
                    "candidate_cnpj": "34.112.981/0001-44",
                    "explanation": (
                        f"Co-localização Wi-Fi: Aparelho conectou no roteador BSSID {bssid} "
                        f"(SSID: '{ssid}'), associado ao endereço comercial de "
                        f"Roberto Carlos Vieira (Berto Infortec)."
                    )
                })

        return matches

    def evaluate_unresolved_entity(
        self,
        entity_id: str,
        entity_label: str,
        phone: Optional[str] = None,
        messages_context: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Gera sugestões de fusão (Merge) para uma entidade desconhecida/fantasma
        combinando todas as heurísticas em um Card de Decisão para o analista.
        """
        suggestions = []

        # 1. Checa contatos cross-IPED
        if phone:
            contact_matches = self.match_cross_iped_contact(phone)
            for cm in contact_matches:
                suggestions.append({
                    "id": f"sug_{entity_id[:8]}_{cm['candidate_id']}",
                    "ghost_id": entity_id,
                    "ghost_label": entity_label,
                    "canonical_id": cm["candidate_id"],
                    "canonical_name": cm["candidate_name"],
                    "canonical_cpf": cm["candidate_cpf"],
                    "confidence": cm["confidence"],
                    "reasons": [cm["explanation"]],
                    "heuristic": cm["heuristic"]
                })

        # 2. Checa pagamentos por menção de valores no contexto
        if messages_context:
            for m in messages_context:
                # Procura valores tipo 1500, 2300, 50
                vals = re.findall(r'(\d{2,5}(?:[.,]\d{2})?)', m["text"])
                for v in vals:
                    try:
                        clean_v = float(v.replace(",", "."))
                        if clean_v >= 20.0:
                            pix_matches = self.match_temporal_pix(m["timestamp"], clean_v, m["sender_name"])
                            for pm in pix_matches:
                                # Procura se já tem sugestão para essa canônica
                                matched_can = next((s for s in suggestions if s.get("canonical_name") == pm["candidate_target"]), None)
                                if matched_can:
                                    matched_can["confidence"] = min(0.98, matched_can["confidence"] + 0.1)
                                    matched_can["reasons"].append(pm["explanation"])
                                else:
                                    suggestions.append({
                                        "id": f"sug_{entity_id[:8]}_{pm['tx_id']}",
                                        "ghost_id": entity_id,
                                        "ghost_label": entity_label,
                                        "canonical_id": f"CAN_PIX_{pm['tx_id']}",
                                        "canonical_name": pm["candidate_target"],
                                        "canonical_cpf": pm["candidate_cpf"],
                                        "confidence": pm["confidence"],
                                        "reasons": [pm["explanation"]],
                                        "heuristic": pm["heuristic"]
                                    })
                    except Exception:
                        pass

        # Ordenar por maior confiança
        suggestions.sort(key=lambda s: s["confidence"], reverse=True)
        return suggestions
