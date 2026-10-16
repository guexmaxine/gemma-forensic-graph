"""
Módulo de Destilação e Aprendizado Ativo (Active Learning / Teacher-Student).
Registra as validações e correções do analista na UI em uma base gold_dataset.jsonl
para futuro fine-tuning de modelos menores (GLiNER / ModernBERT / QLoRA 3B-7B).
"""

import json
import os
import datetime
from typing import Dict, Any, List

class ActiveLearningManager:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

    def log_analyst_feedback(
        self,
        session_id: str,
        text_chunk: str,
        analyst_action: str,  # "CONFIRM_MERGE", "REJECT_MERGE", "EDIT_TAG", "MANUAL_LINK"
        verified_json: Dict[str, Any],
        user: str = "analista_gaeco"
    ) -> Dict[str, Any]:
        """Salva um exemplo validado no gold_dataset.jsonl."""
        entry = {
            "id": f"gold_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "session_id": session_id,
            "analyst_action": analyst_action,
            "analyst_user": user,
            "timestamp": datetime.datetime.now().isoformat(),
            "raw_text": text_chunk,
            "verified_annotations": verified_json
        }

        with open(self.dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return {"status": "saved", "entry_id": entry["id"]}

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de maturidade do gold dataset para fine-tuning."""
        if not os.path.exists(self.dataset_path):
            return {
                "total_examples": 0,
                "target_for_gliner": 1000,
                "progress_pct": 0.0,
                "ready_for_fine_tune": False
            }

        count = 0
        actions = {}
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
                    try:
                        obj = json.loads(line)
                        act = obj.get("analyst_action", "OUTRO")
                        actions[act] = actions.get(act, 0) + 1
                    except Exception:
                        pass

        target = 1000
        pct = min(100.0, round((count / target) * 100.0, 1))

        return {
            "total_examples": count,
            "actions_breakdown": actions,
            "target_for_gliner": target,
            "progress_pct": pct,
            "ready_for_fine_tune": count >= 500
        }
