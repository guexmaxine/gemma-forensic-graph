"""
Motor de Fine-Tuning do GLiNER Forense Multimodal.
Treina o modelo urchade/gliner_multi-v2.1 nos dados limpos (WhatsApp, OCR Documentos, Áudios Whisper, Agenda).
Salva o checkpoint final em models/gliner_forensic_finetuned.
"""

import os
import sys
import json
import time
import random
import torch
from torch.utils.data import DataLoader
from gliner import GLiNER

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetune")
MODEL_OUT_DIR = os.path.join(BASE_DIR, "models", "gliner_forensic_finetuned")
os.makedirs(MODEL_OUT_DIR, exist_ok=True)
torch.set_num_threads(min(8, os.cpu_count() or 4))

ENTITIES_17 = [
    "PESSOA", "ORGANIZACAO", "LOCAL", "DOCUMENTO", "PROCESSO",
    "EVENTO", "VEICULO", "IMOVEL", "VALOR_MONETARIO", "DATA_HORA",
    "EMAIL", "TELEFONE", "CPF_CNPJ", "RELACAO_FAMILIAR", "FUNCAO_CARGO",
    "TRIBUTO", "OUTRO"
]

print("="*70)
print("  INICIANDO FINE-TUNING DO GLiNER FORENSE MULTIMODAL")
print("="*70)

# 1. Carregar datasets sanitizados
train_file = os.path.join(DATA_DIR, "train_dataset.json")
val_file = os.path.join(DATA_DIR, "val_dataset.json")

with open(train_file, "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(val_file, "r", encoding="utf-8") as f:
    val_data = json.load(f)

print(f"Dataset de Treino: {len(train_data)} amostras")
print(f"Dataset de Validação: {len(val_data)} amostras")

# 2. Carregar modelo pré-treinado
print("\nCarregando modelo base: urchade/gliner_multi-v2.1...")
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
dp = model.data_processor
print("Modelo carregado com sucesso.")

# Função de Collate Customizada e Segura
def custom_collate(batch_list):
    per_example_types = [ENTITIES_17 for _ in batch_list]
    raw_batch = dp.collate_raw_batch(batch_list, entity_types=per_example_types)
    tokenized = dp.tokenize_and_prepare_labels(raw_batch, prepare_labels=True)
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "words_mask": tokenized["words_mask"],
        "text_lengths": raw_batch["seq_length"].squeeze(-1),
        "span_idx": raw_batch["span_idx"],
        "span_mask": raw_batch["span_mask"],
        "labels": tokenized["labels"]
    }

BATCH_SIZE = 4
TOTAL_STEPS = 120
EVAL_EVERY = 20

# 3. Otimizador com decaimento de peso
model.train()
# Separação de parâmetros: backbone com LR menor, camadas de representação com LR maior
backbone_params = []
custom_params = []

for name, param in model.model.named_parameters():
    if "token_rep_layer" in name or "bert_layer" in name:
        backbone_params.append(param)
    else:
        custom_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1.5e-5},
    {"params": custom_params, "lr": 3.5e-5}
], weight_decay=0.01)

# Preparar batches de treino
random.seed(42)
shuffled_train = list(train_data)
random.shuffle(shuffled_train)

train_loader = DataLoader(shuffled_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
train_iter = iter(train_loader)

# Preparar batch fixo de validação (16 amostras)
val_sample = val_data[:16]
val_batch = custom_collate(val_sample)

print(f"\nIniciando Loop de Otimização ({TOTAL_STEPS} passos, batch_size={BATCH_SIZE})...")
t0 = time.time()
best_val_loss = float('inf')
loss_history = []

for step in range(1, TOTAL_STEPS + 1):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    optimizer.zero_grad()
    out = model.model(**batch)
    loss = out.loss
    loss.backward()
    
    # Clip de gradiente para estabilidade numérica
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    loss_val = round(loss.item(), 4)
    loss_history.append(loss_val)

    if step % 10 == 0 or step == 1:
        elapsed = round(time.time() - t0, 1)
        print(f"  Passo [{step:3d}/{TOTAL_STEPS}] - Loss Treino: {loss_val:.4f} | Tempo: {elapsed}s")

    # Avaliação periódica no conjunto de validação
    if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
        model.eval()
        with torch.no_grad():
            v_out = model.model(**val_batch)
            v_loss = round(v_out.loss.item(), 4)
        model.train()
        
        print(f"  >>> Avaliação Validação (Passo {step}): Loss = {v_loss:.4f} (Melhor anterior: {best_val_loss:.4f})")
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            print(f"      Novo recorde de Validação! Salvando checkpoint intermediário...")
            model.save_pretrained(MODEL_OUT_DIR)

t_total = round(time.time() - t0, 1)
print("\n" + "="*70)
print(f"Treinamento concluído em {t_total} segundos!")
print(f"Melhor Loss de Validação: {best_val_loss:.4f}")
print(f"Modelo Forense Fine-Tuned salvo em: {MODEL_OUT_DIR}")
print("="*70)

# Salvar metadata do fine-tuning
with open(os.path.join(MODEL_OUT_DIR, "finetune_metadata.json"), "w", encoding="utf-8") as f:
    json.dump({
        "base_model": "urchade/gliner_multi-v2.1",
        "total_steps": TOTAL_STEPS,
        "batch_size": BATCH_SIZE,
        "best_val_loss": best_val_loss,
        "training_time_s": t_total,
        "training_samples": len(train_data),
        "validation_samples": len(val_data),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }, f, ensure_ascii=False, indent=2)

print("Processo de Fine-Tuning finalizado com sucesso!")
