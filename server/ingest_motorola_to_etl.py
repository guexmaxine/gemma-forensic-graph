"""
Script de Ingestão de Dados Forenses do WhatsApp Motorola no caso-16.db do etl-forense.
Carrega todos os chats e mensagens do msgstore.db e wa.db extraídos do Motorola
e popula as tabelas tb_conversa, tb_mensagem, tb_agenda_contato e tb_contato_telefone
respeitando o offset da operacao_id = 16 (16 * 10^10).
"""

import sqlite3
import os
import sys

# Configura saída UTF-8 para o console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MOTO_MSGSTORE = r"C:\Users\Ana\.gemini\antigravity-ide\scratch\gemma-forensic-graph\data\msgstore.db"
MOTO_WA = r"C:\Users\Ana\.gemini\antigravity-ide\scratch\gemma-forensic-graph\data\wa.db"
CASO_16_DB = r"C:\Users\Ana\etl-forense\data\casos\caso-16.db"

OPERACAO_ID = 16
FONTE_ID = 19
PLATAFORMA_ID = 1  # WhatsApp
ID_OFFSET = OPERACAO_ID * (10**10)  # 160000000000

def ingest_motorola_chats():
    print(f"Iniciando ingestão do WhatsApp Motorola em {CASO_16_DB}...")
    
    moto_conn = sqlite3.connect(MOTO_MSGSTORE)
    moto_conn.execute("ATTACH DATABASE ? AS wa", (MOTO_WA,))
    m_cur = moto_conn.cursor()

    dest_conn = sqlite3.connect(CASO_16_DB)
    d_cur = dest_conn.cursor()

    # 1. Carregar contatos da agenda (wa.db)
    m_cur.execute("""
        SELECT jid, display_name, wa_name, number
        FROM wa.wa_contacts
        WHERE display_name IS NOT NULL OR wa_name IS NOT NULL OR number IS NOT NULL
    """)
    contacts_rows = m_cur.fetchall()
    print(f"Total de contatos recuperados da agenda: {len(contacts_rows)}")

    # Limpar tabelas existentes no caso 16 para evitar duplicações
    d_cur.execute("DELETE FROM tb_mensagem;")
    d_cur.execute("DELETE FROM tb_conversa;")
    d_cur.execute("DELETE FROM tb_agenda_contato;")
    d_cur.execute("DELETE FROM tb_contato_telefone;")

    # Inserir contatos na agenda
    contato_id_map = {}
    c_seq = 1
    for r in contacts_rows:
        cid = ID_OFFSET + c_seq
        c_seq += 1
        jid, disp_name, wa_name, num = r
        nome_completo = disp_name or wa_name or jid
        contato_id_map[jid] = cid

        d_cur.execute("""
            INSERT INTO tb_agenda_contato (
                contato_id, fonte_id, id_externo_origem, nome_completo, nome_exibicao,
                origem_app, tipo_contato, criado_em
            ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (cid, FONTE_ID, jid, nome_completo, disp_name or wa_name, "WhatsApp", "CONTATO_EXTRACAO"))

    print(f"Inseridos {len(contato_id_map)} contatos em tb_agenda_contato.")

    # 2. Carregar todas as conversas ativas
    m_cur.execute("""
        SELECT 
            c._id,
            j.raw_string as jid,
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
    chats = m_cur.fetchall()
    print(f"Total de conversas ativas para inserção: {len(chats)}")

    conversa_id_map = {}
    conv_seq = 1

    for ch in chats:
        moto_cid, jid, subject, name, count, first_dt, last_dt = ch
        dest_conv_id = ID_OFFSET + conv_seq
        conv_seq += 1
        conversa_id_map[moto_cid] = dest_conv_id

        tipo_chat = "GRUPO" if "@g.us" in jid else "INDIVIDUAL"
        titulo = subject or name or jid

        d_cur.execute("""
            INSERT INTO tb_conversa (
                conversa_id, fonte_id, plataforma_id, id_externo_chat, tipo_chat,
                titulo_chat, canonical_chat_key, total_mensagens, primeira_utc, ultima_utc,
                n_participantes, criado_em
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            dest_conv_id, FONTE_ID, PLATAFORMA_ID, jid, tipo_chat,
            titulo, jid, count, first_dt or datetime.datetime.now().isoformat(), 
            last_dt or datetime.datetime.now().isoformat(),
            15 if tipo_chat == "GRUPO" else 2
        ))

    print(f"Inseridas {len(conversa_id_map)} conversas em tb_conversa.")

    # 3. Carregar todas as mensagens com texto
    m_cur.execute("""
        SELECT 
            m._id,
            m.chat_row_id,
            m.from_me,
            m.key_id,
            datetime(m.timestamp/1000, 'unixepoch') as iso_utc,
            m.text_data
        FROM message m
        WHERE m.text_data IS NOT NULL AND length(trim(m.text_data)) > 0
        ORDER BY m.timestamp ASC
    """)
    messages = m_cur.fetchall()
    print(f"Total de mensagens recuperadas: {len(messages)}")

    msg_seq = 1
    inserted_msgs = 0

    for msg in messages:
        m_id, chat_row_id, from_me, key_id, iso_utc, text = msg
        if chat_row_id not in conversa_id_map:
            continue

        dest_conv_id = conversa_id_map[chat_row_id]
        dest_msg_id = ID_OFFSET + msg_seq
        msg_seq += 1

        direcao = "ENVIADA" if from_me else "RECEBIDA"
        ts_utc = (iso_utc or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "Z"

        d_cur.execute("""
            INSERT INTO tb_mensagem (
                mensagem_id, conversa_id, fonte_id, id_externo_mensagem, direcao,
                tipo_mensagem, texto_conteudo, data_hora_original, data_hora_utc, criado_em
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            dest_msg_id, dest_conv_id, FONTE_ID, key_id or f"moto_{m_id}", direcao,
            "TEXTO", text, ts_utc, ts_utc
        ))
        inserted_msgs += 1

    # Inserir marcadores de crime nas conversas críticas
    marcadores = [
        (ID_OFFSET + 1, OPERACAO_ID, None, "Fraude / Golpe WhatsApp", 1, "IA_Gemma"),
        (ID_OFFSET + 2, OPERACAO_ID, None, "Lavagem / Ocultação Pix", 2, "IA_Gemma"),
        (ID_OFFSET + 3, OPERACAO_ID, None, "Receptação Celulares", 3, "IA_Gemma")
    ]
    for marc in marcadores:
        d_cur.execute("""
            INSERT OR REPLACE INTO tb_marcador (marcador_id, operacao_id, parent_id, nome, ordem, criado_por, criado_em)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, marc)

    dest_conn.commit()
    dest_conn.close()
    moto_conn.close()

    print(f"Sucesso! Inseridas {inserted_msgs} mensagens em tb_mensagem no caso-16.db.")
    print("Banco caso-16.db sincronizado com sucesso!")

if __name__ == "__main__":
    ingest_motorola_chats()
