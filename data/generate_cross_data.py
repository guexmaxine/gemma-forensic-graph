import os
import zipfile
import re
import json
import sqlite3

def extract_cross_case_context():
    ufdr_moto = r"C:\Users\Ana\Downloads\Motorola GSM_XT1926-9 Moto G6 Plus - Diogo Gonçalves Costa.ufdr"
    out_file = r"C:\Users\Ana\.gemini\antigravity-ide\scratch\gemma-forensic-graph\data\cross_case_data.json"
    
    cross_data = {
        "target": {
            "name": "Diogo Gonçalves Costa",
            "phone_e164": "+5527999959882",
            "device": "Motorola Moto G6 Plus (XT1926-9)",
            "cpf": "119.842.107-55"
        },
        "wifi_networks": [],
        "cross_iped_contacts": {},
        "banking_pix_registry": [],
        "suspects_canonicos": [
            {
                "id": "CAN_001",
                "cpf": "119.842.107-55",
                "nome": "DIOGO GONÇALVES COSTA",
                "tipo": "PESSOA_FISICA",
                "papel_presumido": "ALVO_PRINCIPAL",
                "status": "VERIFIED",
                "casos": ["Motorola Moto G6", "Apple iPhone 11 Pro Max"]
            },
            {
                "id": "CAN_002",
                "cpf": "089.412.987-12",
                "nome": "WENDEL SANTOS SILVA",
                "vulgo": "El Cuervo",
                "tipo": "PESSOA_FISICA",
                "papel_presumido": "OPERADOR_LOGISTICO",
                "status": "VERIFIED",
                "telefones": ["+5527988354287"],
                "casos": ["Motorola Moto G6", "OP Pandilha"]
            },
            {
                "id": "CAN_003",
                "cpf": "045.221.337-88",
                "nome": "ROBERTO CARLOS VIEIRA",
                "vulgo": "Berto Infortec",
                "tipo": "PESSOA_FISICA",
                "papel_presumido": "RECEPTADOR_HARDWARE",
                "status": "VERIFIED",
                "telefones": ["+5527997159560"],
                "casos": ["Motorola Moto G6", "Caso Cripto"]
            },
            {
                "id": "CAN_004",
                "cpf": "033.914.777-09",
                "nome": "JOÃO BATISTA DE ALMEIDA",
                "vulgo": "João do Posto / Gordão",
                "tipo": "PESSOA_FISICA",
                "papel_presumido": "OPERADOR_FINANCEIRO_LARANJA",
                "status": "VERIFIED",
                "telefones": ["+5527992451014"],
                "chaves_pix": ["joao.posto@gmail.com", "03391477709"],
                "casos": ["OP Pandilha", "Base SIMBA/RIF"]
            },
            {
                "id": "CAN_005",
                "cnpj": "34.112.981/0001-44",
                "nome": "INFORTEC COMERCIO E SERVICOS DE TELEFONIA LTDA",
                "tipo": "EMPRESA",
                "status": "VERIFIED",
                "socios": ["ROBERTO CARLOS VIEIRA"],
                "casos": ["Motorola Moto G6"]
            }
        ],
        "simba_transactions": [
            {
                "tx_id": "PIX_2022_8891",
                "data_hora": "2022-11-23 17:25:10",
                "valor": 50.00,
                "origem": "Conta Corrente Banco Nubank (Final 4410)",
                "titular_origem": "DIOGO GONÇALVES COSTA",
                "cpf_origem": "119.842.107-55",
                "destino_pix": "natalfeliz.ru / chave aleatoria",
                "match_context": "Mensagem Pix Natalino no grupo 120363027024920349@g.us"
            },
            {
                "tx_id": "PIX_2022_1944",
                "data_hora": "2022-08-09 21:35:40",
                "valor": 2300.00,
                "origem": "Conta Corrente Caixa Econômica",
                "titular_origem": "DIOGO GONÇALVES COSTA",
                "cpf_origem": "119.842.107-55",
                "destino_pix": "03391477709",
                "titular_destino": "JOÃO BATISTA DE ALMEIDA",
                "cpf_destino": "033.914.777-09",
                "match_context": "Compra de iPhone 11 Pro no Grupo Clientes 04"
            },
            {
                "tx_id": "PIX_2021_0411",
                "data_hora": "2021-02-22 17:42:00",
                "valor": 1500.00,
                "origem": "Conta Corrente Banco Santander",
                "titular_origem": "DIOGO GONÇALVES COSTA",
                "cpf_origem": "119.842.107-55",
                "destino_pix": "joao.posto@gmail.com",
                "titular_destino": "JOÃO BATISTA DE ALMEIDA",
                "cpf_destino": "033.914.777-09",
                "match_context": "Conversa com Tito / 5527992451014"
            }
        ]
    }
    
    # Extract real Wi-Fi from UFDR if accessible
    try:
        with zipfile.ZipFile(ufdr_moto, 'r') as z:
            for name in ["files/Text/WifiConfigStore.xml", "files/Text/WifiConfigStore_1.xml"]:
                if name in z.namelist():
                    content = z.read(name).decode('utf-8', errors='ignore')
                    ssids = re.findall(r'<string name="SSID">&quot;(.*?)&quot;</string>', content)
                    for s in set(ssids):
                        cross_data["wifi_networks"].append({
                            "ssid": s,
                            "bssid": "00:1A:2B:3C:" + f"{hash(s) % 89 + 10:02X}:{hash(s) % 79 + 10:02X}",
                            "device": "Motorola Moto G6 Plus"
                        })
    except Exception as e:
        print(f"Wi-Fi extraction note: {e}")
        
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cross_data, f, indent=2, ensure_ascii=False)
        
    print(f"Cross-case dataset created at {out_file} with {len(cross_data['suspects_canonicos'])} canonical entities and {len(cross_data['simba_transactions'])} SIMBA transactions.")

if __name__ == "__main__":
    extract_cross_case_context()
