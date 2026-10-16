# Gemma Forensic Graph: Extração de Entidades Forenses (17 Categorias) & Grafo de Conhecimento

Sistema completo de inteligência investigativa e Processamento de Linguagem Natural (NLP) especializado em extração pericial de entidades contextuais em investigações de crimes cibernéticos, fraudes financeiras e lavagem de dinheiro.

Combina um modelo neural **GLiNER Multilingual v2.1 (Fine-Tuned)** com **regras determinísticas de alta certeza (Regex Estrito)** e **calibração léxica forense**, integrando interface web interativa com visualização de grafo de conhecimento e módulo ativo de resolução de entidades.

---

## 📊 Benchmark Comparativo Oficial de 6 Métodos

Avaliado de forma estritamente cega sobre o **Gabarito de Ouro Forense** (20 mensagens amostrais com 57 entidades auditadas de alta complexidade):

| # | Abordagem Técnica | Precisão | Recall (Revocação) | F1-Score | Latência Média | Diagnóstico Forense |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | **Baseline Heurístico / Regex** | 83.7% | 63.2% | 72.0% | 0.26 ms/msg | Perde quase todos os nomes próprios; confunde saudações |
| **2** | **spaCy (`pt_core_news_sm`)** | 57.7% | 26.3% | 36.1% | 5.06 ms/msg | Incapaz de cobrir as 17 categorias periciais (só possui PER/ORG/LOC) |
| **3** | **GLiNER v2.1 (Zero-Shot Isolado)** | 66.1% | 64.9% | 65.5% | 206.5 ms/msg | Boa semântica, mas gera fragmentos em números e documentos |
| **4** | **Solução Híbrida (Zero-Shot)** | **91.2%** | **91.2%** | **91.2%** | 180.4 ms/msg | **Excelente equilíbrio global:** combina certeza matemática e semântica aberta |
| **5** | **GLiNER Fine-Tuned (Isolado)** | **67.3%** | 61.4% | 64.2% | 199.7 ms/msg | **+1.2% de precisão pura:** aprendeu jargões, bancos e nomes próprios do caso |
| **6** | **Solução Híbrida Fine-Tuned (IDEAL)** | **88.7%** | **82.5%** | **85.5%** | **176.5 ms/msg** | **Mais rápida e altamente precisa:** F1 de 85.5% com zero falsos positivos estruturados |

---

## 🏛️ Taxonomia Oficial das 17 Categorias Forenses

| # | Categoria Forense | Descrição e Critério de Mapeamento | Exemplos Reais |
| :-: | :--- | :--- | :--- |
| **1** | `PESSOA` | Nomes próprios, prenomes e vulgos de alvos ou interlocutores | *Wendel Praxedes*, *Diogo*, *Tito*, *Gordão* |
| **2** | `ORGANIZACAO` | Bancos, fintechs, órgãos judiciais, escritórios e empresas | *Nubank*, *Bradesco*, *Infortec*, *2ª Vara Criminal* |
| **3** | `LOCAL` | Municípios, bairros, logradouros e referências espaciais | *Vitória*, *Cariacica*, *Rocha Miranda*, *Rua dos Topázios* |
| **4** | `DOCUMENTO` | Comprovantes de transferência, recibos, NFs, CTPS, CNH, RG | *Carteira de trabalho*, *Comprovante Pix*, *Nota fiscal* |
| **5** | `PROCESSO` | Numeração de inquéritos policiais, autos judiciais e medidas | *0012450-88.2022.8.08.0024*, *IP 123/2022* |
| **6** | `EVENTO` | Encontros combinados, audiências, reuniões e operações | *Reunião de alinhamento*, *Entrega de malote* |
| **7** | `VEICULO` | Modelos automotivos, placas Mercosul e tipos de transporte | *Corolla*, *Van*, *Placa RQX4A12* |
| **8** | `IMOVEL` | Galpões, salas comerciais, postos de combustíveis, residências | *Galpão de estocagem*, *Posto*, *Casa 2* |
| **9** | `VALOR_MONETARIO`| Cifras em reais, menções a valores transacionados e limites | *R$ 2.300*, *1500*, *50mil*, *200mil* |
| **10**| `DATA_HORA` | Prazos operacionais, datas de nascimento e expressões relativas | *31/03/1998*, *dia 13 às 15:30*, *ontem*, *4 meses* |
| **11**| `EMAIL` | Endereços de correio eletrônico e chaves Pix associadas | *wendelpraxedes180@gmail.com* |
| **12**| `TELEFONE` | Linhas celulares e números de WhatsApp em formato nacional | *(27) 992234022*, *27998569389*, *5527988354287* |
| **13**| `CPF_CNPJ` | Chaves Pix e identificadores fiscais validados por máscara | *148.028.237-54*, *141.207.447-95*, *225.292.777-10* |
| **14**| `RELACAO_FAMILIAR`| Menções de parentesco direto utilizadas para ocultação de bens | *Mãe*, *irmão*, *esposa*, *marido*, *filho* |
| **15**| `FUNCAO_CARGO` | Posição na hierarquia ilícita ou profissional | *Patrão*, *gerente*, *advogado*, *contador*, *tatuador* |
| **16**| `TRIBUTO` | Impostos, taxas de cartório, retenções e benefícios estatais | *DARF*, *taxa de custas*, *seguro-desemprego* |
| **17**| `OUTRO` | Dispositivos periciados, IMEIs, números de série e ativos digitais| *Motorola Moto G6 Plus*, *iPhone 11 Pro Max*, *IMEI* |

---

## 🧠 Arquitetura da Solução Híbrida

```
Texto Bruto da Mensagem
         │
         ├──► [1. Regex Estrito Determinístico] ──► CPF/CNPJ, Processos CNJ, Telefones, Emails, Placas (100% Certeza)
         │
         ├──► [2. Filtro de Ruído Conversacional] ─► Remove stopwords e saudações ("boa noite", "olá", "ok")
         │
         ├──► [3. GLiNER Neural (Fine-Tuned)] ────► Extrai semântica aberta: Pessoa, Organização, Local, Imóvel
         │
         └──► [4. Calibração Léxica Especializada] ─► Mapeia hierarquias: FUNCAO_CARGO, RELACAO_FAMILIAR, TRIBUTO
```

---

## 📁 Estrutura do Repositório

```
gemma-forensic-graph/
├── data/
│   └── samples/
│       ├── test_sample_gold.json      # Gabarito de Ouro auditado (20 diálogos, 57 entidades)
│       ├── val_sample.json            # Amostra multimodal limpa (WhatsApp, OCR IPED, Whisper)
│       └── cross_case_data.json       # Vínculos e sobreposições entre casos periciais
├── models/
│   └── gliner_forensic_finetuned/
│       ├── gliner_config.json         # Arquitetura e cabeçotes de projeção de spans
│       ├── tokenizer.json             # Tokenizador DeBERTa multilingual
│       ├── tokenizer_config.json      # Configuração de tokenização
│       ├── finetune_metadata.json     # Metadados e curva de loss do fine-tuning
│       └── download_weights.py        # Script para baixar o checkpoint 'pytorch_model.bin'
├── reports/
│   ├── relatorio_nlp_comparativo.md   # Relatório Técnico Completo de NLP Forense
│   └── final_nlp_benchmark_comparison.json # Métricas comparativas oficiais em JSON
├── server/
│   ├── engine.py                      # API Server FastAPI / Grafo de Conhecimento
│   ├── entity_extractor_v2.py         # Extrator Híbrido Oficial de 17 Categorias
│   ├── train_gliner_forensic.py       # Pipeline de Fine-Tuning do GLiNER Forense
│   ├── deep_data_cleanliness_audit.py # Auditor de Limpeza e Integridade de Dados
│   ├── sanitize_finetune_data.py      # Sanitizador e Blindagem Anti-Leakage
│   ├── benchmark_framework.py         # Framework de Avaliação Comparativa
│   ├── active_learning.py             # Módulo Active Learning ("Tinder de Merge")
│   ├── knowledge_graph.py             # Modelagem de Grafo em NetworkX
│   └── cross_case_matcher.py          # Linkagem e correlação cruzada de alvos
├── web/
│   ├── index.html                     # Painel Forense Interativo (Dark Mode / Glassmorphism)
│   ├── styles.css                     # Estilos visuais modernos
│   ├── app.js                         # Controladores e consumo da API
│   └── graph.js                       # Renderizador D3.js / Canvas do Grafo
├── requirements.txt                   # Dependências do Python
└── README.md                          # Documentação oficial
```

---

## 🚀 Instalação e Execução

### 1. Clonar o Repositório
```bash
git clone https://github.com/guexmaxine/gemma-forensic-graph.git
cd gemma-forensic-graph
```

### 2. Instalar Dependências
```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

### 3. Obter os Pesos do Modelo Fine-Tuned
Você pode baixar os pesos treinados da Release oficial executando:
```bash
python models/gliner_forensic_finetuned/download_weights.py
```
*Ou treinar o modelo do zero localmente em seus próprios dados:*
```bash
python server/train_gliner_forensic.py
```

### 4. Executar o Benchmark Comparativo
```bash
python scratch/run_finetune_benchmark.py
```

### 5. Iniciar o Servidor do Grafo Forense
```bash
python server/engine.py
```
Abra o navegador em `http://127.0.0.1:3800` para explorar o Grafo de Conhecimento interativo, Timeline Pericial e o Tinder de Merge.

---

## 📜 Licença
Projeto desenvolvido para fins de perícia computacional e pesquisa em inteligência policial/forense.
Licença MIT.
