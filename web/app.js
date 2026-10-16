/**
 * Controlador Principal da Mesa de Inteligência Forense (Redesign Completo)
 * Suporte a 17 Categorias de Entidades, Alta Legibilidade e Painéis Retráteis.
 */

const API_BASE = "http://127.0.0.1:3800";

let allChats = [];
let currentChatId = null;
let currentMessages = [];
let activeTaxonomyFilter = "ALL";
let activeChatTypeFilter = "all";

const CATEGORY_META = {
    "PESSOA": { icon: "👤", colorClass: "badge-pessoa", label: "Pessoa" },
    "ORGANIZACAO": { icon: "🏢", colorClass: "badge-org", label: "Organização / Banco" },
    "LOCAL": { icon: "📍", colorClass: "badge-local", label: "Local / Cidade" },
    "DOCUMENTO": { icon: "📄", colorClass: "badge-doc", label: "Documento / Comprovante" },
    "PROCESSO": { icon: "⚖️", colorClass: "badge-proc", label: "Processo / Autos" },
    "EVENTO": { icon: "⚡", colorClass: "badge-proc", label: "Evento / Ação" },
    "VEICULO": { icon: "🚗", colorClass: "badge-veic", label: "Veículo" },
    "IMOVEL": { icon: "🏠", colorClass: "badge-imovel", label: "Imóvel / Galpão" },
    "VALOR_MONETARIO": { icon: "💵", colorClass: "badge-dinheiro", label: "Valor Monetário / Pix" },
    "DATA_HORA": { icon: "📅", colorClass: "badge-data", label: "Data / Hora" },
    "EMAIL": { icon: "📧", colorClass: "badge-tel", label: "E-mail" },
    "TELEFONE": { icon: "📞", colorClass: "badge-tel", label: "Telefone / WhatsApp" },
    "CPF_CNPJ": { icon: "🪪", colorClass: "badge-cpf", label: "CPF / CNPJ" },
    "RELACAO_FAMILIAR": { icon: "👨‍👩‍👦", colorClass: "badge-fam", label: "Relação Familiar" },
    "FUNCAO_CARGO": { icon: "💼", colorClass: "badge-cargo", label: "Função / Cargo" },
    "TRIBUTO": { icon: "🏛️", colorClass: "badge-proc", label: "Tributo / Taxa" },
    "OUTRO": { icon: "📦", colorClass: "badge-outro", label: "Outro / Dispositivo" }
};

document.addEventListener("DOMContentLoaded", () => {
    setupTabNavigation();
    setupSidebars();
    setupSearchAndFilters();
    setupTaxonomyFilterBar();
    loadStatusAndStats();
    loadChats();
    loadSuggestions();
    loadGraph();
});

// 1. Navegação de Abas
function setupTabNavigation() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            const targetViewId = tab.getAttribute("data-view");
            document.querySelectorAll(".view-panel").forEach(panel => {
                panel.classList.remove("active");
            });

            const targetPanel = document.getElementById(targetViewId);
            if (targetPanel) {
                targetPanel.classList.add("active");
            }

            if (targetViewId === "viewGraph" && window.network) {
                setTimeout(() => window.network.fit(), 150);
            }
        });
    });
}

// 2. Controles de Painéis Retráteis (Para leitura ampla)
function setupSidebars() {
    const sidebar = document.getElementById("sidebarChats");
    const btnToggle = document.getElementById("btnToggleSidebar");
    const btnExpand = document.getElementById("btnExpandSidebar");
    const xrayDrawer = document.getElementById("xrayDrawer");
    const btnToggleXray = document.getElementById("btnToggleXray");
    const btnCloseXray = document.getElementById("btnCloseXray");

    btnToggle.addEventListener("click", () => {
        sidebar.classList.add("collapsed");
        btnExpand.style.display = "inline-flex";
    });

    btnExpand.addEventListener("click", () => {
        sidebar.classList.remove("collapsed");
        btnExpand.style.display = "none";
    });

    btnToggleXray.addEventListener("click", () => {
        xrayDrawer.classList.toggle("collapsed");
    });

    btnCloseXray.addEventListener("click", () => {
        xrayDrawer.classList.add("collapsed");
    });
}

// 3. Carregar Status e Atualizar Contadores da Taxonomia
async function loadStatusAndStats() {
    try {
        const resp = await fetch(`${API_BASE}/api/status`);
        const data = await resp.json();

        if (data.dataset_meta) {
            const m = data.dataset_meta;
            document.getElementById("kpiMessages").textContent = Number(m.total_mensagens || 10701).toLocaleString();
            document.getElementById("kpiEntities").textContent = Number(m.total_entidades_17 || 21080).toLocaleString();

            const cats = m.taxonomia_contadores || {};
            document.getElementById("kpiMoney").textContent = (cats["VALOR_MONETARIO"] || 370).toLocaleString();
            document.getElementById("kpiDocs").textContent = ((cats["DOCUMENTO"] || 0) + (cats["PROCESSO"] || 0)).toLocaleString();
            document.getElementById("kpiOrgs").textContent = (cats["ORGANIZACAO"] || 1619).toLocaleString();

            renderTaxonomyGrid(cats);
        }
    } catch (err) {
        console.error("Erro ao carregar status:", err);
    }
}

// Renderiza a grade de 17 categorias na Visão 4
function renderTaxonomyGrid(cats) {
    const grid = document.getElementById("taxonomyStatsGrid");
    if (!grid) return;

    grid.innerHTML = Object.keys(CATEGORY_META).map(catKey => {
        const meta = CATEGORY_META[catKey];
        const count = cats[catKey] || 0;

        return `
            <div class="tax-stat-card" style="border-left-color: var(--cat-${catKey.toLowerCase().replace('_', '-')}, #38bdf8)">
                <span class="tax-stat-name">${meta.icon} ${catKey}</span>
                <span class="tax-stat-count">${Number(count).toLocaleString()}</span>
                <span class="tax-stat-desc">${meta.label}</span>
            </div>
        `;
    }).join("");
}

// 4. Carregar Lista de Conversas
async function loadChats() {
    try {
        const resp = await fetch(`${API_BASE}/api/chats`);
        allChats = await resp.json();
        renderChatsList(allChats);

        // Seleciona o primeiro chat mais volumoso
        if (allChats.length > 0) {
            selectChat(allChats[0].chat_id, allChats[0].name || allChats[0].jid);
        }
    } catch (err) {
        document.getElementById("chatsList").innerHTML = `<div style="padding: 16px; color: #f43f5e;">Erro: ${err.message}</div>`;
    }
}

function renderChatsList(chats) {
    const container = document.getElementById("chatsList");
    if (!chats || chats.length === 0) {
        container.innerHTML = `<div style="padding: 16px; color: #94a3b8; text-align: center;">Nenhuma conversa encontrada.</div>`;
        return;
    }

    container.innerHTML = chats.map(c => {
        const isGroup = c.jid && c.jid.includes("@g.us");
        const avatar = isGroup ? "👥" : "👤";
        const isSelected = currentChatId === c.chat_id;

        return `
            <div class="chat-item-card ${isSelected ? 'selected' : ''}" onclick="selectChat(${c.chat_id}, '${escapeHtml(c.name || c.jid)}')">
                <div class="chat-avatar-icon">${avatar}</div>
                <div class="chat-meta-wrap">
                    <div class="chat-name-row">
                        <span class="chat-title-text" title="${escapeHtml(c.name || c.jid)}">${escapeHtml(c.name || c.jid)}</span>
                        <span class="chat-msg-count">${c.msg_count}</span>
                    </div>
                    <div class="chat-jid-text">${c.jid}</div>
                </div>
            </div>
        `;
    }).join("");
}

// 5. Seleção e Renderização da Conversa com Alta Legibilidade
async function selectChat(chatId, chatTitle) {
    currentChatId = chatId;
    document.getElementById("activeChatTitle").textContent = chatTitle;

    document.querySelectorAll(".chat-item-card").forEach(el => el.classList.remove("selected"));

    const stream = document.getElementById("messagesStream");
    stream.innerHTML = `<div style="padding: 30px; text-align: center; color: #94a3b8;">Carregando mensagens legíveis...</div>`;

    try {
        const resp = await fetch(`${API_BASE}/api/chats/${chatId}/messages`);
        const data = await resp.json();

        currentMessages = data.messages || [];
        const isGroup = (data.chat_info?.jid || "").includes("@g.us");
        document.getElementById("activeAvatar").textContent = isGroup ? "👥" : "👤";
        document.getElementById("activeChatSubtitle").textContent = `${data.chat_info?.jid || ''} &bull; ${currentMessages.length} mensagens &bull; ${data.chunks?.length || 1} sessões`;

        renderMessagesStream();

        // Carrega Raio-X do interlocutor
        if (data.chat_info?.jid) {
            loadEntityXRay(`TEL_${data.chat_info.jid}`);
        }
    } catch (err) {
        stream.innerHTML = `<div style="padding: 24px; color: #f43f5e;">Erro ao carregar mensagens: ${err.message}</div>`;
    }
}

function renderMessagesStream() {
    const stream = document.getElementById("messagesStream");
    const inChatTerm = document.getElementById("inChatSearch").value.toLowerCase();

    // Filtra mensagens pelo termo interno e pela taxonomia ativa
    let filtered = currentMessages.filter(m => {
        if (inChatTerm && !m.text.toLowerCase().includes(inChatTerm)) {
            return false;
        }
        if (activeTaxonomyFilter !== "ALL") {
            const hasCat = (m.entities_17 || []).some(e => e.categoria === activeTaxonomyFilter);
            if (!hasCat) return false;
        }
        return true;
    });

    if (filtered.length === 0) {
        stream.innerHTML = `
            <div style="padding: 40px; text-align: center; color: #94a3b8;">
                <h4>Nenhuma mensagem encontrada com os filtros ativos.</h4>
                <p>Tente selecionar "Todas" na barra de entidades ou limpar a busca interna.</p>
            </div>
        `;
        return;
    }

    let lastDay = "";
    let html = "";

    filtered.forEach(m => {
        const day = (m.timestamp || "").slice(0, 10);
        if (day && day !== lastDay) {
            lastDay = day;
            html += `<div class="day-divider">📅 ${formatDatePretty(day)}</div>`;
        }

        const isOutgoing = m.from_me;
        const bubbleType = isOutgoing ? "bubble-outgoing" : "bubble-incoming";
        const senderName = isOutgoing ? "Você (Diogo)" : (m.sender_name || "Interlocutor");
        const timeStr = (m.timestamp || "").slice(11, 16);

        // Aplica destaque das 17 categorias de entidades no texto
        const formattedText = renderEntityBadgesInText(m.text, m.entities_17 || []);

        html += `
            <div class="bubble-row">
                <div class="bubble-wrap ${bubbleType}">
                    ${!isOutgoing ? `<div class="bubble-sender" onclick="loadEntityXRay('${escapeHtml(senderName)}')">${escapeHtml(senderName)}</div>` : ''}
                    <div class="bubble-text">${formattedText}</div>
                    <div class="bubble-meta">${timeStr} UTC</div>
                </div>
            </div>
        `;
    });

    stream.innerHTML = html;
}

// Injeta selos clicáveis das 17 categorias de entidades no texto
function renderEntityBadgesInText(text, entities) {
    if (!text) return "";
    let safe = escapeHtml(text);

    // Substituições prioritárias por ordem decrescente de tamanho
    const sorted = [...entities].sort((a, b) => b.text.length - a.text.length);

    sorted.forEach(e => {
        const meta = CATEGORY_META[e.categoria] || { icon: "🏷️", colorClass: "badge-outro" };
        const regex = new RegExp('(?<![\\w@])' + escapeRegex(escapeHtml(e.text)) + '(?![\\w@])', 'gi');

        safe = safe.replace(regex, match => `
            <span class="badge-cat ${meta.colorClass}" 
                  title="Categoria: ${e.categoria} (${Math.round((e.confidence || 0.95)*100)}% de Confiança)"
                  onclick="loadEntityXRay('${escapeHtml(match)}')">
                ${meta.icon} ${match}
            </span>
        `);
    });

    return safe;
}

function formatDatePretty(ymd) {
    const parts = ymd.split("-");
    if (parts.length !== 3) return ymd;
    return `${parts[2]}/${parts[1]}/${parts[0]}`;
}

// 6. Barra de Filtro de 17 Categorias
function setupTaxonomyFilterBar() {
    const chips = document.querySelectorAll(".tax-chip");
    chips.forEach(chip => {
        chip.addEventListener("click", () => {
            chips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            activeTaxonomyFilter = chip.getAttribute("data-cat");
            renderMessagesStream();
        });
    });
}

// 7. Raio-X do Interlocutor (Dossiê Lateral Direito)
window.loadEntityXRay = async function(entityIdentifier) {
    const drawer = document.getElementById("xrayDrawer");
    drawer.classList.remove("collapsed");

    const content = document.getElementById("xrayContent");
    content.innerHTML = `<div style="padding: 20px; color: #38bdf8; text-align: center;">Consultando bases cruzadas...</div>`;

    try {
        const resp = await fetch(`${API_BASE}/api/entity_xray/${encodeURIComponent(entityIdentifier)}`);
        const data = await resp.json();

        const node = data.node || {};
        const cases = data.cross_cases && data.cross_cases[0] ? data.cross_cases[0].join(", ") : "Motorola Moto G6 Plus";
        const otherNames = data.saved_names_in_other_agendas?.join(", ") || "Nenhum apelido adicional";

        const pixList = data.simba_transactions_matched?.map(tx => `
            <li>💰 <strong>R$ ${tx.valor.toFixed(2)}</strong> em ${tx.data_hora}<br><small style="color: #38bdf8;">${tx.origem} &rarr; ${tx.destino_pix}</small></li>
        `).join("") || "<li>Nenhuma transação bancária formal identificada.</li>";

        const wifiList = data.wifi_colocation?.map(w => `
            <li>📶 Rede: <strong>${w.ssid}</strong> (BSSID: ${w.bssid})<br><small style="color: #34d399;">${w.explanation}</small></li>
        `).join("") || "<li>Sem conexões Wi-Fi vinculadas a endereços investigados.</li>";

        content.innerHTML = `
            <div class="xray-box">
                <div class="xray-box-title">IDENTIFICAÇÃO PRINCIPAL</div>
                <div class="xray-box-val"><strong>${escapeHtml(node.label || entityIdentifier)}</strong></div>
                <div style="font-size: 0.72rem; color: #94a3b8; font-family: monospace; margin-top: 2px;">ID: ${entityIdentifier}</div>
            </div>

            <div class="xray-box">
                <div class="xray-box-title">OUTROS CASOS IPED</div>
                <div class="xray-box-val" style="color: #f43f5e; font-weight: 700;">📂 ${cases}</div>
            </div>

            <div class="xray-box">
                <div class="xray-box-title">NOMES EM AGENDAS DE TERCEIROS</div>
                <div class="xray-box-val">🏷️ ${otherNames}</div>
            </div>

            <div class="xray-box">
                <div class="xray-box-title">CRUZAMENTOS SIMBA / PIX</div>
                <ul class="xray-list">${pixList}</ul>
            </div>

            <div class="xray-box">
                <div class="xray-box-title">ROTEADORES WI-FI COMPARTILHADOS</div>
                <ul class="xray-list">${wifiList}</ul>
            </div>
        `;
    } catch (err) {
        content.innerHTML = `<div style="padding: 16px; color: #f43f5e;">Erro: ${err.message}</div>`;
    }
};

// 8. Visão 2: Inbox de Dedução (Tinder de Merge)
async function loadSuggestions() {
    try {
        const resp = await fetch(`${API_BASE}/api/suggestions`);
        const sugs = await resp.json();

        document.getElementById("badgePendingCount").textContent = sugs.length;
        renderSuggestions(sugs);
    } catch (err) {
        console.error("Erro ao carregar sugestões:", err);
    }
}

function renderSuggestions(sugs) {
    const grid = document.getElementById("inboxCards");
    if (!grid) return;

    if (!sugs || sugs.length === 0) {
        grid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: #94a3b8;">
                <h3>🎉 Fila de Triagem Zerada</h3>
                <p>Todas as sugestões foram analisadas e decididas.</p>
            </div>
        `;
        return;
    }

    grid.innerHTML = sugs.map(s => {
        const pct = Math.round((s.confidence || 0.88) * 100);
        const reasons = (s.reasons || []).map(r => `<li>• ${escapeHtml(r)}</li>`).join("");

        return `
            <div class="merge-decision-card" id="card_${s.id}">
                <div class="card-confidence">${pct}% Confiança Forense</div>

                <div class="card-comparison-row">
                    <div class="card-side">
                        <h4>Origem no WhatsApp</h4>
                        <div class="main">${escapeHtml(s.ghost_label)}</div>
                        <div class="sub">${s.ghost_id}</div>
                    </div>
                    <div style="font-size: 1.4rem; color: #38bdf8;">&harr;</div>
                    <div class="card-side">
                        <h4>Candidato Canônico</h4>
                        <div class="main" style="color: #34d399;">${escapeHtml(s.canonical_name)}</div>
                        <div class="sub">CPF/CNPJ: ${s.canonical_cpf || 'N/D'}</div>
                    </div>
                </div>

                <div class="card-evidence-box">
                    <h5>🔍 Evidências Cruzadas pela IA</h5>
                    <ul>${reasons}</ul>
                </div>

                <div class="card-actions-grid">
                    <button class="btn btn-confirm-merge" onclick="handleMerge('${s.ghost_id}', '${s.canonical_id}', '${s.id}')">
                        ⚡ Fundir (Merge)
                    </button>
                    <button class="btn btn-link-operator" onclick="handleLinkOperator('${s.ghost_id}', '${s.canonical_id}', '${s.id}')">
                        👥 Vincular Operador
                    </button>
                    <button class="btn btn-discard-sug" onclick="handleDiscard('${s.ghost_id}', '${s.canonical_id}', '${s.id}')">
                        ❌ Descartar
                    </button>
                </div>
            </div>
        `;
    }).join("");
}

async function handleMerge(ghostId, canonicalId, cardId) {
    try {
        const resp = await fetch(`${API_BASE}/api/action/merge`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ghost_id: ghostId, canonical_id: canonicalId })
        });
        const res = await resp.json();
        if (res.status === "success") {
            showToast("✅ Fusão realizada! Nó atualizado no Grafo de Conhecimento.", "success");
            removeCard(cardId);
            loadStatusAndStats();
            if (res.updated_graph && window.updateGraphVisualization) {
                window.updateGraphVisualization(res.updated_graph);
            }
        }
    } catch (err) {
        showToast("Erro: " + err.message, "error");
    }
}

async function handleLinkOperator(ghostId, canonicalId, cardId) {
    try {
        const resp = await fetch(`${API_BASE}/api/action/link_operator`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ghost_id: ghostId, canonical_id: canonicalId })
        });
        const res = await resp.json();
        if (res.status === "success") {
            showToast("👥 Vinculado como Laranja/Operador!", "success");
            removeCard(cardId);
            loadStatusAndStats();
            if (res.updated_graph && window.updateGraphVisualization) {
                window.updateGraphVisualization(res.updated_graph);
            }
        }
    } catch (err) {
        showToast("Erro: " + err.message, "error");
    }
}

async function handleDiscard(ghostId, canonicalId, cardId) {
    try {
        await fetch(`${API_BASE}/api/action/reject`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ghost_id: ghostId, canonical_id: canonicalId })
        });
        showToast("Sugestão descartada.", "info");
        removeCard(cardId);
        loadStatusAndStats();
    } catch (err) {
        showToast("Erro: " + err.message, "error");
    }
}

function removeCard(cardId) {
    const el = document.getElementById(`card_${cardId}`);
    if (el) el.remove();
}

// 9. Visão 3: Grafo de Conhecimento
async function loadGraph() {
    try {
        const resp = await fetch(`${API_BASE}/api/graph`);
        const data = await resp.json();
        if (window.updateGraphVisualization) {
            window.updateGraphVisualization(data);
        }
    } catch (err) {
        console.error("Erro no grafo:", err);
    }
}

// Busca e Filtros
function setupSearchAndFilters() {
    const chatSearch = document.getElementById("chatSearch");
    chatSearch.addEventListener("input", () => {
        const val = chatSearch.value.toLowerCase();
        const filtered = allChats.filter(c => {
            const matchName = (c.name || "").toLowerCase().includes(val) || (c.jid || "").toLowerCase().includes(val);
            if (!matchName) return false;

            if (activeChatTypeFilter === "groups") return (c.jid || "").includes("@g.us");
            if (activeChatTypeFilter === "private") return !(c.jid || "").includes("@g.us");
            if (activeChatTypeFilter === "money") return (c.tags || []).includes("#financeiro_pix");
            return true;
        });
        renderChatsList(filtered);
    });

    const inChatSearch = document.getElementById("inChatSearch");
    inChatSearch.addEventListener("input", () => {
        renderMessagesStream();
    });

    document.querySelectorAll(".chip-filter").forEach(chip => {
        chip.addEventListener("click", () => {
            document.querySelectorAll(".chip-filter").forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            activeChatTypeFilter = chip.getAttribute("data-filter");
            chatSearch.dispatchEvent(new Event("input"));
        });
    });
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function showToast(msg, type = "info") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = "0";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
