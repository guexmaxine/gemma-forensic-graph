/**
 * Controlador Visual do Grafo de Conhecimento Forense (Vis.js Network)
 * Renderiza nós Canônicos (Verdes), Fantasmas (Amarelos) e Arestas com pesos de confiança.
 */

let network = null;
let graphData = { nodes: new vis.DataSet([]), edges: new vis.DataSet([]) };

function initGraph() {
    const container = document.getElementById("networkGraph");
    if (!container) return;

    const options = {
        nodes: {
            shape: "dot",
            size: 20,
            font: {
                face: "Plus Jakarta Sans",
                size: 13,
                color: "#f8fafc"
            },
            borderWidth: 2,
            shadow: {
                enabled: true,
                color: "rgba(0,0,0,0.6)",
                size: 8,
                x: 3,
                y: 3
            }
        },
        edges: {
            width: 1.5,
            smooth: {
                type: "continuous"
            },
            font: {
                face: "JetBrains Mono",
                size: 10,
                color: "#94a3b8",
                align: "middle",
                background: "rgba(7, 11, 20, 0.7)"
            },
            arrows: {
                to: { enabled: true, scaleFactor: 0.6 }
            }
        },
        physics: {
            barnesHut: {
                gravitationalConstant: -4000,
                centralGravity: 0.3,
                springLength: 120,
                springConstant: 0.04
            },
            stabilization: {
                iterations: 120
            }
        },
        interaction: {
            hover: true,
            tooltipDelay: 100
        }
    };

    network = new vis.Network(container, graphData, options);

    // Evento de clique em nó para abrir Raio-X
    network.on("click", function (params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            if (window.loadEntityXRay) {
                window.loadEntityXRay(nodeId);
            }
        }
    });
}

function updateGraphVisualization(serverGraph) {
    if (!network) initGraph();

    const nodesArray = [];
    const edgesArray = [];

    serverGraph.nodes.forEach(n => {
        let color = { background: "#f59e0b", border: "#d97706" }; // Amarelo (Fantasma)
        let size = 18 + Math.min(n.degree * 2, 20);

        if (n.status === "VERIFIED") {
            color = { background: "#10b981", border: "#059669" }; // Verde (Canônico)
            size = 28;
        } else if (n.type === "PIX_TRANSAÇÃO") {
            color = { background: "#38bdf8", border: "#0284c7" };
            size = 16;
        }

        nodesArray.push({
            id: n.id,
            label: n.label,
            color: color,
            size: size,
            title: `<b>${n.label}</b><br>Tipo: ${n.type}<br>Status: ${n.status}<br>Grau Conexões: ${n.degree}`
        });
    });

    serverGraph.edges.forEach(e => {
        let edgeColor = "#64748b";
        let dashes = false;

        if (e.relationship === "ENVIOU_PIX") {
            edgeColor = "#38bdf8";
        } else if (e.relationship === "SUGERE_FUSAO_COM") {
            edgeColor = "#f43f5e";
            dashes = [5, 5];
        } else if (e.relationship === "OPERADOR_OU_LARANJA") {
            edgeColor = "#f59e0b";
        }

        const label = e.properties && e.properties.valor 
            ? `R$ ${e.properties.valor}` 
            : e.relationship;

        edgesArray.push({
            id: e.id,
            from: e.source,
            to: e.target,
            label: label,
            color: { color: edgeColor, highlight: "#38bdf8" },
            dashes: dashes
        });
    });

    graphData.nodes.clear();
    graphData.nodes.add(nodesArray);

    graphData.edges.clear();
    graphData.edges.add(edgesArray);

    // Re-estabiliza suavemente
    if (network) {
        network.fit();
    }
}

// Botões da Toolbar do Grafo
document.addEventListener("DOMContentLoaded", () => {
    initGraph();

    const btnReset = document.getElementById("btnGraphReset");
    if (btnReset) {
        btnReset.addEventListener("click", () => {
            if (network) network.fit();
        });
    }

    const btnPhysics = document.getElementById("btnGraphPhysics");
    if (btnPhysics) {
        let paused = false;
        btnPhysics.addEventListener("click", () => {
            paused = !paused;
            network.setOptions({ physics: { enabled: !paused } });
            btnPhysics.textContent = paused ? "▶️ Retomar Física" : "⏸️ Pausar Física";
        });
    }
});
