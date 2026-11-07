/**
 * Extreme Weather Event Grouping Studio - Client Application
 */

let dashboardData = null;
let pcaChart = null;
let radarChart = null;
let stormMap = null;
let currentPCAColor = "hierarchical";

// Cluster color mappings
const CLUSTER_COLORS_H = {
    1: "#0284c7", // Sky blue (Synoptic Winter)
    2: "#e11d48", // Rose/Red (Catastrophic Outbreaks)
    3: "#10b981", // Emerald (Convective Storms)
    4: "#f59e0b", // Amber
    5: "#8b5cf6"  // Purple
};

const CLUSTER_COLORS_D = {
    "-1": "#94a3b8", // Grey for noise
    "0": "#10b981",  // Emerald core convective
    "1": "#0284c7",  // Sky blue synoptic
    "2": "#e11d48"   // Rose extreme
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
    fetchDashboardData();
    setupClassifierForm();
});

// Tab Switching
function switchTab(tabId) {
    document.querySelectorAll(".tab-pane").forEach(el => el.classList.add("hidden"));
    document.querySelectorAll(".nav-tab").forEach(el => el.classList.remove("active"));

    const targetTab = document.getElementById(tabId);
    if (targetTab) {
        targetTab.classList.remove("hidden");
    }

    const btn = document.getElementById(`btn-${tabId}`);
    if (btn) {
        btn.classList.add("active");
    }

    // Handle map resizing
    if (tabId === "tab-map" && stormMap) {
        setTimeout(() => stormMap.invalidateSize(), 200);
    }

    // Handle PCA chart resizing
    if (tabId === "tab-pca") {
        setTimeout(() => {
            if (pcaChart) {
                pcaChart.resize();
                pcaChart.update();
            } else if (dashboardData && dashboardData.sample_points) {
                initPCAChart();
            }
        }, 150);
    }

    // Handle Radar chart resizing
    if (tabId === "tab-overview" && radarChart) {
        setTimeout(() => {
            radarChart.resize();
            radarChart.update();
        }, 150);
    }
}

// Fetch Core Data
async function fetchDashboardData() {
    try {
        console.log("Fetching dashboard data from /api/data...");
        const res = await fetch("/api/data");
        if (!res.ok) {
            throw new Error(`HTTP error ${res.status}: ${res.statusText}`);
        }
        dashboardData = await res.json();

        if (dashboardData.error) {
            console.error("Dashboard backend error:", dashboardData.error);
            return;
        }

        console.log("Dashboard data successfully loaded:", Object.keys(dashboardData));

        populateSummaryBadges(dashboardData.summary);
        populatePreprocessingTable(dashboardData.feature_preprocessing);
        populateHierarchicalTable(dashboardData.hierarchical_results);
        populateDBSCANTable(dashboardData.dbscan_results);
        populatePCALoadings(dashboardData.pca_loadings);
        
        initRadarChart();
        initPCAChart();
        initStormMap();
        loadResponsibleAIReport();

    } catch (err) {
        console.error("Failed to load or parse dashboard data:", err);
    }
}

// Populate Badges & KPIs
function populateSummaryBadges(summary) {
    if (!summary) return;

    setText("badge-total-records", summary.total_records ? summary.total_records.toLocaleString() : "-");
    setText("badge-sample-size", summary.sample_size ? summary.sample_size.toLocaleString() : "-");
    setText("badge-best-h", summary.best_hierarchical || "-");
    setText("badge-best-d", summary.best_dbscan || "-");

    setText("kpi-h-name", summary.best_hierarchical || "-");
    setText("kpi-h-sil", summary.hierarchical_silhouette != null ? Number(summary.hierarchical_silhouette).toFixed(4) : "-");
    setText("kpi-h-ch", summary.hierarchical_ch != null ? Number(summary.hierarchical_ch).toFixed(1) : "-");

    setText("kpi-d-name", summary.best_dbscan || "-");
    setText("kpi-d-sil", summary.dbscan_silhouette != null ? Number(summary.dbscan_silhouette).toFixed(4) : "-");
    setText("kpi-d-noise", summary.dbscan_noise_pct != null ? `${Number(summary.dbscan_noise_pct).toFixed(1)}%` : "-");

    setText("kpi-pca-var", summary.pca_variance_2d != null ? `${(Number(summary.pca_variance_2d) * 100).toFixed(1)}%` : "-");
    setText("kpi-pca-3var", summary.pca_variance_3d != null ? `${(Number(summary.pca_variance_3d) * 100).toFixed(1)}%` : "-");
    setText("kpi-cross-ari", summary.cross_method_ari != null ? Number(summary.cross_method_ari).toFixed(3) : "-");
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.innerText = text;
}

// Preprocessing Table
function populatePreprocessingTable(features) {
    const tbody = document.getElementById("table-preprocessing-body");
    if (!tbody || !features) return;
    tbody.innerHTML = "";

    features.forEach(row => {
        const tr = document.createElement("tr");
        tr.className = "hover:bg-slate-800/40 transition";
        const missingPct = row.missing_percentage != null ? Number(row.missing_percentage).toFixed(1) : "0.0";
        tr.innerHTML = `
            <td class="px-4 py-3 font-semibold text-white font-mono">${row.feature || ""}</td>
            <td class="px-4 py-3">${row.missing_in_sample != null ? row.missing_in_sample : 0}</td>
            <td class="px-4 py-3 font-mono">${missingPct}%</td>
            <td class="px-4 py-3 text-sky-300 font-mono text-[11px]">${row.transformation || ""}</td>
            <td class="px-4 py-3 text-slate-300">${row.imputation || ""}</td>
            <td class="px-4 py-3 text-slate-400 font-mono text-[11px]">${row.standardization || ""}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Hierarchical Table
function populateHierarchicalTable(rows) {
    const tbody = document.getElementById("table-hierarchical-body");
    if (!tbody || !rows) return;
    tbody.innerHTML = "";

    const bestH = dashboardData && dashboardData.summary ? dashboardData.summary.best_hierarchical : "";

    rows.forEach(r => {
        const isSingleton = r.smallest_cluster != null && r.smallest_cluster <= 3;
        const isSelected = r.configuration === bestH;

        const cpccStr = r.cophenetic_correlation != null ? Number(r.cophenetic_correlation).toFixed(4) : "N/A";
        const silStr = r.silhouette_evaluation_sample != null ? Number(r.silhouette_evaluation_sample).toFixed(4) : "N/A";
        const dbStr = r.davies_bouldin_index != null ? Number(r.davies_bouldin_index).toFixed(3) : "N/A";
        const chStr = r.calinski_harabasz_index != null ? Number(r.calinski_harabasz_index).toFixed(1) : "N/A";
        const ratioStr = r.cluster_size_ratio != null ? Number(r.cluster_size_ratio).toFixed(1) : "N/A";

        const tr = document.createElement("tr");
        tr.className = `hover:bg-slate-800/40 transition ${isSelected ? "bg-indigo-950/40 font-semibold" : ""}`;
        tr.innerHTML = `
            <td class="px-3 py-2.5 font-mono text-white flex items-center space-x-1.5">
                ${isSelected ? '<span class="w-2 h-2 rounded-full bg-emerald-400"></span>' : ''}
                <span>${r.configuration || ""}</span>
            </td>
            <td class="px-3 py-2.5 capitalize">${r.method || ""}</td>
            <td class="px-3 py-2.5 font-mono">${r.requested_clusters != null ? r.requested_clusters : ""}</td>
            <td class="px-3 py-2.5 font-mono text-sky-400">${cpccStr}</td>
            <td class="px-3 py-2.5 font-mono ${r.silhouette_evaluation_sample != null && r.silhouette_evaluation_sample > 0.35 ? 'text-emerald-400' : 'text-slate-300'}">
                ${silStr}
            </td>
            <td class="px-3 py-2.5 font-mono text-slate-300">${dbStr}</td>
            <td class="px-3 py-2.5 font-mono text-slate-300">${chStr}</td>
            <td class="px-3 py-2.5 font-mono ${isSingleton ? 'text-rose-400 font-bold' : 'text-slate-300'}">
                ${r.smallest_cluster != null ? r.smallest_cluster : "-"} ${isSingleton ? '⚠️' : ''}
            </td>
            <td class="px-3 py-2.5 font-mono text-slate-300">${r.largest_cluster != null ? r.largest_cluster : "-"}</td>
            <td class="px-3 py-2.5 font-mono text-slate-400">${ratioStr}</td>
        `;
        tbody.appendChild(tr);
    });
}

// DBSCAN Table
function populateDBSCANTable(rows) {
    const tbody = document.getElementById("table-dbscan-body");
    if (!tbody || !rows) return;
    tbody.innerHTML = "";

    const bestD = dashboardData && dashboardData.summary ? dashboardData.summary.best_dbscan : "";

    rows.forEach(r => {
        const isSelected = r.configuration === bestD;
        const noisePctStr = r.noise_percentage != null ? Number(r.noise_percentage).toFixed(1) + "%" : "0.0%";
        const silStr = r.silhouette_non_noise_evaluation_sample != null ? Number(r.silhouette_non_noise_evaluation_sample).toFixed(4) : "N/A";
        const dbStr = r.davies_bouldin_index != null ? Number(r.davies_bouldin_index).toFixed(3) : "N/A";
        const chStr = r.calinski_harabasz_index != null ? Number(r.calinski_harabasz_index).toFixed(1) : "N/A";

        const tr = document.createElement("tr");
        tr.className = `hover:bg-slate-800/40 transition ${isSelected ? "bg-purple-950/40 font-semibold" : ""}`;
        tr.innerHTML = `
            <td class="px-3 py-2.5 font-mono text-white flex items-center space-x-1.5">
                ${isSelected ? '<span class="w-2 h-2 rounded-full bg-emerald-400"></span>' : ''}
                <span>${r.configuration || ""}</span>
            </td>
            <td class="px-3 py-2.5">${r.eps != null ? r.eps : ""}</td>
            <td class="px-3 py-2.5">${r.min_samples != null ? r.min_samples : ""}</td>
            <td class="px-3 py-2.5 text-white font-bold">${r.clusters_excluding_noise != null ? r.clusters_excluding_noise : 0}</td>
            <td class="px-3 py-2.5 text-amber-400">${r.noise_count != null ? r.noise_count : 0}</td>
            <td class="px-3 py-2.5 text-amber-400">${noisePctStr}</td>
            <td class="px-3 py-2.5 text-slate-300">${r.core_points != null ? r.core_points : 0}</td>
            <td class="px-3 py-2.5 text-slate-300">${r.border_points != null ? r.border_points : 0}</td>
            <td class="px-3 py-2.5 ${r.silhouette_non_noise_evaluation_sample != null && r.silhouette_non_noise_evaluation_sample > 0.3 ? 'text-emerald-400' : 'text-slate-300'}">
                ${silStr}
            </td>
            <td class="px-3 py-2.5">${dbStr}</td>
            <td class="px-3 py-2.5">${chStr}</td>
        `;
        tbody.appendChild(tr);
    });
}

// PCA Loadings
function populatePCALoadings(loadings) {
    const tbody = document.getElementById("table-pca-loadings-body");
    if (!tbody || !loadings) return;
    tbody.innerHTML = "";

    loadings.forEach(row => {
        const pc1 = row.PC1_loading != null ? Number(row.PC1_loading) : 0;
        const pc2 = row.PC2_loading != null ? Number(row.PC2_loading) : 0;
        const pc3 = row.PC3_loading != null ? Number(row.PC3_loading) : 0;

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td class="px-3 py-2 font-mono text-white font-semibold">${row.feature || ""}</td>
            <td class="px-3 py-2 font-mono ${pc1 > 0.3 ? 'text-sky-400' : pc1 < -0.3 ? 'text-rose-400' : 'text-slate-300'}">
                ${pc1.toFixed(3)}
            </td>
            <td class="px-3 py-2 font-mono ${pc2 > 0.3 ? 'text-sky-400' : pc2 < -0.3 ? 'text-rose-400' : 'text-slate-300'}">
                ${pc2.toFixed(3)}
            </td>
            <td class="px-3 py-2 font-mono ${pc3 > 0.3 ? 'text-sky-400' : pc3 < -0.3 ? 'text-rose-400' : 'text-slate-300'}">
                ${pc3.toFixed(3)}
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Dendrogram Switcher
function setDendrogram(method) {
    document.querySelectorAll("[id^='btn-dend-']").forEach(btn => {
        btn.className = "px-3 py-1.5 rounded font-medium text-slate-300 hover:text-white";
    });
    const activeBtn = document.getElementById(`btn-dend-${method}`);
    if (activeBtn) {
        activeBtn.className = "px-3 py-1.5 rounded font-medium bg-indigo-600 text-white";
    }
    const img = document.getElementById("dendrogram-img");
    if (img) {
        img.src = `/plots/dendrogram_${method}.png`;
    }
}

// Radar Profile Chart
function initRadarChart() {
    if (typeof Chart === "undefined") {
        console.warn("Chart.js is not loaded yet.");
        return;
    }
    const ctx = document.getElementById("chart-radar-profiles");
    if (!ctx) return;

    if (radarChart) {
        radarChart.destroy();
    }

    radarChart = new Chart(ctx, {
        type: "radar",
        data: {
            labels: ["Casualties", "Property Damage", "Crop Damage", "Duration", "Seasonality"],
            datasets: [
                {
                    label: "C1: Synoptic Winter",
                    data: [-0.3, -0.4, -0.2, 1.8, -1.2],
                    borderColor: "#0284c7",
                    backgroundColor: "rgba(2, 132, 199, 0.2)",
                    borderWidth: 2
                },
                {
                    label: "C2: Catastrophic Outbreaks",
                    data: [3.4, 1.9, 0.8, 0.4, 0.3],
                    borderColor: "#e11d48",
                    backgroundColor: "rgba(225, 29, 72, 0.2)",
                    borderWidth: 2
                },
                {
                    label: "C3: Convective Storms",
                    data: [-0.1, 0.3, 0.0, -0.8, 0.9],
                    borderColor: "#10b981",
                    backgroundColor: "rgba(16, 185, 129, 0.2)",
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: "#334155" },
                    grid: { color: "#1e293b" },
                    pointLabels: { color: "#94a3b8", font: { size: 10, family: "Inter" } },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: { labels: { color: "#cbd5e1", font: { size: 11, family: "Inter" } } }
            }
        }
    });
}

// Interactive PCA Scatter Chart
function initPCAChart() {
    if (typeof Chart === "undefined") {
        console.warn("Chart.js is not loaded yet.");
        return;
    }
    const ctx = document.getElementById("chart-pca-scatter");
    if (!ctx || !dashboardData || !dashboardData.sample_points) return;

    if (pcaChart) {
        pcaChart.destroy();
    }

    const datasets = buildPCADatasets(currentPCAColor);

    pcaChart = new Chart(ctx, {
        type: "scatter",
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    const el = elements[0];
                    const datasetIdx = el.datasetIndex;
                    const index = el.index;
                    const point = pcaChart.data.datasets[datasetIdx].raw_items[index];
                    displayPointDetails(point);
                }
            },
            scales: {
                x: {
                    title: { display: true, text: "Principal Component 1 (Seasonal & Duration Axis)", color: "#94a3b8" },
                    grid: { color: "#1e293b" },
                    ticks: { color: "#64748b" }
                },
                y: {
                    title: { display: true, text: "Principal Component 2 (Casualty & Severity Axis)", color: "#94a3b8" },
                    grid: { color: "#1e293b" },
                    ticks: { color: "#64748b" }
                }
            },
            plugins: {
                legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const p = context.dataset.raw_items[context.dataIndex];
                            if (!p) return "";
                            const dmg = (p.PROPERTY_DAMAGE_USD || 0).toLocaleString();
                            const dur = (p.DURATION_HOURS != null ? Number(p.DURATION_HOURS).toFixed(1) : 0);
                            return `${p.EVENT_TYPE || 'Event'} (${p.STATE || ''}) | Dur: ${dur}h | Dmg: $${dmg}`;
                        }
                    }
                }
            }
        }
    });
}

function buildPCADatasets(mode) {
    if (!dashboardData || !dashboardData.sample_points) return [];
    const points = dashboardData.sample_points;

    const groups = {};
    points.forEach(p => {
        if (p.PC1 == null || p.PC2 == null) return;
        const label = String(mode === "hierarchical" ? p.HIERARCHICAL_CLUSTER : p.DBSCAN_CLUSTER);
        if (!groups[label]) groups[label] = [];
        groups[label].push(p);
    });

    const datasets = [];
    Object.keys(groups).sort().forEach(clusterKey => {
        const clusterPoints = groups[clusterKey];
        const color = mode === "hierarchical"
            ? (CLUSTER_COLORS_H[clusterKey] || "#6366f1")
            : (CLUSTER_COLORS_D[clusterKey] || "#64748b");

        const clusterName = mode === "hierarchical"
            ? `Hierarchical Cluster ${clusterKey}`
            : (clusterKey === "-1" ? "DBSCAN Noise (-1)" : `DBSCAN Cluster ${clusterKey}`);

        datasets.push({
            label: clusterName,
            data: clusterPoints.map(p => ({ x: Number(p.PC1), y: Number(p.PC2) })),
            backgroundColor: color,
            borderColor: color,
            pointRadius: clusterKey === "-1" ? 2.5 : 3.5,
            pointHoverRadius: 6,
            raw_items: clusterPoints
        });
    });

    return datasets;
}

function setPCAColorMode(mode) {
    currentPCAColor = mode;
    const btnH = document.getElementById("btn-pca-color-h");
    const btnD = document.getElementById("btn-pca-color-d");

    if (btnH) {
        btnH.className = mode === "hierarchical"
            ? "px-3 py-1 rounded font-medium bg-indigo-600 text-white"
            : "px-3 py-1 rounded font-medium text-slate-300 hover:text-white";
    }
    if (btnD) {
        btnD.className = mode === "dbscan"
            ? "px-3 py-1 rounded font-medium bg-purple-600 text-white"
            : "px-3 py-1 rounded font-medium text-slate-300 hover:text-white";
    }

    if (pcaChart) {
        pcaChart.data.datasets = buildPCADatasets(mode);
        pcaChart.update();
    }
}

function displayPointDetails(p) {
    if (!p) return;
    const box = document.getElementById("pca-inspector-content");
    if (!box) return;

    const propDmg = (p.PROPERTY_DAMAGE_USD || 0).toLocaleString();
    const cropDmg = (p.CROP_DAMAGE_USD || 0).toLocaleString();
    const dur = (p.DURATION_HOURS != null ? Number(p.DURATION_HOURS).toFixed(2) : "0.0");

    box.innerHTML = `
        <div class="space-y-1.5">
            <div class="text-white font-bold text-sm">${p.EVENT_TYPE || 'Extreme Event'}</div>
            <div class="text-slate-400 font-mono text-[11px]">${p.CZ_NAME || 'Zone'}, ${p.STATE || ''} (${p.YEAR || ''})</div>
            <div class="mt-2 pt-2 border-t border-slate-800 space-y-1">
                <div class="flex justify-between"><span>Duration:</span><strong class="text-white">${dur} hrs</strong></div>
                <div class="flex justify-between"><span>Property Damage:</span><strong class="text-emerald-400">$${propDmg}</strong></div>
                <div class="flex justify-between"><span>Crop Damage:</span><strong class="text-emerald-400">$${cropDmg}</strong></div>
                <div class="flex justify-between"><span>Injuries / Deaths:</span><strong class="text-rose-400">${p.TOTAL_INJURIES || 0} / ${p.TOTAL_DEATHS || 0}</strong></div>
                <div class="flex justify-between"><span>Hierarchical:</span><strong class="text-sky-400">Cluster ${p.HIERARCHICAL_CLUSTER}</strong></div>
                <div class="flex justify-between"><span>DBSCAN:</span><strong class="text-purple-400">${p.DBSCAN_CLUSTER === -1 ? 'Noise (-1)' : 'Cluster ' + p.DBSCAN_CLUSTER}</strong></div>
                <div class="flex justify-between font-mono text-[10px] text-slate-400"><span>Coordinates:</span><span>PC1=${p.PC1}, PC2=${p.PC2}</span></div>
            </div>
        </div>
    `;
}

// Leaflet Map Initialization
function initStormMap() {
    if (typeof L === "undefined") {
        console.warn("Leaflet.js is not loaded yet.");
        return;
    }
    if (!dashboardData || !dashboardData.sample_points) return;
    const mapContainer = document.getElementById("storm-map");
    if (!mapContainer) return;

    if (stormMap) {
        stormMap.remove();
        stormMap = null;
    }

    try {
        stormMap = L.map("storm-map", {
            center: [38.5, -96.5],
            zoom: 4,
            minZoom: 3,
            maxZoom: 16
        });

        // 1. Esri World Dark Gray Canvas: 100% Free, NO API key required, zero watermark
        const esriDark = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", {
            attribution: 'Tiles &copy; Esri, DeLorme, NAVTEQ &copy; NOAA',
            maxZoom: 16
        });

        // 2. OpenStreetMap Standard: 100% Free & Open Source, NO API key required
        const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; NOAA',
            maxZoom: 19
        });

        // Add Dark theme by default
        esriDark.addTo(stormMap);

        // Add Layer switcher control in top-right
        L.control.layers({
            "Dark Canvas (Esri - No Key)": esriDark,
            "OpenStreetMap (No Key)": osm
        }, null, { position: "topright" }).addTo(stormMap);

        const validPoints = dashboardData.sample_points.filter(p => 
            p.BEGIN_LAT != null && p.BEGIN_LON != null && 
            !isNaN(p.BEGIN_LAT) && !isNaN(p.BEGIN_LON)
        );

        validPoints.slice(0, 600).forEach(p => {
            const color = CLUSTER_COLORS_H[p.HIERARCHICAL_CLUSTER] || "#3b82f6";
            const marker = L.circleMarker([p.BEGIN_LAT, p.BEGIN_LON], {
                radius: 5,
                fillColor: color,
                color: "#0f172a",
                weight: 1,
                opacity: 0.9,
                fillOpacity: 0.85
            }).addTo(stormMap);

            const propDmg = (p.PROPERTY_DAMAGE_USD || 0).toLocaleString();
            const casualties = (p.TOTAL_INJURIES || 0) + (p.TOTAL_DEATHS || 0);
            const dur = (p.DURATION_HOURS != null ? Number(p.DURATION_HOURS).toFixed(1) : 0);

            marker.bindPopup(`
                <div class="text-xs space-y-1">
                    <strong class="text-white text-sm block">${p.EVENT_TYPE || 'Event'}</strong>
                    <div class="text-slate-300">${p.CZ_NAME || ''}, ${p.STATE || ''} (${p.YEAR || ''})</div>
                    <div>Duration: <strong>${dur} hrs</strong></div>
                    <div>Property Damage: <strong class="text-emerald-400">$${propDmg}</strong></div>
                    <div>Casualties: <strong class="text-rose-400">${casualties}</strong></div>
                    <div class="text-[10px] text-sky-400 mt-1 font-semibold">Hierarchical Cluster ${p.HIERARCHICAL_CLUSTER}</div>
                </div>
            `);
        });

        console.log(`Rendered ${validPoints.slice(0, 600).length} storm markers on the map without any API key.`);

    } catch (e) {
        console.warn("Leaflet map initialization warning:", e);
    }
}

// Live Classifier Simulator Form
function setupClassifierForm() {
    const form = document.getElementById("form-classifier");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const payload = {
            duration_hours: parseFloat(document.getElementById("input-duration").value),
            property_damage: parseFloat(document.getElementById("input-prop-damage").value),
            crop_damage: parseFloat(document.getElementById("input-crop-damage").value),
            injuries: parseFloat(document.getElementById("input-injuries").value),
            deaths: parseFloat(document.getElementById("input-deaths").value),
            month: parseInt(document.getElementById("input-month").value)
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const result = await res.json();

            if (result.error) {
                alert("Classification error: " + result.error);
                return;
            }

            // Update output UI
            setText("res-h-archetype", `Cluster ${result.hierarchical.cluster}: ${result.hierarchical.archetype_name}`);
            setText("res-h-desc", result.hierarchical.archetype_desc);
            setText("res-h-dist", `Distance to centroid: ${result.hierarchical.distance_to_centroid}`);

            setText("res-d-cluster", result.dbscan.assigned_cluster);
            setText("res-d-note", result.dbscan.is_outlier
                ? "Flagged as extreme outlier / noise: falls beyond dense core regions."
                : `Proximity to core cluster centroid: ${result.dbscan.distance_to_core_centroid}`);

            setText("res-pca-pc1", result.pca_projection.PC1);
            setText("res-pca-pc2", result.pca_projection.PC2);

            // Add user point to PCA scatter plot
            addUserPointToPCA(result.pca_projection.PC1, result.pca_projection.PC2, payload);

        } catch (err) {
            console.error("Prediction request failed:", err);
        }
    });
}

function addUserPointToPCA(pc1, pc2, payload) {
    if (!pcaChart) return;

    // Remove previous user point dataset if exists
    pcaChart.data.datasets = pcaChart.data.datasets.filter(d => d.label !== "★ Your Custom Scenario");

    pcaChart.data.datasets.push({
        label: "★ Your Custom Scenario",
        data: [{ x: pc1, y: pc2 }],
        backgroundColor: "#f59e0b",
        borderColor: "#ffffff",
        borderWidth: 2,
        pointRadius: 9,
        pointStyle: "star",
        raw_items: [{
            EVENT_TYPE: "Custom Scenario",
            STATE: "Simulated",
            YEAR: 2026,
            DURATION_HOURS: payload.duration_hours,
            PROPERTY_DAMAGE_USD: payload.property_damage,
            CROP_DAMAGE_USD: payload.crop_damage,
            TOTAL_INJURIES: payload.injuries,
            TOTAL_DEATHS: payload.deaths,
            HIERARCHICAL_CLUSTER: "Simulated",
            DBSCAN_CLUSTER: "Simulated",
            PC1: pc1,
            PC2: pc2
        }]
    });

    pcaChart.update();
}

// Load Responsible AI Text
async function loadResponsibleAIReport() {
    try {
        const res = await fetch("/api/responsible_ai");
        const data = await res.json();
        const el = document.getElementById("responsible-ai-text");
        if (el) el.innerText = data.content;
    } catch (err) {
        console.error("Failed to load responsible AI text:", err);
    }
}
