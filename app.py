"""Flask web application for NOAA Extreme Weather Event Grouping Studio.

Provides an interactive, zero-build frontend dashboard for exploring:
- Preprocessing and feature engineering documentation
- Hierarchical clustering (linkages, dendrograms, cophenetic correlations)
- DBSCAN density clustering (parameter sensitivity, k-distance elbow, noise analysis)
- PCA 2D/3D projection, scree plot, and feature loadings biplot
- Cluster meteorological profiles in original units ($ USD, casualties, hours)
- Interactive US geographic map of clustered extreme events
- Live storm event classification simulator
- Model stability (ARI matrix) and Responsible AI report
"""

import json
from pathlib import Path
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "outputs" / "dashboard_data.json"
PLOTS_DIR = ROOT / "outputs" / "plots"
METRICS_DIR = ROOT / "outputs" / "metrics"

app = Flask(__name__, static_folder="static", template_folder="templates")


def clean_for_json(obj):
    """Recursively convert float NaNs and Infs to None for valid RFC 8259 JSON."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    return obj


def load_dashboard_data():
    """Load cached dashboard payload with NaN sanitization."""
    if not DATA_FILE.exists():
        return None
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return clean_for_json(data)


@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    """Return the complete dashboard dataset with standard RFC 8259 JSON compliance."""
    data = load_dashboard_data()
    if data is None:
        return jsonify({"error": "Dashboard data not found. Please run main.py first."}), 404
    return jsonify(clean_for_json(data))


@app.route("/api/responsible_ai")
def get_responsible_ai():
    """Return the Responsible AI and interpretation report."""
    rep_path = METRICS_DIR / "responsible_interpretation.txt"
    if rep_path.exists():
        content = rep_path.read_text(encoding="utf-8")
    else:
        content = "Responsible AI report not generated yet."
    return jsonify({"content": content})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Live interactive storm event classification and PCA projection."""
    data = load_dashboard_data()
    if data is None or "model_params" not in data:
        return jsonify({"error": "Model parameters unavailable. Run main.py."}), 500

    payload = request.get_json(force=True)
    try:
        duration_hours = float(payload.get("duration_hours", 1.0))
        prop_damage = float(payload.get("property_damage", 0.0))
        crop_damage = float(payload.get("crop_damage", 0.0))
        injuries = float(payload.get("injuries", 0.0))
        deaths = float(payload.get("deaths", 0.0))
        month = int(payload.get("month", 6))
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400

    # 1. Feature transformations
    log_injuries = float(np.log1p(max(0.0, injuries)))
    log_deaths = float(np.log1p(max(0.0, deaths)))
    log_prop_damage = float(np.log1p(max(0.0, prop_damage)))
    log_crop_damage = float(np.log1p(max(0.0, crop_damage)))
    log_duration = float(np.log1p(max(0.0, duration_hours)))
    month_sin = float(np.sin(2.0 * np.pi * month / 12.0))
    month_cos = float(np.cos(2.0 * np.pi * month / 12.0))

    feature_vec = np.array([
        log_injuries, log_deaths, log_prop_damage, log_crop_damage,
        log_duration, month_sin, month_cos
    ])

    params = data["model_params"]
    mean = np.array(params["scaler_mean"])
    scale = np.array(params["scaler_scale"])

    # Standardize
    scaled = (feature_vec - mean) / scale

    # 2. PCA Projection (PC1, PC2, PC3)
    pca_components = np.array(params["pca_components"])
    pca_mean = np.array(params["pca_mean"])
    # Scaled features were already centered by StandardScaler, but apply PCA centering
    pc_coords = np.dot(pca_components, (scaled - pca_mean))

    # 3. Nearest Cluster Assignment (Hierarchical)
    h_centroids = params["hierarchical_centroids"]
    h_dists = {}
    for cid, centroid_dict in h_centroids.items():
        c_vec = np.array([centroid_dict[feat] for feat in params["features"]])
        dist = float(np.linalg.norm(scaled - c_vec))
        h_dists[cid] = dist
    assigned_h = min(h_dists, key=h_dists.get)

    # 4. Nearest Cluster Assignment (DBSCAN)
    d_centroids = params["dbscan_centroids"]
    d_dists = {}
    for cid, centroid_dict in d_centroids.items():
        c_vec = np.array([centroid_dict[feat] for feat in params["features"]])
        dist = float(np.linalg.norm(scaled - c_vec))
        d_dists[cid] = dist
    min_d_dist = min(d_dists.values()) if d_dists else 999.0
    assigned_d = min(d_dists, key=d_dists.get) if d_dists else "Unknown"

    # If distance to nearest core cluster centroid is large (> 2.5 std), flag as likely noise/anomaly
    is_dbscan_noise = bool(min_d_dist > 2.5)
    dbscan_label = "Noise / Extreme Outlier (-1)" if is_dbscan_noise else f"Cluster {assigned_d}"

    # Meteorological archetype interpretation
    archetypes = {
        "1": {
            "name": "Prolonged Winter & Synoptic Systems",
            "desc": "Long duration (24-72+ hrs), winter seasonality (Nov-March), low-to-moderate direct casualties, typical of heavy snow, blizzards, and synoptic cold fronts."
        },
        "2": {
            "name": "Catastrophic High-Impact Hazard Outbreaks",
            "desc": "Significant human casualties and severe property/crop destruction. Characterized by violent tornadoes, major flash floods, and severe heat/inundation crises."
        },
        "3": {
            "name": "Frequent Localized Convective Storms",
            "desc": "Sudden onset, short duration (<2 hrs), spring/summer seasonality. Typified by thunderstorm winds, severe hail, and isolated flash flooding with localized damage."
        }
    }
    archetype_info = archetypes.get(assigned_h, {
        "name": f"Extreme Weather Archetype {assigned_h}",
        "desc": "Multi-hazard meteorological event pattern."
    })

    return jsonify({
        "input_features": {
            "duration_hours": duration_hours,
            "property_damage": prop_damage,
            "crop_damage": crop_damage,
            "injuries": injuries,
            "deaths": deaths,
            "month": month,
        },
        "transformed_features": {
            "LOG_TOTAL_INJURIES": round(log_injuries, 4),
            "LOG_TOTAL_DEATHS": round(log_deaths, 4),
            "LOG_PROPERTY_DAMAGE_USD": round(log_prop_damage, 4),
            "LOG_CROP_DAMAGE_USD": round(log_crop_damage, 4),
            "LOG_DURATION_HOURS": round(log_duration, 4),
            "MONTH_SIN": round(month_sin, 4),
            "MONTH_COS": round(month_cos, 4),
        },
        "pca_projection": {
            "PC1": round(float(pc_coords[0]), 3),
            "PC2": round(float(pc_coords[1]), 3),
            "PC3": round(float(pc_coords[2]), 3) if len(pc_coords) > 2 else 0.0,
        },
        "hierarchical": {
            "cluster": int(assigned_h),
            "distance_to_centroid": round(h_dists[assigned_h], 3),
            "archetype_name": archetype_info["name"],
            "archetype_desc": archetype_info["desc"]
        },
        "dbscan": {
            "assigned_cluster": dbscan_label,
            "distance_to_core_centroid": round(min_d_dist, 3),
            "is_outlier": is_dbscan_noise
        }
    })


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    """Serve static plot images."""
    return send_from_directory(PLOTS_DIR, filename)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("NOAA Extreme Weather Event Grouping Studio")
    print("Serving interactive dashboard at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
