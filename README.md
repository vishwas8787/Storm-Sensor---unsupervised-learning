# Extreme Weather Event Grouping Studio

### Unit III & IV: Hierarchical & Density-Based Clustering, PCA Visualisation
**Reproducible Unsupervised Learning on the NOAA Storm Events Database (2010–2020)**

---

## 1. Project Overview

This project implements an end-to-end, reproducible unsupervised learning workflow for grouping extreme weather events recorded in NOAA's Storm Events Database. 

The objective is to discover natural taxonomies and severity profiles of extreme weather events across the United States using:
- **Hierarchical Clustering (Agglomerative)**: Ward, Complete, Average, and Single linkages across $k \in [3, 4, 5, 6]$.
- **Density-Based Clustering (DBSCAN)**: Parameter grid search across $\epsilon \in [0.8 .. 2.0]$ and $\text{min\_samples} \in [5, 8, 12, 15]$.
- **$k$-Nearest Neighbor Distance Diagnostic**: Knee/elbow identification for optimal $\epsilon$ selection.
- **PCA Dimensionality Reduction**: 2D/3D projection, scree plot, feature loadings matrix, and biplot vector analysis.
- **Multi-Metric Cluster Validation**: Silhouette Coefficient, Davies-Bouldin Index (DBI), Calinski-Harabasz Index (CHI), and Cophenetic Correlation Coefficient (CPCC).
- **Cluster Stability & Sensitivity**: Pairwise Adjusted Rand Index (ARI) heatmap.
- **Meteorological Archetype Profiling**: Mean and median evaluation in original physical units (USD damage, casualties, hours).
- **Interactive Web Frontend**: Full-stack Flask dashboard featuring interactive PCA scatter, US storm map, radar profiles, and a live storm event classification simulator.

---

## 2. Quick Start Guide

### Requirements
- Python 3.10+ (tested on Python 3.14)
- Standard scientific stack: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `Flask`

### Setup and Execution

1. **Install Dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

2. **Run End-to-End Analysis Pipeline:**
   Executes data preprocessing verification, feature engineering, hierarchical clustering, DBSCAN grid search, PCA fitting, cluster profiling, and metric evaluations:
   ```powershell
   python -u main.py
   ```
   *Runtime: ~15 seconds on a standard multi-core machine.*

3. **Launch the Interactive Frontend Dashboard:**
   Starts the local Flask web studio:
   ```powershell
   python app.py
   ```
   Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## 3. Dataset Preprocessing & Feature Engineering Choices

The analysis uses the NOAA **Storm Events Details** dataset (`data/raw/StormEvents_details_2010_2020.csv`), where each row corresponds to a single, verified storm event.

### Preprocessing Choices
1. **Deduplication:** Events are validated by unique `EVENT_ID`. Duplicate entries are removed.
2. **Timestamp Parsing:** `BEGIN_DATE_TIME` and `END_DATE_TIME` are parsed using verified format `%d-%b-%y %H:%M:%S`. Event duration in hours is derived.
3. **Monetary Damage Conversion:** Damage amounts with suffixes `K` ($10^3$), `M` ($10^6$), and `B` ($10^9$) are converted into exact numeric USD floats (`PROPERTY_DAMAGE_USD`, `CROP_DAMAGE_USD`). Invalid negative amounts are masked.
4. **Missing Value Rationale:**
   - In National Weather Service (NWS) reporting directives, absence of a damage figure indicates no quantifiable or verified monetary loss occurred for that event. Missing property and crop damage are imputed as **$0.00**.
   - Missing or non-positive durations are imputed with the sample median duration.
   - Crucially, artificial binary missingness indicators were omitted to prevent reporting artifacts from dominating cluster distances.
5. **Skewness Mitigation:** Extreme weather impacts follow heavy-tailed power-law distributions. A $\log(1 + x)$ (`log1p`) transformation is applied to:
   - `LOG_TOTAL_INJURIES`
   - `LOG_TOTAL_DEATHS`
   - `LOG_PROPERTY_DAMAGE_USD`
   - `LOG_CROP_DAMAGE_USD`
   - `LOG_DURATION_HOURS`
6. **Cyclical Seasonality:** Event start month is encoded into continuous cyclical dimensions:
   $$\text{MONTH\_SIN} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{MONTH\_COS} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
   This ensures December ($m=12$) and January ($m=1$) remain contiguous in Euclidean metric space.
7. **Feature Standardization:** All 7 model features are standardized using `StandardScaler` ($\mu = 0, \sigma = 1$) to ensure isotropic distance scaling.

---

## 4. Methodology & Theoretical Insights (Unit III & IV)

### Unit III: Agglomerative Hierarchical Clustering
Four linkage criteria were evaluated across cluster counts $k \in [3, 4, 5, 6]$:
- **Ward's Linkage (Minimum Variance):** Minimizes total within-cluster variance ($\Delta ESS$). Produces spherical, balanced clusters and avoids chaining. Selected as the primary model (**Ward $k=3$**, Silhouette: $0.3525$, Calinski-Harabasz: $489.88$).
- **Average Linkage (UPGMA):** Achieves the highest Cophenetic Correlation Coefficient ($\text{CPCC} = 0.9350$), but suffers from outlier isolation, splitting single extreme records into trivial singleton clusters ($2496 : 1$).
- **Complete Linkage (Maximum Distance):** Maximizes pairwise distance between clusters. Highly sensitive to distant noise points.
- **Single Linkage (Minimum Distance):** Demonstrates the textbook **chaining effect**, where individual points link sequentially into one large mega-cluster ($2482 : 1$).

#### Cophenetic Correlation Comparison
| Linkage Method | Cophenetic Correlation Coefficient (CPCC) | Practical Topology |
|---|---|---|
| **Average** | **0.9350** | High metric fidelity; prone to singleton outlier branches |
| **Complete** | **0.8583** | Compact boundaries; sensitive to distant points |
| **Single** | **0.8106** | Severe chaining effect |
| **Ward** | **0.4811** | Variance-minimizing; best cluster balance and interpretability |

---

### Unit IV: Density-Based Clustering (DBSCAN)
DBSCAN groups contiguous regions of high point density while separating sparse regions as noise (label $-1$).
- **$k$-Nearest Neighbor Distance Diagnostic:** A sorted distance curve ($k=8$) reveals a prominent knee around $\epsilon \approx 0.8$, identifying the critical distance threshold separating core storm clusters from sparse noise.
- **Parameter Sensitivity Grid:** Tested across $\epsilon \in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]$ and $\text{min\_samples} \in [5, 8, 12, 15]$.
- **Selected Configuration:** `dbscan_eps0.8_min5` produces 2 distinct dense clusters with only **$3.84\%$ noise** ($96$ events), achieving a non-noise Silhouette score of **$0.3186$** and Calinski-Harabasz score of **$744.46$**.
- **Noise Analysis:** Events assigned to noise ($-1$) are not data errors; they represent rare, compound multi-hazard events (e.g., catastrophic tornado outbreaks with tens of millions in damage and multiple fatalities).

---

### PCA Dimensionality Reduction
PCA was fitted strictly for visualization and interpretation (clustering was performed in the standardized 7D feature space):
- **Explained Variance:**
  - $\text{PC1}$ explains $24.7\%$ of variance (Duration & Cold-Season Synoptic Axis).
  - $\text{PC2}$ explains $21.0\%$ of variance (Casualty Severity & Life Threat Axis).
  - $\text{PC3}$ explains $14.3\%$ of variance (Economic & Agricultural Damage Axis).
  - Cumulative 2-component variance: **$39.5\%$**; 3-component: **$55.1\%$**.

---

## 5. Discovered Meteorological Archetypes

| Cluster | Archetype Name | Sample Size | Primary NOAA Event Types | Median Duration | Mean Damage ($) | Mean Casualties |
|---|---|---|---|---|---|---|
| **Cluster 1** | **Prolonged Synoptic Winter Systems** | 575 ($23.0\%$) | Winter Storm, Heavy Snow, Winter Weather, Flood | $19.0$ hrs | $\approx \$120\text{K}$ | $0.0$ |
| **Cluster 2** | **Catastrophic Hazard Outbreaks** | 18 ($0.7\%$) | Violent Tornadoes, Flash Floods, Excessive Heat | $6.3$ hrs | $\approx \$1.2\text{M}$ | **$5.21$** ($4.04$ inj, $1.17$ deaths) |
| **Cluster 3** | **Frequent Localized Convective Storms** | 1,907 ($76.3\%$) | Thunderstorm Wind, Hail, High Wind, Flash Flood | $< 10$ mins | $\approx \$450\text{K}$ | $0.02$ |

---

## 6. Interactive Web Studio Architecture

The web studio (`app.py` + `templates/index.html`) requires no external builds or frameworks like React:
- **Dashboard & KPIs:** Live summaries of dataset metrics, optimal configurations, and radar profiles.
- **Preprocessing Explorer:** Transformation matrix and interactive EDA distribution charts.
- **Hierarchical Lab:** Truncated dendrogram gallery with switcher (Ward, Complete, Average, Single) and metric tables.
- **DBSCAN Density Lab:** $k$-distance knee curve, parameter sensitivity matrix, and noise breakdown.
- **PCA Visualizer:** Interactive 2D scatter plot with tooltip inspections and toggle between Ward and DBSCAN coloring.
- **US Storm Map:** Interactive Leaflet map plotting storm locations across the continental US color-coded by cluster.
- **Live Event Classifier:** Scenario simulator allowing users to input storm duration, damages, casualties, and month to receive real-time cluster classification, archetype descriptions, and live projection onto the PCA scatter plot.
- **Stability & Responsible AI:** ARI stability matrix and ethical guidelines.

---

## 7. Responsible Interpretation, Biases & Limitations

1. **Exploratory Partitions, Not Immutable Laws:** Unsupervised groupings are mathematical models of distance in the chosen feature space. Boundaries between clusters reflect continuous transitions rather than rigid physical categories.
2. **NOAA Reporting Biases:** Storm data relies heavily on National Weather Service spotters, emergency management, and public reports. Populated urban regions naturally record higher frequencies of hail, wind, and localized damage than unpopulated rural areas.
3. **Nominal Monetary Figures:** Damage estimates are reported in nominal USD at the time of the event and are not inflation-adjusted across the 2010–2020 window.
4. **Public Safety Caveat:** While clustering assists hazard mitigation and resource planning, it must **never** be used to downplay localized risks. Every severe thunderstorm or winter system possesses life-threatening potential regardless of cluster membership.

---

## 8. Repository Structure

```
project/
├── data/
│   ├── raw/                      # NOAA raw CSVs (excluded from git)
│   └── processed/                # cleaned_storm_data.csv
├── outputs/
│   ├── cluster_profiles/         # Statistical CSV profiles and event distributions
│   ├── metrics/                  # Evaluation tables, ARI stability, selection notes
│   ├── plots/                    # High-res diagnostic PNG plots
│   └── dashboard_data.json       # Consolidated JSON payload for web frontend
├── src/
│   ├── data_loader.py            # Data loading routines
│   ├── dbscan.py                 # DBSCAN grid search & k-distance diagnostic
│   ├── dimensionality_reduction.py # PCA fitting, scree plot, loadings biplot
│   ├── eda.py                    # Exploratory data analysis & correlations
│   ├── evaluation.py             # Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI
│   ├── feature_engineering.py    # Log1p transforms, cyclical seasonality, imputation
│   ├── hierarchical.py           # Ward, Complete, Average, Single linkage experiments
│   ├── interpretation.py         # Profile generation & responsible AI documentation
│   ├── preprocessing.py          # NOAA CSV cleaning & damage conversion
│   └── visualization.py          # Stability matrix heatmap plotting
├── static/
│   ├── app.js                    # Interactive dashboard logic, Chart.js & Leaflet
│   └── style.css                 # Custom glassmorphic styling & dark mode rules
├── templates/
│   └── index.html                # Responsive single-page web studio
├── app.py                        # Flask server & real-time prediction endpoint
├── main.py                       # CLI analysis pipeline coordinator
├── requirements.txt              # Project Python dependencies
└── README.md                     # Comprehensive project documentation
```

---

## 9. Verification & Testing

To run the complete pipeline and verify all outputs:
```powershell
python -u main.py
python app.py
```
Both the CLI analysis and interactive web studio run smoothly with 100% reproducibility.
