# Extreme Weather Event Grouping

An introductory, reproducible unsupervised learning project using NOAA's Storm Events Details data. It compares agglomerative hierarchical clustering and DBSCAN, with PCA plots, silhouette analysis, cluster profiles, and parameter stability. It uses Python, pandas, NumPy, matplotlib, seaborn, SciPy, and scikit-learn; no notebook or web framework is required.

## Dataset

Put `StormEvents_details_2010_2020.csv` in `data/raw/`. This project currently analyzes the 2010–2020 subset. It uses the details table because one row represents one event. It does not use the merged file, whose one-to-many joins can multiply events. The NOAA files are excluded from Git.

The preprocessing stage reads the details file and writes `data/processed/cleaned_storm_data.csv`. `main.py` reuses the processed file when it is newer than the raw CSV; otherwise it refreshes it automatically. The generated processed data should reflect the same 2010–2020 date range.

## Run in Windows PowerShell

From the project root:

```powershell
python -m pip install -r requirements.txt
python -u src/preprocessing.py
python -u main.py
```

`main.py` refreshes preprocessing when needed. The analysis uses a fixed random sample of up to 2,500 events (`random_state=42`) for both clustering methods. Hierarchical clustering requires pairwise comparisons, so it runs on this reproducible sample rather than the full dataset.

## Preprocessing and feature choices

- Both NOAA date columns are parsed using the verified `%d-%b-%y %H:%M:%S` format. No rows are dropped for missing magnitude or dates.
- Property and crop damage suffixes K, M, and B are converted to USD numeric columns. Original damage fields are retained. Unparsed values remain missing and are counted; missing damage is not turned into zero.
- Direct and indirect injuries and deaths are converted to numeric values; invalid negative counts become missing. Total injuries and deaths are then derived.
- The clustering features are log1p injuries, deaths, property damage, crop damage, and duration, plus cyclical month sine and cosine. The log transform reduces skew. `EVENT_TYPE` and `STATE` are excluded from distance calculations to avoid high-cardinality categorical influence; event type is used to profile clusters afterward.
- Missing model features are filled with the sample median and missingness indicators are added. Features are standardized before distance-based clustering. The feature preprocessing report records missing values and the imputation choice.

These are analytical choices, not claims that missing damage means no damage. Median imputation and missingness indicators still affect distances and should be considered when interpreting groups.

## Analysis methods

- **Hierarchical:** single, complete, average, and Ward linkage with Euclidean distance, at requested k values 3, 4, and 5. A truncated dendrogram is saved for each linkage. Ties can mean the actual number of groups is below requested k. Automatic selection screens out solutions whose smallest group is under 2% of the sample before comparing silhouette, because singleton branches can inflate the score; the exact rule is recorded in `selection_notes.txt`.
- **DBSCAN:** standardized features, eps values 0.8, 1.2, 1.6, and 2.0, and min_samples 8 and 15. The k-distance curve is provided as a diagnostic. Noise counts, noise percentage, cluster sizes, and parameter sensitivity are saved.
- **PCA:** two components are used only to visualise fitted assignments. Clustering and silhouette evaluation use the standardized imputed feature space, not the two-dimensional PCA projection. Explained variance and component loadings are saved.
- **Silhouette:** scores for hierarchical settings are computed on the same deterministic evaluation subset. DBSCAN silhouette is computed on non-noise observations only; it does not evaluate noise assignments. The selection note states the applied rule and flags when no DBSCAN setting meets it.
- **Stability:** adjusted Rand index (ARI) compares assignment pairs on the same sampled records. ARI measures agreement, not correctness.

## Outputs

Generated files go under `outputs/`:

- `plots/`: missing values, event-type frequency, correlations, dendrograms, k-distance, DBSCAN parameter sensitivity, PCA cluster views, silhouette scores, separate hierarchical and DBSCAN profile heatmaps, and ARI stability heatmap.
- `metrics/`: missingness, feature preprocessing, hierarchical and DBSCAN results, pairwise ARI, PCA explained variance/loadings, final comparison, and selection notes.
- `cluster_profiles/`: cluster sizes, mean/median impact and duration values, plus event-type proportions for selected hierarchical and DBSCAN assignments.

Generated CSVs and plots are ignored by Git; regenerate them with `python -u main.py`.

## Responsible interpretation and limitations

These are exploratory groupings, not permanent weather classes or causal findings. The NOAA record reflects changing reporting practices, event definitions, geographic coverage, and historical data availability. Damage values are not adjusted for inflation. Fatalities, injuries, damage, duration, and seasonal timing describe different aspects of an event, while the selected features, log transform, scaling, imputation, sample, linkage, and DBSCAN parameters all change the resulting geometry. DBSCAN may label many events as noise or merge groups as eps changes. A silhouette score is a compact separation measure and does not establish that a cluster is meaningful. Describe observed profiles and uncertainty rather than assigning definitive real-world categories.

## Suggested milestone commit

```powershell
git add main.py README.md src
git commit -m "complete hierarchical and density clustering analysis"
```
