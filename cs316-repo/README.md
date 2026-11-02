# Resource-Aware Crop Recommendation

CS316 Introduction to AI and Data Science · Prince Sultan University · Term 261

Author: Abdulrahman Alsohaibani (221110482)

A machine-learning system that recommends which crop to plant from soil and
weather readings and, among suitable crops, prefers the ones that need less
added fertiliser and irrigation water. The project supports UN Sustainable Development Goals
2 (Zero Hunger), 6 (Clean Water), 12 (Responsible Consumption and Production)
and 13 (Climate Action).

## 1. Overview

A standard crop recommender returns the single most likely crop for a field.
That crop may be suitable but expensive in inputs, for example a crop that
grows with 230 mm of rain planted on a field that gets 45 mm. This project
adds a re-ranking layer on top of the classifier. It looks at the
model's top-3 suitable crops, estimates how much fertiliser (N, P, K) and
water each would need *on top of* what the field already has, and picks
the crop with the best balance of suitability and resource cost.

## 2. Results at a glance

| Item | Result |
|---|---|
| Dataset | 2,200 samples, 7 features, 22 crops (100 each), no missing values |
| Models compared | 8 (Logistic Regression, Gaussian NB, k-NN, SVM, Decision Tree, Random Forest, Extra Trees, Hist Gradient Boosting) |
| Selected model | Random Forest (100 trees, max_depth 12, max_features "sqrt") |
| 5-fold CV accuracy | 99.55 % ± 0.56 |
| Held-out test accuracy | 99.55 % (438 / 440), macro F1 0.995, top-3 accuracy 100 % |
| Re-ranking, noisy fields (20 % sensor noise) | water gap −21 %, fertiliser gap −6 %, accuracy change −0.9 pp |
| Re-ranking, 50 % less rain | water gap −29 %, fertiliser gap +23 %, total resource cost −24 % |
| Memory: DataFrame | 246,832 B → 52,523 B (−78.7 %) with int16 / float32 / category |
| Memory: saved model | 3.23 MB → 0.43 MB with joblib compression |

All numbers are produced by `python main.py` and saved in `results/metrics.json`.

## 3. How to run

```bash
git clone <this-repo-url>
cd cs316-repo
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                     # about 1 minute on a laptop
```

The script prints each step and writes:

* `figures/`: Figures 1–9 used in the report and slides
* `results/`: CSV tables, `metrics.json` and the trained model `final_model.joblib`

To get a recommendation for your own field:

```python
import joblib, numpy as np, pandas as pd
from src.sustainability import ResourceAwareRecommender

model   = joblib.load("results/final_model.joblib")
profile = pd.read_csv("results/crop_resource_profile.csv", index_col=0)
rec     = ResourceAwareRecommender(model, profile)
rec.fert_scale, rec.water_scale = 258.0, 122.67   # values learned in main.py

# N, P, K, temperature (°C), humidity (%), pH, rainfall (mm)
field = [40, 35, 30, 30.0, 35.0, 7.6, 40.0]
print(rec.explain(field))
```

(`crop_resource_profile.csv` holds medians from the full dataset; `main.py`
uses medians from the training rows only, which differ by at most a few units.)

## 4. Repository structure

```
cs316-repo/
├── main.py                 # runs the whole pipeline end to end
├── requirements.txt
├── data/
│   └── crop_recommendation.csv
├── src/
│   ├── config.py           # paths, column names, dtypes, seeds, settings
│   ├── data_loader.py      # memory-optimised loading, tracemalloc, chunked reading
│   ├── eda.py              # summary statistics and Figures 1-4
│   ├── models.py           # 8 models + grid search + model choice
│   ├── evaluation.py       # confusion matrix, importance, noise test, ablation (Figs 5-8)
│   └── sustainability.py   # resource-aware re-ranking layer (Fig 9)
├── figures/                # generated charts
└── results/                # generated tables, metrics.json, final model
```

## 5. Method summary

1. Load with compact dtypes (int16 for N/P/K, float32 for climate,
   category for the label) and measure the saving with
   `DataFrame.memory_usage(deep=True)` and `tracemalloc`. A chunked reader
   (500 rows at a time) shows the statistics can be computed without
   loading the whole file.
2. EDA: class balance, distributions, correlations, and a
   per-crop input-needs table that later becomes the re-ranker's
   "requirement profile".
3. Split 80/20, stratified by crop, seed 42.
4. Model selection: each of the 8 models is tuned with `GridSearchCV`
   and 5-fold stratified CV on the training set only. Scale-sensitive models
   use a `Pipeline` with `StandardScaler` to avoid leakage. The final model is
   chosen by CV accuracy, not test accuracy.
5. Evaluation: confusion matrix, per-class precision/recall/F1,
   permutation importance, top-3 accuracy, robustness to added sensor noise,
   and an ablation without soil data.
6. Sustainability layer: for each field and each of the top-3 crops,
   fertiliser gap = Σ max(0, need − soil) over N, P, K and water gap =
   max(0, rain need − rainfall). Both are scaled to 0–1, and
   score = α·probability − (1−α)·cost with α = 0.5.

## 6. Dataset

Crop Recommendation Dataset by Atharva Ingle, published on Kaggle:
<https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>.
The copy in `data/` was taken from the public mirror in the
[Harvestify](https://github.com/Gladiator07/Harvestify) repository
(`Data-processed/crop_recommendation.csv`). Columns: N, P, K (soil nutrient
ratios), temperature (°C), humidity (%), pH, rainfall (mm), label (crop).

## 7. Limitations

* The dataset is clean and its classes are well separated, which explains
  the very high accuracy. Real farm data will be noisier (see the noise test).
* The data reflects Indian growing conditions. Crop needs are taken from the
  data, not from agronomic tables.
* The re-ranker estimates the gap in inputs, not their price or
  environmental cost per unit.

## 8. Use of AI

AI assistance was used while writing this code and documentation. All code
was run and checked, and every reported number comes from `main.py`.
