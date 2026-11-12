# Resource-Aware Crop Recommendation

CS316 Introduction to AI and Data Science · Prince Sultan University · Term 261

Author: Abdulrahman Alsohaibani (221110482)

A machine-learning model that recommends which crop to plant from soil and weather readings. When more than one crop is suitable, it prefers the crop that needs less extra fertiliser and irrigation water. The project supports UN Sustainable Development Goals 2 (Zero Hunger), 6 (Clean Water), 12 (Responsible Consumption and Production) and 13 (Climate Action).

## What the notebook does

| Section | Content |
|---|---|
| 1. Load the data | 2,200 fields, 7 features, 22 crops; memory optimisation with smaller data types (−64.7 %) |
| 2. Cleaning and outliers | No missing values or duplicates; IQR outlier check (outliers kept, they belong to real crops) |
| 3. Exploration | Class balance, histograms, correlation heatmap, rainfall per crop |
| 4. Hypothesis testing | One-way ANOVA: all 7 features differ significantly between crops |
| 5. Normalisation and PCA | StandardScaler fitted on the training set; PCA for a 2D view |
| 6. Models | Logistic Regression, Naive Bayes, k-NN, Decision Tree (entropy), Random Forest with 5-fold cross-validation |
| 7. Evaluation | Random Forest: 99.55 % test accuracy (438 / 440), confusion matrix, feature importance |
| 8. Sustainability step | Re-ranks the top 3 crops by fertiliser and water shortfall; in a dry season it cuts the extra water needed by 29 % |
| 9. New field | Try the recommender on any field |
| 10–11 | Ethics, limitations and conclusion |

## How to run

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Jupyter and open `CS316_Crop_Recommendation.ipynb`:
   ```bash
   jupyter notebook
   ```
   (Or open the notebook in VS Code with the Python and Jupyter extensions.)
4. Click **Run All**. The whole notebook runs in under a minute and needs no internet connection.

All charts are also saved to the `figures/` folder.

## Folder structure

```
cs316-repo/
├── CS316_Crop_Recommendation.ipynb   # all code, outputs and explanations
├── data/crop_recommendation.csv      # dataset
├── figures/                          # charts produced by the notebook
├── requirements.txt
└── README.md
```

## Dataset

Crop Recommendation Dataset by Atharva Ingle, Kaggle: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset (copy from the GitHub repository Gladiator07/Harvestify).
