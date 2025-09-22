# P170M109 — CIDM Lab 1: Input Analysis & Price Prediction (KNN / DT / RF)

This project implements **P1 → P3** of the lab:

* **P1**: Input data analysis & preprocessing (before/after)
* **P2**: Predict apartment **monthly price** using **KNN**, **Decision Tree**, **Random Forest**
* **P3**: Hyperparameter selection and evaluation on Train/Val/Test (MAE, MAPE, RMSE, R²)

Everything is done in a single, editable Jupyter notebook.

---

## 📁 Contents

* **`CIDM_LD1_P1_to_P3_FINAL.ipynb`** — main notebook with:

  * **Robust data loading** via an upward-search helper (`find_file_upwards`) for:

    * `apartments_for_rent_classified_100K.csv`
    * `apartments_for_rent_classified_10K.csv`
  * **P1 – Data analysis & preprocessing**

    * Numeric vs categorical **data-quality reports** (missing %, unique, summary stats, skew)
    * **Before/after** plots:

      * `price_monthly` (normalized from `price` + `price_type`)
      * `square_feet` (hist/box)
      * Top-15 categorical bar (e.g., state/city/region)
    * **Derived features**: `pet_cat_allowed`, `pet_dog_allowed`, `amenity_count`, `price_per_sqft`
    * **ABT** assembly; **preprocessing**: USD filter, coordinate sanity, size sanity, clip top 0.5% of price, median-impute numerics, de-duplicate
    * Standardization demo (z-score) for numerics used by KNN
  * **P2 – Modeling**

    * **KNN**: k ∈ {3, 5, 10} (uses **scaled** numerics)
    * **Decision Tree**: `max_depth` ∈ {5, 10, None} × `min_samples_leaf` ∈ {1, 5, 10}
    * **Random Forest**: `n_estimators` ∈ {50, 100, 200} × `min_samples_leaf` ∈ {1, 2}
    * Baseline (mean predictor)
  * **P3 – Selection & final test**

    * Pick best hyperparameters by **Validation MAE**
    * Retrain on **Train+Val**, evaluate on **Test**
    * Print **MAE, MAPE, RMSE, R²** consistently
    * (Optional) Random Forest **feature importances**

---

## 🗄️ Data

* Source: **UCI – Apartment for Rent Classified**
  [https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified)

The notebook looks for, in order:

1. `apartments_for_rent_classified_100K.csv`
2. `apartments_for_rent_classified_10K.csv`

It will search in the current directory and upward through parent folders. You can also place the CSVs in `/mnt/data/` if you’re on a hosted environment.

> **Encoding tip** (from assignment): If the CSV opens garbled, in Excel use
> **Data → From Text/CSV → Import → Load**, then save as **UTF-8** CSV.

---

## ▶️ Quick start

```bash
# 1) Install deps (Python 3.9+ recommended)
pip install -U pandas numpy scikit-learn matplotlib

# 2) Open the notebook
jupyter notebook CIDM_LD1_P1_to_P3_FINAL.ipynb
# Run cells top-to-bottom
```

---

## 🔧 Notes & tweaks

* **Target normalization**: `price_monthly` is computed from `price` + `price_type`

  * weekly → × 52/12, daily → × 30, fortnightly → × 26/12, yearly → ÷ 12, hourly → × (24×30), else monthly
* **Scaling**: applied to **numeric** features for KNN (trees are scale-invariant; kept consistent across models)
* **Outliers**: top **0.5%** of `price_monthly` clipped for stability (adjust `0.995` quantile)
* **Size sanity**: keep `square_feet` in **\[120, 8000]** (tweak if your data needs)
* **Hyperparameters**: extend lists for a wider search if desired
* **Everything is plain code** — you can comment/modify any line

---

## 📚 References

* scikit-learn KNN (regression):
  [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
* scikit-learn Decision Trees:
  [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)
* scikit-learn RandomForestRegressor:
  [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
* Dataset:
  [https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified)

---

