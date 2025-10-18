# CIDM Lab 2 Package (Final)

This package contains a **ready-to-run** notebook and templates to satisfy **all LD2 requirements**:

- **Part 1 – ANN Regression** on *the same dataset as Lab 1* (UCI Apartment for Rent Classified).
- **Part 2 – Image Classification** (from scratch + transfer learning) — provide your own dataset.
- **Part 3 – Object Detection / Segmentation** (YOLOv8; seg optional) — provide your own dataset and annotations.

## Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

> If you plan to run Parts 2–3, you need GPU-enabled PyTorch (see https://pytorch.org/get-started/locally/).
> For detection/segmentation with YOLOv8:
```bash
pip install ultralytics
```

### 2) Place your Lab 1 CSV(s)
Put either of these files in the notebook folder (or any parent folder), or `/mnt/data/`:
- `apartments_for_rent_classified_100K.csv`
- `apartments_for_rent_classified_10K.csv`

### 3) Provide your image datasets
- **Classification (Part 2):** Place images in `data_cls/{train,val,test}/{class}/...`.
- **Detection (Part 3):** Place images/labels following YOLO layout under `data_od/` and edit `data_od/data.yaml`:
  - images: `images/{train,val,test}`
  - labels: `labels/{train,val,test}`
  - set `names:` to your class list.

### 4) Run the notebook
```bash
jupyter notebook "CIDM_Lab2_Final.ipynb"
```

## What's inside

- `CIDM_Lab2_Final.ipynb` — full Part 1 with added plots/tables; complete templates for Parts 2–3.
- `requirements.txt` — base dependencies for Part 1 and plotting.
- `data_cls/` — empty structure to guide classification dataset layout.
- `data_od/` — YOLO template (images/labels structure + `data.yaml` stub).
- `templates/` — extra helper snippets.
- `scripts/` — small utilities (confusion matrix plotting, result tables).

## Notes
- **MAPE** handles divide-by-zero via masking.
- **No data leakage:** scalers/OHE are fit on train only via `ColumnTransformer`.
- **Same split** across baselines and ANN for fair comparison.
