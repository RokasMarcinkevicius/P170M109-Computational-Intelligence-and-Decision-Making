# CIDM Lab 2 Package

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


## Repository Structure

```
cidm_lab2_improved/
├── data_classification/
│   ├── train/
│   │   ├── classA/  (red square images)
│   │   └── classB/  (blue circle images)
│   ├── val/
│   │   ├── classA/
│   │   └── classB/
│   └── test/
│       ├── classA/
│       └── classB/
├── data_object_detection/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
├── notebooks/
│   └── (place your Jupyter notebooks here)
├── classification_task.py
└── README.md
```

### Classification Dataset

The classification dataset is deliberately simple: **Class A** consists of
images containing a red square on a white background, and **Class B**
contains images with a blue circle.  There are 10 training images, 1
validation image and 1 test image per class.  You can easily add more
variety by creating additional images or introducing noise/rotation to the
shapes.

### Object Detection Dataset

The detection dataset reuses a similar idea.  Each image contains exactly
one shape (either a red square or a blue circle).  The corresponding label
files in `labels/` contain a single line with the format:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised to the range [0, 1], which is the format
expected by YOLOv8.  The `data.yaml` file configures the names of the
classes and the relative paths to the images and labels; it can be passed
directly to the Ultralytics CLI for training or evaluation.

## Baseline Models and Metrics

The script [`classification_task.py`](classification_task.py) provides a
complete example of loading tabular data from a CSV file, performing
preprocessing, training several baseline regression models (K‑Nearest
Neighbours, Decision Tree, Random Forest) and evaluating them.  The
following metrics are reported:

| Metric | Description | Interpretation |
|---|---|---|
| **MAE (Mean Absolute Error)** | The average absolute difference between actual and predicted values. | Lower values indicate more accurate predictions. |
| **MAPE (Mean Absolute Percentage Error)** | The average absolute percentage difference; insensitive to scale. | Lower values (closer to 0%) are better. |
| **RMSE (Root Mean Squared Error)** | Square root of the average squared error; penalises large errors. | Lower values indicate better fit, but units remain in the original scale. |
| **R² (Coefficient of Determination)** | Proportion of target variance explained by the model. | Values closer to 1 indicate that the model explains most of the variability. |

The accompanying code defines helper functions `mape_masked`, `rmse`, and
`train_and_evaluate_model` to make these calculations explicit.  All
functions and sections are annotated with docstrings and inline comments so
that you can learn from reading the code itself.