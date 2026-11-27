# Data

The datasets are **not** included in this repository. Download them from their official sources and arrange them as below (or point `CCSATDA_DATA` at a folder with this layout).

```
data/
├── ISIC2018_Task3_Training_Input/            # ISIC 2018 Task 3 training images (*.jpg)
├── ISIC2018_Task3_Training_GroundTruth/
│   └── ISIC2018_Task3_Training_GroundTruth.csv
├── ISIC2018_Task3_Test_Input/
├── ISIC2018_Task3_Test_GroundTruth/
│   └── ISIC2018_Task3_Test_GroundTruth.csv
├── metadata.csv                              # PAD-UFES-20 metadata
└── images/
    ├── imgs_part_1/                          # PAD-UFES-20 images (*.png)
    ├── imgs_part_2/
    └── imgs_part_3/
```

| Dataset | Role | Source |
|---|---|---|
| ISIC 2018 Task 3 (HAM10000) | Source domain, dermoscopy, labelled | https://challenge.isic-archive.com/data/#2018 |
| PAD-UFES-20 | Target domain, smartphone photos | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |

## Class mapping

Five classes shared by both domains:

| Shared class | ISIC 2018 | PAD-UFES-20 |
|---|---|---|
| BCC | BCC | BCC |
| MEL | MEL | MEL |
| NV | NV | NEV |
| AKIEC | AKIEC | ACK + SCC |
| BKL | BKL | SEK |

ISIC classes DF and VASC have no counterpart in PAD-UFES-20 and are dropped.

## Splits

The splits are deterministic (`random_state=42`) and built by `build_datasets_with_nested_subsets` in `src/run_ultra_low_resource_study.py`:

- **ISIC 2018 training set:** 85/15 split into train/val, giving 8,294 training images.
- **PAD-UFES-20 (2,298 photos):**
  - stratified 70/15/15 split into a training pool (1,608), validation (345) and test (345, of which 8 are melanoma);
  - the low-label budgets are nested stratified subsets of the pool: 50 ⊂ 100 ⊂ 200 ⊂ 1,608.

The extended study (`src/train_cc_satda_v2.py`) uses its own split, where AKIEC = ACK only, giving 1,474 / 316 / 316 images. Its numbers are therefore **not** directly comparable to the main table.
