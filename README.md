# Battery AI — Solid Electrolyte Ionic Conductivity Predictor

ML pipeline for predicting and discovering Li-based solid electrolyte materials with high ionic conductivity.

## Results
| Model | Data | Features | MAE (log S/cm) | R² |
|-------|------|----------|----------------|-----|
| Random Forest v1 | 49 | 7 | 2.289 | 0.276 |
| Random Forest v2 | 598 | 132 | 1.224 | 0.660 |
| XGBoost v2 | 598 | 132 | 1.149 | **0.675** |

> Successfully reproduced OBELiX paper benchmark (RF baseline R² ~0.65)

## Top Inverse-Designed Candidates
| Formula | Predicted Conductivity (S/cm) |
|---------|-------------------------------|
| Li5BrP2ClS4 | 1.46e-3 |
| Li7P2BrS4 | 5.02e-4 |
| Li7Cl7P6 | 2.44e-4 |
| Li5PS | 1.32e-4 |
| Li6Cl3P4 | 1.22e-4 |

> Li-P-S and Li-P-S-halide systems dominate top candidates, consistent with known Argyrodite-family electrolytes.

## Data Sources
- [Materials Project](https://materialsproject.org) — 9,992 Li-containing materials
- [OBELiX](https://github.com/NRC-Mila/OBELiX) — 599 experimentally measured ionic conductivities

## Pipeline
## Installation
```bash
conda create -n battery-ai python=3.11 -y
conda activate battery-ai
pip install mp-api pymatgen chgnet pandas scikit-learn matminer matplotlib xgboost
pip install git+https://github.com/NRC-Mila/OBELiX.git
```

## Usage
```bash
python3 data/fetch_mp.py
python3 data/fetch_obelix.py
python3 data/explore.py
python3 data/merge_datasets.py
python3 models/train_xgb_v2.py
python3 models/inverse_design.py
```

## Roadmap
- [x] Random Forest baseline (R² = 0.66)
- [x] XGBoost with hyperparameter tuning (R² = 0.675)
- [x] Inverse design pipeline
- [ ] GNN-based model (target R² > 0.75)
- [ ] Screening report for LG Energy Solution / Samsung SDI
