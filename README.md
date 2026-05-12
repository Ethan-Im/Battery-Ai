# Battery AI — Solid Electrolyte Ionic Conductivity Predictor

ML pipeline for predicting ionic conductivity of Li-based solid electrolyte materials.

## Results
| Model | Data | Features | MAE (log S/cm) | R² |
|-------|------|----------|----------------|-----|
| Random Forest v1 | 49 | 7 | 2.289 | 0.276 |
| Random Forest v2 | 598 | 132 | 1.224 | **0.660** |

> Successfully reproduced OBELiX paper benchmark (RF baseline R² ~0.65)

## Data Sources
- [Materials Project](https://materialsproject.org) — 9,992 Li-containing materials
- [OBELiX](https://github.com/NRC-Mila/OBELiX) — 599 experimentally measured ionic conductivities

## Pipeline
## Installation
```bash
conda create -n battery-ai python=3.11 -y
conda activate battery-ai
pip install mp-api pymatgen chgnet pandas scikit-learn matminer matplotlib
pip install git+https://github.com/NRC-Mila/OBELiX.git
```

## Usage
```bash
python3 data/fetch_mp.py
python3 data/fetch_obelix.py
python3 data/explore.py
python3 data/merge_datasets.py
python3 models/train_rf_v2.py
```

## Roadmap
- [ ] GNN-based model (target R² > 0.75)
- [ ] Inverse design pipeline for high-conductivity candidate generation
- [ ] Screening against LG Energy Solution / Samsung SDI target materials
