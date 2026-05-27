import joblib, numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from obelix import OBELiX
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

ob = OBELiX()
rows = []
for entry in ob:
    rows.append({
        "formula": entry.get("Reduced Composition"),
        "ionic_conductivity": entry.get("Ionic conductivity (S cm-1)"),
        "space_group_num": entry.get("Space group #"),
        "a": entry.get("a"), "b": entry.get("b"), "c": entry.get("c"),
        "alpha": entry.get("alpha"), "beta": entry.get("beta"), "gamma": entry.get("gamma"),
    })

df = pd.DataFrame(rows).dropna(subset=["ionic_conductivity", "formula"])
df["log_conductivity"] = np.log10(df["ionic_conductivity"])
ep = ElementProperty.from_preset("magpie")

def safe_featurize(formula):
    try:
        return ep.featurize(Composition(formula))
    except:
        return None

df["magpie"] = df["formula"].apply(safe_featurize)
df = df.dropna(subset=["magpie"])

X_magpie = pd.DataFrame(df["magpie"].tolist(), columns=ep.feature_labels())
struct = ["space_group_num", "a", "b", "c", "alpha", "beta", "gamma"]
X_struct = df[struct].fillna(df[struct].median())
X = pd.concat([X_magpie, X_struct.reset_index(drop=True)], axis=1)
y = df["log_conductivity"].reset_index(drop=True)

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = joblib.load("models/ensemble_models.pkl")
xgb_pred = models["xgb"].predict(X_test)
lgbm_pred = models["lgbm"].predict(X_test)

ensemble = (xgb_pred + lgbm_pred) / 2
print(f"XGB+LGBM R²:  {r2_score(y_test, ensemble):.3f}")
print(f"XGB+LGBM MAE: {mean_absolute_error(y_test, ensemble):.3f}")
