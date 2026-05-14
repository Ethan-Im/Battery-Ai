import pandas as pd
import numpy as np
from pymatgen.core import Structure
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/alignn_dataset.csv")
df = df.dropna(subset=["log_conductivity"])
print(f"전체 데이터: {len(df)}개")

# 구조 로드
structures, targets, valid_idx = [], [], []
for i, row in df.iterrows():
    try:
        struct = Structure.from_file(row["cif_path"])
        structures.append(struct)
        targets.append(row["log_conductivity"])
        valid_idx.append(i)
    except:
        pass

print(f"유효 구조: {len(structures)}개")

# train/test split
idx = list(range(len(structures)))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

train_structs = [structures[i] for i in train_idx]
train_targets = [targets[i] for i in train_idx]
test_structs = [structures[i] for i in test_idx]
test_targets = [targets[i] for i in test_idx]

# MEGNet 모델
model = MEGNetModel(100, 2, nblocks=3, lr=1e-3)

print("학습 시작...")
model.train(
    train_structs, train_targets,
    validation_structures=test_structs,
    validation_targets=test_targets,
    epochs=100,
    batch_size=16,
    verbose=1,
)

# 평가
pred = model.predict_structures(test_structs)
pred = [p[0] for p in pred]

mae = mean_absolute_error(test_targets, pred)
r2 = r2_score(test_targets, pred)
print(f"\nMAE: {mae:.3f} (log S/cm)")
print(f"R²:  {r2:.3f}")

model.save_model("models/megnet_ionic.hdf5")
print("모델 저장 완료: models/megnet_ionic.hdf5")
