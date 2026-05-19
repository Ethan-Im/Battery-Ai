import pandas as pd

df1 = pd.read_csv("data/inverse_design_candidates.csv")
df2 = pd.read_csv("data/smart_search_candidates.csv")

df1 = df1.rename(columns={"predicted_log_conductivity": "log_conductivity"})

df = pd.concat([df1, df2]).drop_duplicates("formula")
df = df.sort_values("log_conductivity", ascending=False)
df["predicted_conductivity"] = 10 ** df["log_conductivity"]

print("🏆 최종 Top 20 후보:")
print(df.head(20)[["formula", "log_conductivity", "predicted_conductivity"]].to_string(index=False))

df.to_csv("data/final_candidates.csv", index=False)
print("\n저장 완료: data/final_candidates.csv")
