import pandas as pd
import numpy as np

df = pd.read_parquet("Flimdejong_block_or_pen_ee_delta.parquet")
deltas = np.stack(df["delta_ee"].to_numpy())  # shape (63740, 7)
labels = ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "d_gripper"]

for i, label in enumerate(labels):
    col = deltas[:, i]
    print(f"{label:10s}  min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}  std={col.std():.4f}")

print("any NaN:", np.isnan(deltas).any())
print("any inf:", np.isinf(deltas).any())

print("unique episodes:", df["episode_index"].nunique())