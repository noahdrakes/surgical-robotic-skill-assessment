import pandas as pd
import matplotlib.pyplot as plt

# ---- Load CSV ----
csv_path = "accel_left1.csv"   # <-- change this to your filename
df = pd.read_csv(csv_path)

# ---- Time axis (convert ns to seconds, relative) ----
t_ns = df["ts_ns"].to_numpy()
t = (t_ns - t_ns[0]) * 1e-9  # seconds, start at 0

# ---- Extract linear acceleration ----
ax = df["linear_accel_x"].to_numpy()
ay = df["linear_accel_y"].to_numpy()
az = df["linear_accel_z"].to_numpy()

# ---- Plot ----
plt.figure(figsize=(10, 5))
plt.plot(t, ax, label="linear_accel_x")
plt.plot(t, ay, label="linear_accel_y")
plt.plot(t, az, label="linear_accel_z")

plt.xlabel("Time (s)")
plt.ylabel("Linear Acceleration")
plt.title("Linear Acceleration (XYZ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# ---- Save instead of show ----
output_path = "linear_acceleration_xyz.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved plot to: {output_path}")
plt.show()