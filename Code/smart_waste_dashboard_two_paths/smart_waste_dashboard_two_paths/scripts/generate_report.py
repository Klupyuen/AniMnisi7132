import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("docs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

conn = sqlite3.connect("database.db")
camera_df = pd.read_sql_query("SELECT * FROM camera_events", conn)
system_df = pd.read_sql_query("SELECT * FROM system_states", conn)
conn.close()

if system_df.empty:
    print("No system state data found. Generate test data first.")
    raise SystemExit

system_df["timestamp"] = pd.to_datetime(system_df["timestamp"])
system_df = system_df.drop_duplicates().sort_values("timestamp")

for col in ["plastic_weight", "glass_weight", "metal_weight", "paper_weight"]:
    system_df[col] = system_df[col].clip(lower=0)

system_df["total_weight"] = (
    system_df["plastic_weight"] +
    system_df["glass_weight"] +
    system_df["metal_weight"] +
    system_df["paper_weight"]
)

latest = system_df.iloc[-1]
latest_weights = {
    "plastic": latest["plastic_weight"],
    "glass": latest["glass_weight"],
    "metal": latest["metal_weight"],
    "paper": latest["paper_weight"]
}
heaviest = max(latest_weights, key=latest_weights.get)

summary_path = os.path.join("docs", "kpi_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("KPI SUMMARY\n")
    f.write("=====================\n")
    f.write(f"Heaviest compartment: {heaviest} ({latest_weights[heaviest]:.2f} g)\n")
    f.write(f"Total latest weight: {latest['total_weight']:.2f} g\n")
    f.write(f"Average total weight: {system_df['total_weight'].mean():.2f} g\n")
    f.write(f"Weight std dev: {system_df['total_weight'].std():.2f} g\n")
    f.write(f"Emergency stop count: {int(system_df['emergency_stop'].sum())}\n")
    if not camera_df.empty:
        f.write(f"Latest detected type: {camera_df.iloc[-1]['type']}\n")
        f.write(f"Latest object name: {camera_df.iloc[-1]['object_name']}\n")

plt.figure(figsize=(9, 4))
plt.plot(system_df["timestamp"], system_df["total_weight"])
plt.title("Total Weight Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Total Weight (g)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "total_weight_over_time.png"))
plt.close()

plt.figure(figsize=(7, 4))
plt.bar(list(latest_weights.keys()), list(latest_weights.values()))
plt.title("Latest Compartment Weights")
plt.xlabel("Compartment")
plt.ylabel("Weight (g)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "latest_compartment_weights.png"))
plt.close()

plt.figure(figsize=(7, 4))
plt.hist(system_df["total_weight"], bins=10)
plt.title("Distribution of Total Weight")
plt.xlabel("Total Weight (g)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "total_weight_histogram.png"))
plt.close()

full_counts = {
    "plastic": int((system_df["plastic_status"] == "FULL").sum()),
    "glass": int((system_df["glass_status"] == "FULL").sum()),
    "metal": int((system_df["metal_status"] == "FULL").sum()),
    "paper": int((system_df["paper_status"] == "FULL").sum())
}
plt.figure(figsize=(7, 4))
plt.bar(list(full_counts.keys()), list(full_counts.values()))
plt.title("Number of FULL Status Records")
plt.xlabel("Compartment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "full_status_counts.png"))
plt.close()

if not camera_df.empty:
    camera_type_counts = camera_df["type"].value_counts()
    plt.figure(figsize=(7, 4))
    plt.bar(camera_type_counts.index.tolist(), camera_type_counts.values.tolist())
    plt.title("Detected Type Distribution")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "camera_type_distribution.png"))
    plt.close()

mean_val = system_df["total_weight"].mean()
std_val = system_df["total_weight"].std() if system_df["total_weight"].std() != 0 else 1.0
system_df["z_score"] = (system_df["total_weight"] - mean_val) / std_val
anomalies = system_df[np.abs(system_df["z_score"]) > 2]

plt.figure(figsize=(9, 4))
plt.plot(system_df["timestamp"], system_df["total_weight"], label="Total Weight")
plt.scatter(anomalies["timestamp"], anomalies["total_weight"], label="Anomaly")
plt.title("Anomaly Detection on Total Weight")
plt.xlabel("Timestamp")
plt.ylabel("Total Weight (g)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "anomaly_detection.png"))
plt.close()

anomaly_csv = os.path.join("docs", "anomaly_table.csv")
anomalies[["timestamp", "total_weight", "z_score"]].to_csv(anomaly_csv, index=False)

system_export_csv = os.path.join("docs", "system_states_export.csv")
camera_export_csv = os.path.join("docs", "camera_events_export.csv")
system_df.to_csv(system_export_csv, index=False)
camera_df.to_csv(camera_export_csv, index=False)

print("Analytics report generated successfully.")
