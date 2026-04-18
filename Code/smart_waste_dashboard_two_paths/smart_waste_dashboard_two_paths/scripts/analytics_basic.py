import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("database.db")
camera_df = pd.read_sql_query("SELECT * FROM camera_events", conn)
system_df = pd.read_sql_query("SELECT * FROM system_states", conn)
conn.close()

print("\n=== CAMERA EVENTS HEAD ===")
print(camera_df.head())

print("\n=== SYSTEM STATES HEAD ===")
print(system_df.head())

if system_df.empty:
    print("\nNo system state data found.")
    raise SystemExit

system_df["timestamp"] = pd.to_datetime(system_df["timestamp"])
system_df["total_weight"] = (
    system_df["plastic_weight"] +
    system_df["glass_weight"] +
    system_df["metal_weight"] +
    system_df["paper_weight"]
)

print("\n=== BASIC STATISTICS ===")
print(system_df[["plastic_weight", "glass_weight", "metal_weight", "paper_weight", "total_weight"]].describe())

latest = system_df.iloc[-1]
weights = {
    "plastic": latest["plastic_weight"],
    "glass": latest["glass_weight"],
    "metal": latest["metal_weight"],
    "paper": latest["paper_weight"]
}
heaviest = max(weights, key=weights.get)

print("\n=== KPI SUMMARY ===")
print(f"Heaviest compartment: {heaviest} ({weights[heaviest]:.2f} g)")
print(f"Total latest weight: {latest['total_weight']:.2f} g")
print(f"Emergency stop count: {int(system_df['emergency_stop'].sum())}")

if not camera_df.empty:
    print("\n=== CAMERA TYPE COUNTS ===")
    print(camera_df["type"].value_counts())
