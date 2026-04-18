import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS camera_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    object_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    image_base64 TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS system_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plastic_weight REAL NOT NULL,
    glass_weight REAL NOT NULL,
    metal_weight REAL NOT NULL,
    paper_weight REAL NOT NULL,
    plastic_status TEXT NOT NULL,
    glass_status TEXT NOT NULL,
    metal_status TEXT NOT NULL,
    paper_status TEXT NOT NULL,
    emergency_stop INTEGER NOT NULL,
    motor_status TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
""")

conn.commit()
conn.close()

print("Database and tables created successfully.")
