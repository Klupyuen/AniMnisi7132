import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

print("\n=== camera_events ===")
cursor.execute("SELECT * FROM camera_events")
for row in cursor.fetchall():
    print(row[:4], "... image_base64 omitted" if row[4] else "")

print("\n=== system_states ===")
cursor.execute("SELECT * FROM system_states")
for row in cursor.fetchall():
    print(row)

conn.close()
