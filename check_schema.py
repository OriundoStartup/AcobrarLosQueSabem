import sqlite3

conn = sqlite3.connect('data/db/hipica_3fn.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print("=== TABLAS EN LA BASE DE DATOS ===\n")
for table in tables:
    table_name = table[0]
    print(f"\n--- {table_name} ---")
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    schema = cursor.fetchone()[0]
    print(schema)

conn.close()
