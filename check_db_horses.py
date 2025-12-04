
import sqlite3

conn = sqlite3.connect('data/db/hipica_normalizada.db')
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM participantes WHERE (nombre_caballo IS NULL OR nombre_caballo = '') AND resultado_final IS NULL")
count = cursor.fetchone()[0]
print(f"Empty horses count: {count}")

if count > 0:
    cursor.execute("SELECT * FROM participantes WHERE (nombre_caballo IS NULL OR nombre_caballo = '') AND resultado_final IS NULL LIMIT 5")
    rows = cursor.fetchall()
    print("Sample empty rows:")
    for row in rows:
        print(row)

conn.close()
