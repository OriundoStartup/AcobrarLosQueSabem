
import sqlite3

conn = sqlite3.connect('data/db/hipica_normalizada.db')
cursor = conn.cursor()

# Delete empty horses
cursor.execute("DELETE FROM participantes WHERE nombre_caballo IS NULL OR nombre_caballo = ''")
print(f"Deleted {cursor.rowcount} empty rows")

# Create unique index to prevent duplicates
try:
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_participantes_carrera_caballo ON participantes(carrera_id, nombre_caballo)")
    print("Created unique index on (carrera_id, nombre_caballo)")
except Exception as e:
    print(f"Error creating index: {e}")

conn.commit()
conn.close()
