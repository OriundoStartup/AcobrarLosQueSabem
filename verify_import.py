import sqlite3
from pathlib import Path

db_path = "data/db/hipica_3fn.db"

if not Path(db_path).exists():
    print(f"❌ Base de datos no existe: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("="*60)
print("📊 ESTADÍSTICAS DE LA BASE DE DATOS")
print("="*60)

# Contar registros en cada tabla
tables = [
    ('dim_hipodromos', 'Hipódromos'),
    ('dim_caballos', 'Caballos'),
    ('dim_jinetes', 'Jinetes'),
    ('dim_entrenadores', 'Entrenadores'),
    ('dim_studs', 'Studs'),
    ('fact_carreras', 'Carreras'),
    ('fact_participaciones', 'Participaciones'),
]

for table_name, display_name in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"{display_name:20s}: {count:6d}")

print("\n" + "="*60)
print("📅 CARRERAS POR FECHA")
print("="*60)

cursor.execute("""
    SELECT fecha, COUNT(*) as num_carreras
    FROM fact_carreras
    GROUP BY fecha
    ORDER BY fecha DESC
""")

for row in cursor.fetchall():
    print(f"{row[0]}: {row[1]} carreras")

print("\n" + "="*60)
print("🏇 TOP 10 CABALLOS CON MÁS PARTICIPACIONES")
print("="*60)

cursor.execute("""
    SELECT c.nombre, COUNT(*) as participaciones
    FROM fact_participaciones p
    JOIN dim_caballos c ON p.caballo_id = c.id
    GROUP BY c.nombre
    ORDER BY participaciones DESC
    LIMIT 10
""")

for row in cursor.fetchall():
    print(f"{row[0]:30s}: {row[1]} participaciones")

conn.close()
print("\n✅ Verificación completada!")
