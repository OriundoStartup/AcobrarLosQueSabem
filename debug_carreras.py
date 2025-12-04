import sqlite3
import pandas as pd

conn = sqlite3.connect('data/db/hipica_3fn.db')

# Verificar carreras futuras
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM fact_carreras WHERE fecha >= date('now')")
print(f"Carreras futuras en BD: {cursor.fetchone()[0]}")

# Ver últimas fechas
cursor.execute("SELECT fecha, COUNT(*) as total FROM fact_carreras GROUP BY fecha ORDER BY fecha DESC LIMIT 5")
print("\nÚltimas fechas en BD:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} carreras")

# Probar la query CORREGIDA de get_proximas_carreras
print("\nProbando query CORREGIDA:")
df = pd.read_sql_query("""
    SELECT 
        h.nombre as Hipodromo,
        c.fecha as Fecha,
        c.nro_carrera as Carrera,
        'N/A' as Hora,
        c.distancia_metros || 'm' as Distancia,
        'Carrera' as Condicion,
        (SELECT COUNT(*) FROM fact_participaciones p WHERE p.carrera_id = c.id) as Participantes
    FROM fact_carreras c
    JOIN dim_hipodromos h ON c.hipodromo_id = h.id
    WHERE c.fecha >= date('now', '-7 days')
    ORDER BY c.fecha DESC, c.nro_carrera ASC
""", conn)

print(f"Carreras encontradas: {len(df)}")
if len(df) > 0:
    print("\nPrimeras 5 carreras:")
    print(df.head())

conn.close()
