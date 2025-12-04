import sqlite3
import pandas as pd

conn = sqlite3.connect('data/db/hipica_3fn.db')

# Ver hipódromos disponibles
print("=" * 60)
print("HIPÓDROMOS EN LA BASE DE DATOS:")
print("=" * 60)
df_hip = pd.read_sql_query("SELECT * FROM dim_hipodromos", conn)
print(df_hip)

# Ver carreras futuras
print("\n" + "=" * 60)
print("CARRERAS FUTURAS (sin resultado):")
print("=" * 60)
query = """
SELECT 
    fc.fecha,
    fc.nro_carrera,
    fc.hipodromo_id,
    dh.codigo,
    dh.nombre,
    COUNT(*) as participantes
FROM fact_carreras fc
LEFT JOIN dim_hipodromos dh ON fc.hipodromo_id = dh.id
JOIN fact_participaciones fp ON fp.carrera_id = fc.id
WHERE fp.resultado_final IS NULL
GROUP BY fc.fecha, fc.nro_carrera, fc.hipodromo_id
LIMIT 10
"""
df_carreras = pd.read_sql_query(query, conn)
print(df_carreras)

# Ver si hay carreras con hipodromo_id NULL
print("\n" + "=" * 60)
print("VERIFICANDO CARRERAS CON HIPODROMO_ID NULL:")
print("=" * 60)
query_null = """
SELECT COUNT(*) as count
FROM fact_carreras fc
JOIN fact_participaciones fp ON fp.carrera_id = fc.id
WHERE fp.resultado_final IS NULL AND fc.hipodromo_id IS NULL
"""
df_null = pd.read_sql_query(query_null, conn)
print(f"Carreras con hipodromo_id NULL: {df_null['count'].iloc[0]}")

conn.close()
