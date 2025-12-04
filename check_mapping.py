import sqlite3
import pandas as pd
import os

print("=" * 80)
print("ANÁLISIS DE MAPEO DE HIPÓDROMOS")
print("=" * 80)

# 1. Ver archivos CSV disponibles
print("\n1. ARCHIVOS CSV EN exports/raw:")
print("-" * 80)
csv_files = [f for f in os.listdir('exports/raw') if f.endswith('.csv')]
for f in sorted(csv_files):
    print(f"   {f}")

# 2. Ver configuración de hipódromos en BD
print("\n2. HIPÓDROMOS EN BASE DE DATOS:")
print("-" * 80)
conn = sqlite3.connect('data/db/hipica_3fn.db')
df_hip = pd.read_sql_query("SELECT * FROM dim_hipodromos", conn)
print(df_hip)

# 3. Ver qué datos hay en fact_carreras por hipódromo
print("\n3. CARRERAS POR HIPÓDROMO EN BD:")
print("-" * 80)
query = """
SELECT 
    dh.id,
    dh.codigo,
    dh.nombre,
    COUNT(DISTINCT fc.id) as total_carreras,
    MIN(fc.fecha) as fecha_min,
    MAX(fc.fecha) as fecha_max
FROM dim_hipodromos dh
LEFT JOIN fact_carreras fc ON fc.hipodromo_id = dh.id
GROUP BY dh.id, dh.codigo, dh.nombre
"""
df_stats = pd.read_sql_query(query, conn)
print(df_stats)

# 4. Ver ejemplos de carreras recientes por hipódromo
print("\n4. EJEMPLOS DE CARRERAS RECIENTES:")
print("-" * 80)
query = """
SELECT 
    fc.fecha,
    dh.codigo as hipodromo,
    fc.nro_carrera,
    COUNT(fp.id) as participantes
FROM fact_carreras fc
JOIN dim_hipodromos dh ON fc.hipodromo_id = dh.id
LEFT JOIN fact_participaciones fp ON fp.carrera_id = fc.id
GROUP BY fc.fecha, dh.codigo, fc.nro_carrera
ORDER BY fc.fecha DESC
LIMIT 20
"""
df_recent = pd.read_sql_query(query, conn)
print(df_recent)

# 5. Verificar si hay inconsistencias
print("\n5. VERIFICACIÓN DE NOMENCLATURA:")
print("-" * 80)
print("Según los archivos CSV:")
print("  - PROGRAMA_CHC_*.csv y resul_CHC_*.csv → Club Hípico (CHC)")
print("  - resul_HC_*.csv → Hipódromo Chile (HC)")
print("\nEn la base de datos tenemos:")
for _, row in df_hip.iterrows():
    print(f"  - ID {row['id']}: código '{row['codigo']}', nombre '{row['nombre']}'")

print("\n⚠️ PROBLEMA DETECTADO:")
print("  Los códigos en BD son 'CHC' y 'HS', pero los archivos usan 'CHC' y 'HC'")
print("  'HS' debería ser 'HC' para Hipódromo Chile")

conn.close()
