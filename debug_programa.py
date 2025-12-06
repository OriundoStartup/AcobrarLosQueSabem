import sqlite3
import pandas as pd
from pathlib import Path

# Conectar a BD
db_path = Path("data/db/hipica_3fn.db")
conn = sqlite3.connect(str(db_path))

# Verificar carreras del 05-12-2025 en CHC
print("=" * 80)
print("CARRERAS EN BD PARA 2025-12-05 - CHC")
print("=" * 80)

df_carreras = pd.read_sql_query("""
    SELECT c.fecha, h.codigo, c.nro_carrera, COUNT(p.id) as participantes
    FROM fact_carreras c
    JOIN dim_hipodromos h ON c.hipodromo_id = h.id
    LEFT JOIN fact_participaciones p ON c.id = p.carrera_id
    WHERE c.fecha = '2025-12-05' AND h.codigo IN ('CHC', 'CHS')
    GROUP BY c.id
    ORDER BY c.nro_carrera
""", conn)

print(f"\nTotal carreras encontradas: {len(df_carreras)}")
print(df_carreras.to_string())

# Verificar participantes de la carrera 1
print("\n" + "=" * 80)
print("PARTICIPANTES CARRERA 1 - 2025-12-05 - CHC")
print("=" * 80)

df_part = pd.read_sql_query("""
    SELECT 
        p.partidor,
        c_cab.nombre as caballo,
        j.nombre as jinete,
        p.peso_programado
    FROM fact_participaciones p
    JOIN fact_carreras c ON p.carrera_id = c.id
    JOIN dim_hipodromos h ON c.hipodromo_id = h.id
    JOIN dim_caballos c_cab ON p.caballo_id = c_cab.id
    LEFT JOIN dim_jinetes j ON p.jinete_id = j.id
    WHERE c.fecha = '2025-12-05' 
      AND h.codigo IN ('CHC', 'CHS')
      AND c.nro_carrera = 1
    ORDER BY p.partidor
""", conn)

print(f"\nTotal participantes carrera 1: {len(df_part)}")
print(df_part.to_string())

# Leer CSV original
print("\n" + "=" * 80)
print("CABALLOS EN CSV - CARRERA 1")
print("=" * 80)

csv_path = Path("exports/raw/PROGRAMA_CHC_2025-12-05.csv")
if csv_path.exists():
    df_csv = pd.read_csv(csv_path)
    df_csv_c1 = df_csv[df_csv['Carrera'] == 1]
    print(f"\nTotal en CSV carrera 1: {len(df_csv_c1)}")
    print(df_csv_c1[['Numero', 'Ejemplar', 'Jinete']].to_string())
else:
    print("CSV no encontrado")

conn.close()
