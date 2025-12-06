import sqlite3
import pandas as pd

conn = sqlite3.connect('data/db/hipica_3fn.db')

# Buscar nombres de caballos similares que podrían ser duplicados
query = """
SELECT 
    c1.id as id1,
    c1.nombre as nombre1,
    c2.id as id2,
    c2.nombre as nombre2,
    (SELECT COUNT(*) FROM fact_participaciones WHERE caballo_id = c1.id) as carreras1,
    (SELECT COUNT(*) FROM fact_participaciones WHERE caballo_id = c2.id) as carreras2
FROM dim_caballos c1
JOIN dim_caballos c2 ON UPPER(TRIM(c1.nombre)) = UPPER(TRIM(c2.nombre))
    AND c1.id < c2.id
ORDER BY c1.nombre
LIMIT 10
"""

df = pd.read_sql_query(query, conn)

print("=" * 80)
print("CABALLOS DUPLICADOS EN dim_caballos")
print("=" * 80)

if len(df) > 0:
    print(f"\n❌ Encontrados {len(df)} caballos con nombres duplicados:")
    print(df.to_string(index=False))
    print("\n¡ESTE ES EL PROBLEMA!")
    print("El ETL está creando múltiples registros en dim_caballos")
    print("para el mismo caballo cuando procesa RESUL")
else:
    print("\n✅ No hay caballos duplicados en dim_caballos")
    print("\nEntonces el problema debe ser diferente...")
    
    # Verificar si hay variaciones menores en nombres
    query2 = """
    SELECT nombre, COUNT(*) as veces
    FROM dim_caballos
    WHERE nombre LIKE '%PERFECT%' OR nombre LIKE '%FELIZ%'
    GROUP BY nombre
    """
    
    df2 = pd.read_sql_query(query2, conn)
    print("\nBuscando variaciones de nombres:")
    print(df2.to_string(index=False))

conn.close()
