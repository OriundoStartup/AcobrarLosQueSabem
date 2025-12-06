import sqlite3
import pandas as pd

conn = sqlite3.connect('data/db/hipica_3fn.db')

# Verificar si los nombres de caballos son idénticos entre duplicados
query = """
SELECT 
    c.fecha,
    c.nro_carrera,
    p1.id as part1_id,
    p2.id as part2_id,
    cab.nombre as caballo,
    p1.resultado_final as resultado1,
    p2.resultado_final as resultado2,
    p1.partidor as partidor1,
    p2.partidor as partidor2
FROM fact_participaciones p1
JOIN fact_participaciones p2 ON p1.carrera_id = p2.carrera_id 
    AND p1.caballo_id = p2.caballo_id 
    AND p1.id < p2.id
JOIN fact_carreras c ON p1.carrera_id = c.id
JOIN dim_caballos cab ON p1.caballo_id = cab.id
WHERE c.fecha = '2025-11-17' AND c.nro_carrera = 1
ORDER BY cab.nombre
LIMIT 5
"""

df = pd.read_sql_query(query, conn)

print("=" * 80)
print("ANÁLISIS DE DUPLICADOS - Mismo caballo_id, misma carrera")
print("=" * 80)
print("\nEjemplos de duplicados:")
print(df.to_string(index=False))

if len(df) > 0:
    print("\n✅ CONFIRMADO: Duplicados son el MISMO caballo (mismo caballo_id)")
    print("   El problema NO es normalización de nombres")
    print("   El problema es que _upsert_participacion NO está encontrando")
    print("   la participación existente cuando procesa el RESUL")
    
    # Verificar si los partidores son iguales
    if (df['partidor1'] == df['partidor2']).all():
        print("\n✅ Los partidores SON iguales")
        print("   Entonces _upsert_participacion DEBERÍA encontrar la participación")
    else:
        print("\n❌ Los partidores son DIFERENTES")
        print("   Puede ser que el RESUL no tenga información del partidor")

conn.close()
