import sqlite3
import pandas as pd

conn = sqlite3.connect('data/db/hipica_3fn.db')

# Query para encontrar casos específicos de duplicados
query = """
SELECT 
    c.fecha,
    c.nro_carrera,
    cab.nombre as caballo,
    p.caballo_id,
    p.resultado_final,
    p.id as participacion_id
FROM fact_participaciones p
JOIN fact_carreras c ON p.carrera_id = c.id
JOIN dim_caballos cab ON p.caballo_id = cab.id
WHERE c.fecha = '2025-11-17' AND c.nro_carrera = 1
ORDER BY cab.nombre, p.resultado_final
"""

df = pd.read_sql_query(query, conn)

print("=" * 80)
print("PARTICIPACIONES CARRERA 1 - 2025-11-17")
print("=" * 80)
print(f"\nTotal participaciones: {len(df)}")
print("\nDetalle:")
print(df[['caballo', 'caballo_id', 'resultado_final', 'participacion_id']].to_string(index=False))

# Buscar duplicados
duplicados = df.groupby('caballo_id').size()
duplicados = duplicados[duplicados > 1]

if len(duplicados) > 0:
    print(f"\n❌ Caballos con múltiples participaciones:")
    for cab_id, count in duplicados.items():
        print(f"   Caballo ID {cab_id}: {count} participaciones")
        df_cab = df[df['caballo_id'] == cab_id]
        print(f"      Nombre: {df_cab.iloc[0]['caballo']}")
        for _, row in df_cab.iterrows():
            print(f"      - Participación {row['participacion_id']}: resultado={row['resultado_final']}")
else:
    print("\n✅ No hay caballos duplicados (cada caballo aparece una vez)")

conn.close()
