import sqlite3

conn = sqlite3.connect('data/db/hipica_3fn.db')
cursor = conn.cursor()

# Teoría: Las participaciones duplicadas están en CARRERAS DIFERENTES (mismo día, mismo nro)
# porque cuando se procesa el RESUL, el ETL crea una nueva carrera en vez de encontrar la existente

query = """
SELECT 
    c1.id as carrera1_id,
    c2.id as carrera2_id,
    c1.fecha,
    h.codigo as hip,
    c1.nro_carrera,
    (SELECT COUNT(*) FROM fact_participaciones WHERE carrera_id = c1.id) as part_en_c1,
    (SELECT COUNT(*) FROM fact_participaciones WHERE carrera_id = c2.id) as part_en_c2
FROM fact_carreras c1
JOIN fact_carreras c2 ON c1.fecha = c2.fecha 
    AND c1.hipodromo_id = c2.hipodromo_id 
    AND c1.nro_carrera = c2.nro_carrera
    AND c1.id < c2.id
JOIN dim_hipodromos h ON c1.hipodromo_id = h.id
WHERE c1.fecha = '2025-11-17'
ORDER BY c1.nro_carrera
"""

cursor.execute(query)
duplicados_carreras = cursor.fetchall()

print("=" * 80)
print("INVESTIGACIÓN: ¿Hay carreras duplicadas?")
print("=" * 80)

if duplicados_carreras:
    print(f"\n❌ ENCONTRADO EL PROBLEMA! Hay {len(duplicados_carreras)} carreras duplicadas:")
    print("\n| Carrera 1 ID | Carrera 2 ID | Fecha | Hip | Nro | Part C1 | Part C2 |")
    print("|" + "-" * 75 + "|")
    for row in duplicados_carreras[:10]:
        print(f"| {row[0]:12d} | {row[1]:12d} | {row[2]} | {row[3]:>3s} | {row[4]:3d} | {row[5]:7d} | {row[6]:7d} |")
    
    print("\n🔍 Explicación:")
    print("   1. Se procesó PROGRAMA → creó carrera con ID X")
    print("   2. Se procesó RESUL → creó NUEVA carrera con ID Y (en vez de usar X)")
    print("   3. Cada carrera tiene su propio conjunto de participaciones")
    print("   4. Por eso _upsert_participacion no encuentra match (busca en carrera diferente)")
    
    print("\n✅ SOLUCIÓN:")
    print("   _upsert_carrera debe buscar más agresivamente antes de crear nueva")
else:
    print("\n✅ No hay carreras duplicadas")
    print("\nEntonces el problema es otro... vamos a investigar más")

conn.close()
