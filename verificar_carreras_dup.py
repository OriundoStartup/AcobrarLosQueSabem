import sqlite3

conn = sqlite3.connect('data/db/hipica_3fn.db')
cursor = conn.cursor()

# Verificar si hay carreras duplicadas
cursor.execute('''
SELECT fecha, hipodromo_id, nro_carrera, COUNT(*) as veces
FROM fact_carreras
GROUP BY fecha, hipodromo_id, nro_carrera
HAVING COUNT(*) > 1
ORDER BY fecha DESC, nro_carrera
''')

duplicados = cursor.fetchall()

if duplicados:
    print(f"❌ Encontradas {len(duplicados)} carreras duplicadas:")
    for row in duplicados[:10]:
        print(f"   Fecha: {row[0]}, Hip: {row[1]}, Carrera: {row[2]}, Veces: {row[3]}")
else:
    print("✅ No hay carreras duplicadas en fact_carreras")
    print("\nEl problema NO es carreras duplicadas, es participaciones duplicadas.")

conn.close()
