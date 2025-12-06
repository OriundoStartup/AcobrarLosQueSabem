"""
Script de limpieza: Eliminar participaciones duplicadas sin resultado
para fechas que ya tienen resultados.

Este script limpia registros obsoletos que quedaron cuando se procesó
primero un PROGRAMA y luego un RESUL para la misma fecha.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

db_path = Path("data/db/hipica_3fn.db")

print("=" * 80)
print("LIMPIEZA DE PARTICIPACIONES DUPLICADAS")
print("=" * 80)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# 1. Identificar fechas con duplicados
query_duplicados = """
SELECT DISTINCT c.fecha
FROM fact_participaciones p
JOIN fact_carreras c ON p.carrera_id = c.id
WHERE p.resultado_final IS NULL
  AND EXISTS (
    SELECT 1 
    FROM fact_participaciones p2
    JOIN fact_carreras c2 ON p2.carrera_id = c2.id
    WHERE c2.fecha = c.fecha 
      AND c2.nro_carrera = c.nro_carrera
      AND p2.resultado_final IS NOT NULL
  )
ORDER BY c.fecha
"""

cursor.execute(query_duplicados)
fechas_duplicadas = [row[0] for row in cursor.fetchall()]

if not fechas_duplicadas:
    print("\n✅ No hay carreras duplicadas para limpiar")
    conn.close()
    exit(0)

print(f"\n⚠️  Encontradas {len(fechas_duplicadas)} fechas con duplicados:")
for fecha in fechas_duplicadas:
    print(f"   - {fecha}")

# 2. Contar registros a eliminar
query_count = """
SELECT COUNT(*)
FROM fact_participaciones p
JOIN fact_carreras c ON p.carrera_id = c.id
WHERE p.resultado_final IS NULL
  AND c.fecha IN ({})
  AND EXISTS (
    SELECT 1 
    FROM fact_participaciones p2
    JOIN fact_carreras c2 ON p2.carrera_id = c2.id
    WHERE c2.fecha = c.fecha 
      AND c2.nro_carrera = c.nro_carrera
      AND p2.resultado_final IS NOT NULL
  )
""".format(','.join(['?'] * len(fechas_duplicadas)))

cursor.execute(query_count, fechas_duplicadas)
total_a_eliminar = cursor.fetchone()[0]

print(f"\n📊 Se eliminarán {total_a_eliminar} participaciones sin resultado")
print(f"   (de carreras que ya tienen resultados)")

# Confirmar
respuesta = input("\n¿Deseas continuar con la limpieza? (si/no): ")

if respuesta.lower() not in ['si', 's', 'sí', 'yes', 'y']:
    print("\n❌ Operación cancelada")
    conn.close()
    exit(0)

# 3. Eliminar registros
print("\n🔄 Eliminando registros duplicados...")

# Primero, obtener los IDs de las participaciones a eliminar
query_ids = """
SELECT p.id
FROM fact_participaciones p
JOIN fact_carreras c ON p.carrera_id = c.id
WHERE p.resultado_final IS NULL
  AND c.fecha IN ({})
  AND EXISTS (
    SELECT 1 
    FROM fact_participaciones p2
    JOIN fact_carreras c2 ON p2.carrera_id = c2.id
    WHERE c2.fecha = c.fecha 
      AND c2.nro_carrera = c.nro_carrera
      AND p2.resultado_final IS NOT NULL
  )
""".format(','.join(['?'] * len(fechas_duplicadas)))

cursor.execute(query_ids, fechas_duplicadas)
ids_a_eliminar = [row[0] for row in cursor.fetchall()]

# Eliminar en lotes
batch_size = 100
total_eliminados = 0

for i in range(0, len(ids_a_eliminar), batch_size):
    batch = ids_a_eliminar[i:i+batch_size]
    placeholders = ','.join(['?'] * len(batch))
    cursor.execute(f"DELETE FROM fact_participaciones WHERE id IN ({placeholders})", batch)
    total_eliminados += cursor.rowcount
    print(f"   Eliminados: {total_eliminados}/{len(ids_a_eliminar)}")

# 4. Limpiar carreras huérfanas (sin participantes)
print("\n🔄 Limpiando carreras huérfanas...")

query_huerfanas = """
DELETE FROM fact_carreras
WHERE id NOT IN (SELECT DISTINCT carrera_id FROM fact_participaciones)
"""

cursor.execute(query_huerfanas)
carreras_eliminadas = cursor.rowcount

print(f"   Eliminadas {carreras_eliminadas} carreras sin participantes")

# 5. Commit
conn.commit()

print("\n✅ Limpieza completada exitosamente")
print(f"   Total participaciones eliminadas: {total_eliminados}")
print(f"   Total carreras huérfanas eliminadas: {carreras_eliminadas}")

# 6. Agregar constraint UNIQUE
print("\n🔧 Agregando constraint UNIQUE para prevenir duplicados futuros...")

try:
    cursor.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS idx_participacion_unica 
        ON fact_participaciones(carrera_id, caballo_id)
    ''')
    conn.commit()
    print("   ✅ Constraint UNIQUE agregado exitosamente")
    print("   La BD ahora rechazará automáticamente duplicados")
except Exception as e:
    print(f"   ⚠️  Error agregando constraint: {e}")

# 7. Verificación final
cursor.execute(query_duplicados)
duplicados_restantes = cursor.fetchall()

if not duplicados_restantes:
    print("\n✅ No quedan duplicados en la base de datos")
else:
    print(f"\n⚠️  Aún quedan {len(duplicados_restantes)} fechas con duplicados")

conn.close()

print("\n📝 RECOMENDACIÓN: Regenera las predicciones para actualizar el JSON")
print("   python -c \"from app.data_sync import run_predictions_only; run_predictions_only()\"")
