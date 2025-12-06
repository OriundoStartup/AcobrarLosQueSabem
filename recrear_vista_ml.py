"""
Script para recrear la vista v_ml_training_data con mejoras.
"""

import sqlite3
from pathlib import Path

db_path = Path("data/db/hipica_3fn.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print("=" * 80)
print("RECREANDO VISTA v_ml_training_data")
print("=" * 80)

# 1. Eliminar vista existente
print("\n1. Eliminando vista existente...")
try:
    cursor.execute("DROP VIEW IF EXISTS v_ml_training_data")
    print("   ✅ Vista eliminada")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# 2. Crear vista mejorada
print("\n2. Creando vista mejorada...")

vista_sql = """
CREATE VIEW v_ml_training_data AS
    SELECT
        -- IDs y metadatos
        fp.id AS participacion_id,
        fc.fecha,
        fc.nro_carrera,
        fc.hipodromo_id,
        fc.distancia_metros,
        COALESCE(fc.superficie_id, 1) AS superficie_id,

        -- Datos del participante
        fp.caballo_id,
        fp.jinete_id,
        fp.partidor,
        fp.peso_programado,
        fp.edad_anos,
        fp.handicap,

        -- Stats del caballo (de tabla de agregación)
        COALESCE(acs.total_carreras, 0) AS caballo_carreras_previas,
        COALESCE(acs.tasa_victoria, 0) AS caballo_tasa_victoria,
        COALESCE(acs.posicion_promedio, 5) AS caballo_pos_promedio,
        COALESCE(acs.dias_sin_correr, 30) AS caballo_dias_descanso,
        COALESCE(acs.racha_actual, 0) AS caballo_racha,

        -- Stats del jinete
        COALESCE(ajs.tasa_victoria, 0) AS jinete_tasa_victoria,
        COALESCE(ajs.posicion_promedio, 5) AS jinete_pos_promedio,

        -- Stats de combo caballo+jinete
        COALESCE(acj.tasa_victoria_combo, 0) AS combo_tasa_victoria,
        COALESCE(acj.carreras_juntos, 0) AS combo_carreras,

        -- Target (variable a predecir)
        fp.resultado_final AS target

    FROM fact_participaciones fp
    JOIN fact_carreras fc ON fp.carrera_id = fc.id
    LEFT JOIN agg_caballo_stats acs ON fp.caballo_id = acs.caballo_id
    LEFT JOIN agg_jinete_stats ajs ON fp.jinete_id = ajs.jinete_id
    LEFT JOIN agg_combo_caballo_jinete acj ON fp.caballo_id = acj.caballo_id 
        AND fp.jinete_id = acj.jinete_id
    WHERE fp.resultado_final IS NOT NULL  -- Solo datos con resultado (para entrenamiento)
"""

try:
    cursor.execute(vista_sql)
    conn.commit()
    print("   ✅ Vista creada exitosamente")
except Exception as e:
    print(f"   ❌ Error: {e}")
    conn.rollback()
    conn.close()
    exit(1)

# 3. Verificar vista
print("\n3. Verificando vista...")

try:
    cursor.execute("SELECT COUNT(*) FROM v_ml_training_data")
    count = cursor.fetchone()[0]
    print(f"   ✅ Vista funciona - {count} registros")
    
    # Verificar columnas
    cursor.execute("PRAGMA table_info(v_ml_training_data)")
    columnas = cursor.fetchall()
    print(f"\n   Columnas ({len(columnas)}):")
    for col in columnas:
        print(f"      - {col[1]}: {col[2] if col[2] else 'ANY'}")
    
    # Verificar sample de datos
    cursor.execute("SELECT * FROM v_ml_training_data LIMIT 1")
    row = cursor.fetchone()
    if row:
        print(f"\n   ✅ Sample de datos OK (target={row[-1]})")
    
except Exception as e:
    print(f"   ❌ Error al verificar: {e}")

conn.close()

print("\n" + "=" * 80)
print("✅ PROCESO COMPLETADO")
print("=" * 80)
print("\n📝 La vista v_ml_training_data ha sido recreada con mejoras:")
print("   - Agregado campo 'nro_carrera' para mejor contexto")
print("   - Agregado campo 'handicap' (estaba faltando)")
print("   - Mejorados comentarios SQL para claridad")
print("   - Mantiene filtro WHERE resultado_final IS NOT NULL")
