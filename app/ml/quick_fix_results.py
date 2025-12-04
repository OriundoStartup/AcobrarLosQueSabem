"""
Script rápido para verificar y arreglar resultado_final
"""

import sqlite3
import re

DB_PATH = "data/db/hipica_3fn.db"

def check_and_fix():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("="*60)
    print("🔍 VERIFICANDO RESULTADOS")
    print("="*60)
    
    # Ver qué hay en las participaciones recién importadas
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN resultado_final IS NOT NULL THEN 1 ELSE 0 END) as con_resultado
        FROM fact_participaciones
        WHERE source_file LIKE '%resul%'
    """)
    
    row = cursor.fetchone()
    print(f"\n📊 Participaciones de archivos de RESULTADOS:")
    print(f"   Total: {row[0]}")
    print(f"   Con resultado_final: {row[1]}")
    
    if row[1] == 0 and row[0] > 0:
        print("\n⚠️ Hay participaciones de resultados pero sin resultado_final")
        print("   Esto significa que 'posicion' no se mapeó a 'resultado_final'")
        
        # Ver una muestra
        cursor.execute("""
            SELECT id, caballo_id, source_file
            FROM fact_participaciones
            WHERE source_file LIKE '%resul%'
            LIMIT 5
        """)
        
        print("\n   Muestra de participaciones:")
        for row in cursor.fetchall():
            print(f"      ID: {row[0]}, Caballo: {row[1]}, File: {row[2]}")
        
        # Intentar extraer posición del nombre del archivo o buscar otra columna
        print("\n💡 SOLUCIÓN: Necesitamos actualizar el importador para mapear posicion → resultado_final")
        
    else:
        print("\n✅ Los resultados están correctamente mapeados!")
        
        # Calcular agregaciones
        print("\n⚙️ Calculando agregaciones...")
        
        # Stats caballos
        cursor.execute("""
            INSERT OR REPLACE INTO agg_caballo_stats (
                caballo_id, total_carreras, victorias, segundo_lugar, tercer_lugar,
                tasa_victoria, posicion_promedio, dias_sin_correr, racha_actual,
                fecha_actualizacion
            )
            SELECT 
                fp.caballo_id,
                COUNT(*) as total_carreras,
                SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) as victorias,
                SUM(CASE WHEN fp.resultado_final = 2 THEN 1 ELSE 0 END) as segundo_lugar,
                SUM(CASE WHEN fp.resultado_final = 3 THEN 1 ELSE 0 END) as tercer_lugar,
                CAST(SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as tasa_victoria,
                AVG(CAST(fp.resultado_final AS FLOAT)) as posicion_promedio,
                0 as dias_sin_correr,
                0 as racha_actual,
                CURRENT_TIMESTAMP
            FROM fact_participaciones fp
            WHERE fp.resultado_final IS NOT NULL
            GROUP BY fp.caballo_id
        """)
        
        count = cursor.rowcount
        conn.commit()
        print(f"   ✅ {count} caballos con stats")
        
        # Stats jinetes
        cursor.execute("""
            INSERT OR REPLACE INTO agg_jinete_stats (
                jinete_id, total_carreras, victorias,
                tasa_victoria, posicion_promedio, fecha_actualizacion
            )
            SELECT 
                fp.jinete_id,
                COUNT(*) as total_carreras,
                SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) as victorias,
                CAST(SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as tasa_victoria,
                AVG(CAST(fp.resultado_final AS FLOAT)) as posicion_promedio,
                CURRENT_TIMESTAMP
            FROM fact_participaciones fp
            WHERE fp.resultado_final IS NOT NULL AND fp.jinete_id IS NOT NULL
            GROUP BY fp.jinete_id
        """)
        
        count = cursor.rowcount
        conn.commit()
        print(f"   ✅ {count} jinetes con stats")
        
        # Verificar vista ML
        cursor.execute("SELECT COUNT(*) FROM v_ml_training_data")
        ml_count = cursor.fetchone()[0]
        
        print(f"\n📊 Registros en v_ml_training_data: {ml_count}")
        
        if ml_count > 0:
            print("\n✅ ¡LISTO PARA ENTRENAR ML!")
            print("\n   Ejecuta: python app/ml/predictor.py")
        else:
            print("\n⚠️ La vista ML sigue vacía")
            print("   Verifica que la vista incluya las agregaciones")
    
    conn.close()

if __name__ == "__main__":
    check_and_fix()