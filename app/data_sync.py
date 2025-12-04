import json
import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

# Add app directory to path to allow imports
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

try:
    from ml.predictor import run_full_pipeline as ml_pipeline, MLConfig
except ImportError:
    # Fallback for when running from root
    sys.path.append(str(BASE_DIR.parent))
    from app.ml.predictor import run_full_pipeline as ml_pipeline, MLConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = BASE_DIR.parent / "data" / "db" / "hipica_3fn.db"
OUTPUT_DIR = BASE_DIR / "output"

def get_db_connection():
    """Create a database connection."""
    if DB_PATH.exists():
        return sqlite3.connect(str(DB_PATH))
    return None

def load_predictions_json() -> Dict[str, Any]:
    """
    Carga el JSON de predicciones y lo adapta al formato esperado por la vista.
    Renombra campos y reestructura datos.
    """
    json_path = OUTPUT_DIR / "predicciones_detalle.json"
    
    if not json_path.exists():
        return {}
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 1. Adaptar Metadata
        if 'metadata' in data:
            if 'metricas_entrenamiento' in data['metadata']:
                data['metadata']['metricas'] = data['metadata'].pop('metricas_entrenamiento')
                
        # 2. Adaptar Predicciones
        if 'predicciones' in data:
            for pred in data['predicciones']:
                # Renombrar 'detalle' a 'predicciones' (esperado por la vista)
                if 'detalle' in pred:
                    pred['predicciones'] = pred.pop('detalle')
                    
                # Renombrar 'puntaje' a 'puntaje_calculado' en cada pick
                if 'predicciones' in pred:
                    for pick in pred['predicciones']:
                        if 'puntaje' in pick:
                            pick['puntaje_calculado'] = pick.pop('puntaje')
                            
        # 3. Adaptar Patrones
        if 'patrones' in data:
            new_patterns = {}
            for key, value in data['patrones'].items():
                if 'quinela' in key:
                    new_patterns['quinelas'] = value
                elif 'trifecta' in key:
                    new_patterns['trifectas'] = value
                elif 'superfecta' in key:
                    new_patterns['superfectas'] = value
                else:
                    new_patterns[key] = value
            data['patrones'] = new_patterns
            
        return data
        
    except Exception as e:
        logger.error(f"Error cargando JSON de predicciones: {e}")
        return {}

def run_full_pipeline() -> Tuple[bool, str]:
    """
    Ejecuta el pipeline ML completo.
    Retorna (success, logs).
    """
    try:
        # Capturar logs en memoria si es necesario, por ahora retornamos un string simple
        # En un entorno real, podríamos redirigir stdout/stderr
        
        config = MLConfig(db_path=str(DB_PATH), output_dir=str(OUTPUT_DIR))
        metrics, predictions = ml_pipeline(config)
        
        log_msg = f"Pipeline ejecutado exitosamente.\nCarreras predichas: {len(predictions)}\n"
        if metrics:
            log_msg += f"RMSE Test: {metrics.get('rmse_test', 0):.4f}"
            
        return True, log_msg
        
    except Exception as e:
        logger.error(f"Error ejecutando pipeline: {e}")
        return False, str(e)

def load_advanced_stats() -> Dict[str, Any]:
    """
    Carga estadísticas avanzadas desde la BD.
    """
    conn = get_db_connection()
    if not conn:
        return {}
        
    stats = {}
    
    try:
        # Top Jinetes
        query_jinetes = """
        SELECT 
            j.nombre,
            s.total_carreras,
            s.victorias,
            s.tasa_victoria * 100 as win_rate
        FROM agg_jinete_stats s
        JOIN dim_jinetes j ON s.jinete_id = j.id
        WHERE s.total_carreras >= 1
        ORDER BY s.tasa_victoria DESC
        LIMIT 10
        """
        stats['jinetes'] = pd.read_sql_query(query_jinetes, conn).to_dict('records')
        
        # Stats por Hipódromo y Distancia (para el gráfico)
        query_trends = """
        SELECT 
            h.codigo as hipodromo_codigo,
            c.distancia_metros,
            COUNT(*) as total_carreras,
            AVG(p.resultado_final) as avg_posicion
        FROM fact_carreras c
        JOIN dim_hipodromos h ON c.hipodromo_id = h.id
        JOIN fact_participaciones p ON c.id = p.carrera_id
        WHERE p.resultado_final IS NOT NULL
        GROUP BY h.codigo, c.distancia_metros
        HAVING COUNT(*) >= 1
        """
        stats['hipodromos_distancia'] = pd.read_sql_query(query_trends, conn).to_dict('records')
        
    except Exception as e:
        logger.error(f"Error cargando stats avanzadas: {e}")
        
    finally:
        conn.close()
        
    return stats

def get_db_stats() -> Dict[str, int]:
    """
    Retorna estadísticas básicas de la BD.
    """
    conn = get_db_connection()
    if not conn:
        return {"carreras": 0, "participantes": 0}
        
    try:
        carreras = pd.read_sql_query("SELECT COUNT(*) as c FROM fact_carreras", conn).iloc[0]['c']
        participantes = pd.read_sql_query("SELECT COUNT(*) as c FROM fact_participaciones", conn).iloc[0]['c']
        
        return {
            "carreras": carreras,
            "participantes": participantes
        }
    except:
        return {"carreras": 0, "participantes": 0}
    finally:
        if conn:
            conn.close()
