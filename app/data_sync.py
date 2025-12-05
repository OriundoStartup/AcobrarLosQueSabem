"""
================================================================================
PISTA INTELIGENTE - UTILS (SERVICE LAYER)
================================================================================
Módulo: data_sync.py

Capa de servicios para la interfaz Streamlit.
Integra con el orchestrator y proporciona funciones adaptadas para la UI.

Autor: Pista Inteligente Team
================================================================================
"""

import json
import pandas as pd
import sqlite3
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from contextlib import contextmanager
import sys

# Setup paths para la estructura del proyecto
BASE_DIR = Path(__file__).parent      # app/
ROOT_DIR = BASE_DIR.parent            # raíz del proyecto
ML_DIR = BASE_DIR / "ml"              # app/ml/
ETL_DIR = BASE_DIR / "etl"            # app/etl/

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(ETL_DIR))

# Importar módulos del sistema
from orchestrator import PistaInteligenteOrchestrator, OrchestratorConfig
from predictor import RacePredictor, MLConfig, PredictionResult
from etl_pipeline import ETLPipeline, ETLConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths por defecto
DB_PATH = ROOT_DIR / "data" / "db" / "hipica_3fn.db"
OUTPUT_DIR = ML_DIR / "output"
MODEL_PATH = ROOT_DIR / "models" / "predictor_latest.joblib"


# ==============================================================================
# CONEXIÓN A BASE DE DATOS
# ==============================================================================

@contextmanager
def get_db_connection():
    """Context manager para conexión a BD."""
    conn = None
    try:
        if DB_PATH.exists():
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            yield conn
        else:
            yield None
    finally:
        if conn:
            conn.close()


def get_db_stats() -> Dict[str, int]:
    """Retorna estadísticas básicas de la BD."""
    with get_db_connection() as conn:
        if not conn:
            return {"carreras": 0, "participantes": 0, "caballos": 0, "jinetes": 0}
        
        try:
            stats = {}
            queries = {
                "carreras": "SELECT COUNT(*) FROM fact_carreras",
                "participantes": "SELECT COUNT(*) FROM fact_participaciones",
                "caballos": "SELECT COUNT(*) FROM dim_caballos",
                "jinetes": "SELECT COUNT(*) FROM dim_jinetes"
            }
            
            for key, query in queries.items():
                result = pd.read_sql_query(query, conn)
                stats[key] = int(result.iloc[0, 0])
            
            return stats
        except Exception as e:
            logger.error(f"Error obteniendo stats de BD: {e}")
            return {"carreras": 0, "participantes": 0, "caballos": 0, "jinetes": 0}


# ==============================================================================
# PIPELINE INTEGRADO
# ==============================================================================

def run_full_pipeline(
    csv_path: Optional[str] = None,
    force_retrain: bool = False
) -> Tuple[bool, str, Optional[Dict]]:
    """
    Ejecuta el pipeline completo usando el orchestrator.
    
    Args:
        csv_path: Ruta opcional a CSV con datos nuevos
        force_retrain: Forzar reentrenamiento del modelo
        
    Returns:
        Tuple (success, log_message, results_dict)
    """
    try:
        config = OrchestratorConfig(
            db_path=str(DB_PATH),
            model_path=str(MODEL_PATH),
            output_dir=str(OUTPUT_DIR)
        )
        
        with PistaInteligenteOrchestrator(config) as orchestrator:
            results = orchestrator.run_full_pipeline(
                csv_path=csv_path,
                force_retrain=force_retrain
            )
        
        # Construir mensaje de log
        log_parts = ["✅ Pipeline ejecutado exitosamente"]
        
        if results.get('etl'):
            etl = results['etl']
            log_parts.append(f"📊 ETL: {etl.get('records_inserted', 0)} insertados")
        
        if results.get('training') and results['training'].get('metrics'):
            metrics = results['training']['metrics']
            log_parts.append(f"🤖 R² Test: {metrics.get('r2_test', 0):.4f}")
        
        if results.get('predictions'):
            pred = results['predictions']
            log_parts.append(f"🔮 Predicciones: {pred.get('count', 0)} carreras")
        
        log_msg = "\n".join(log_parts)
        return True, log_msg, results
        
    except Exception as e:
        logger.exception(f"Error ejecutando pipeline: {e}")
        return False, f"❌ Error: {str(e)}", None


def run_etl_only(csv_path: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Ejecuta solo el ETL (sin entrenamiento ni predicción).
    
    Args:
        csv_path: Ruta al CSV
        
    Returns:
        Tuple (success, log_message, results_dict)
    """
    try:
        config = ETLConfig(
            db_path=str(DB_PATH),
            update_aggregations=True
        )
        
        with ETLPipeline(config) as pipeline:
            result = pipeline.process_csv(csv_path)
        
        log_msg = (
            f"✅ ETL completado\n"
            f"📄 Archivo: {Path(csv_path).name}\n"
            f"📊 Registros: {result.records_raw}\n"
            f"✅ Insertados: {result.records_inserted}\n"
            f"🔄 Actualizados: {result.records_updated}\n"
            f"❌ Rechazados: {result.records_rejected}"
        )
        
        if result.detection_info:
            log_msg += f"\n🔍 Detectado: {result.detection_info.get('csv_type', 'N/A')}"
            log_msg += f" | Hipódromo: {result.detection_info.get('hipodromo', 'N/A')}"
        
        return True, log_msg, {
            'batch_id': result.batch_id,
            'records_raw': result.records_raw,
            'records_inserted': result.records_inserted,
            'records_updated': result.records_updated,
            'records_rejected': result.records_rejected,
            'detection': result.detection_info
        }
        
    except Exception as e:
        logger.exception(f"Error en ETL: {e}")
        return False, f"❌ Error: {str(e)}", None


def run_predictions_only() -> Tuple[bool, str, Optional[List]]:
    """
    Ejecuta solo predicciones con modelo existente.
    
    Returns:
        Tuple (success, log_message, predictions_list)
    """
    try:
        config = OrchestratorConfig(
            db_path=str(DB_PATH),
            model_path=str(MODEL_PATH),
            output_dir=str(OUTPUT_DIR)
        )
        
        with PistaInteligenteOrchestrator(config) as orchestrator:
            predictions = orchestrator.predict_only()
        
        if predictions:
            log_msg = f"✅ {len(predictions)} carreras predichas"
            return True, log_msg, predictions
        else:
            return True, "⚠️ No hay carreras futuras para predecir", []
            
    except FileNotFoundError as e:
        return False, f"⚠️ Modelo no encontrado. Ejecute entrenamiento primero.", None
    except Exception as e:
        logger.exception(f"Error en predicción: {e}")
        return False, f"❌ Error: {str(e)}", None


# ==============================================================================
# CARGA DE PREDICCIONES PARA UI
# ==============================================================================

def load_predictions_json() -> Dict[str, Any]:
    """
    Carga el JSON de predicciones y lo adapta al formato esperado por la vista.
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


def load_predictions_csv() -> Optional[pd.DataFrame]:
    """Carga el CSV de predicciones como DataFrame."""
    csv_path = OUTPUT_DIR / "predicciones.csv"
    
    if not csv_path.exists():
        return None
    
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error cargando CSV de predicciones: {e}")
        return None


# ==============================================================================
# ESTADÍSTICAS AVANZADAS PARA UI
# ==============================================================================

def load_advanced_stats() -> Dict[str, Any]:
    """Carga estadísticas avanzadas desde la BD."""
    with get_db_connection() as conn:
        if not conn:
            return {}
        
        stats = {}
        
        try:
            # Top 10 Jinetes por tasa de victoria
            query_jinetes = """
            SELECT 
                j.nombre,
                s.total_carreras,
                s.victorias,
                ROUND(s.tasa_victoria * 100, 1) as win_rate
            FROM agg_jinete_stats s
            JOIN dim_jinetes j ON s.jinete_id = j.id
            WHERE s.total_carreras >= 5
            ORDER BY s.tasa_victoria DESC
            LIMIT 10
            """
            stats['top_jinetes'] = pd.read_sql_query(query_jinetes, conn).to_dict('records')
            
            # Top 10 Caballos por tasa de victoria
            query_caballos = """
            SELECT 
                c.nombre,
                s.total_carreras,
                s.victorias,
                ROUND(s.tasa_victoria * 100, 1) as win_rate,
                s.dias_sin_correr
            FROM agg_caballo_stats s
            JOIN dim_caballos c ON s.caballo_id = c.id
            WHERE s.total_carreras >= 3
            ORDER BY s.tasa_victoria DESC
            LIMIT 10
            """
            stats['top_caballos'] = pd.read_sql_query(query_caballos, conn).to_dict('records')
            
            # Stats por Hipódromo
            query_hipodromos = """
            SELECT 
                h.nombre as hipodromo,
                h.codigo,
                COUNT(DISTINCT c.id) as total_carreras,
                COUNT(p.id) as total_participaciones
            FROM fact_carreras c
            JOIN dim_hipodromos h ON c.hipodromo_id = h.id
            LEFT JOIN fact_participaciones p ON c.id = p.carrera_id
            GROUP BY h.id
            ORDER BY total_carreras DESC
            """
            stats['hipodromos'] = pd.read_sql_query(query_hipodromos, conn).to_dict('records')
            
            # Distribución por distancia
            query_distancias = """
            SELECT 
                CASE 
                    WHEN distancia_metros <= 1100 THEN 'Sprint (≤1100m)'
                    WHEN distancia_metros <= 1400 THEN 'Milla Corta (1100-1400m)'
                    WHEN distancia_metros <= 1700 THEN 'Milla (1400-1700m)'
                    ELSE 'Fondo (>1700m)'
                END as categoria,
                COUNT(*) as total_carreras
            FROM fact_carreras
            WHERE distancia_metros IS NOT NULL
            GROUP BY categoria
            ORDER BY MIN(distancia_metros)
            """
            stats['distancias'] = pd.read_sql_query(query_distancias, conn).to_dict('records')
            
            # Carreras recientes
            query_recientes = """
            SELECT 
                c.fecha,
                h.codigo as hipodromo,
                c.nro_carrera,
                c.distancia_metros,
                COUNT(p.id) as participantes
            FROM fact_carreras c
            JOIN dim_hipodromos h ON c.hipodromo_id = h.id
            LEFT JOIN fact_participaciones p ON c.id = p.carrera_id
            GROUP BY c.id
            ORDER BY c.fecha DESC, c.nro_carrera
            LIMIT 20
            """
            stats['carreras_recientes'] = pd.read_sql_query(query_recientes, conn).to_dict('records')
            
        except Exception as e:
            logger.error(f"Error cargando stats avanzadas: {e}")
        
        return stats


def get_model_info() -> Optional[Dict[str, Any]]:
    """Obtiene información del modelo guardado."""
    if not MODEL_PATH.exists():
        return None
    
    try:
        import joblib
        model_data = joblib.load(str(MODEL_PATH))
        
        return {
            'version': model_data.get('version', 'N/A'),
            'trained_at': model_data.get('trained_at', 'N/A'),
            'metrics': model_data.get('metrics', {}),
            'n_features': len(model_data.get('feature_columns', []))
        }
    except Exception as e:
        logger.error(f"Error cargando info del modelo: {e}")
        return None


def get_carreras_pendientes() -> List[Dict]:
    """Obtiene carreras sin resultado (pendientes de predicción)."""
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            query = """
            SELECT 
                c.fecha,
                h.codigo as hipodromo,
                c.nro_carrera,
                c.distancia_metros,
                c.hora_programada,
                COUNT(p.id) as participantes
            FROM fact_carreras c
            JOIN dim_hipodromos h ON c.hipodromo_id = h.id
            LEFT JOIN fact_participaciones p ON c.id = p.carrera_id
            WHERE NOT EXISTS (
                SELECT 1 FROM fact_participaciones p2 
                WHERE p2.carrera_id = c.id AND p2.resultado_final IS NOT NULL
            )
            GROUP BY c.id
            ORDER BY c.fecha, c.nro_carrera
            """
            return pd.read_sql_query(query, conn).to_dict('records')
        except Exception as e:
            logger.error(f"Error obteniendo carreras pendientes: {e}")
            return []