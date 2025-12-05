"""
================================================================================
PISTA INTELIGENTE - ORQUESTADOR PRINCIPAL
================================================================================
Integra ETL Pipeline y Predictor en un flujo unificado.

Flujo completo:
1. ETL: Cargar/actualizar datos desde CSV o BD legacy
2. ML: Entrenar modelo con datos actualizados
3. Predict: Generar predicciones para carreras futuras
4. Export: Guardar resultados

Uso:
    # Flujo completo desde CSV
    python orchestrator.py --csv data/carreras_nuevas.csv
    
    # Migración inicial + entrenamiento
    python orchestrator.py --migrate data/db/legacy.db
    
    # Solo predicción (modelo existente)
    python orchestrator.py --predict-only
    
    # Reentrenar modelo
    python orchestrator.py --retrain

Autor: Pista Inteligente Team
================================================================================
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

# Setup paths para la estructura del proyecto
BASE_DIR = Path(__file__).parent  # app/ml/
APP_DIR = BASE_DIR.parent         # app/
ROOT_DIR = APP_DIR.parent         # raíz del proyecto
ETL_DIR = APP_DIR / "etl"         # app/etl/

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ETL_DIR))
sys.path.insert(0, str(BASE_DIR))

# Importar módulos del sistema
from etl_pipeline import ETLPipeline, ETLConfig, ETLBatchResult, SourceType, migrate_legacy_to_3fn
from predictor import RacePredictor, MLConfig, PredictionResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

@dataclass
class OrchestratorConfig:
    """Configuración del orquestador."""
    # Paths
    db_path: str = "data/db/hipica_3fn.db"
    model_path: str = "models/predictor_latest.joblib"
    output_dir: str = "app/ml/output" 
    
    # ETL
    update_aggregations: bool = True
    
    # ML
    model_type: str = "xgboost"
    retrain_threshold_days: int = 7  # Reentrenar si modelo tiene más de N días
    min_samples_train: int = 100
    
    # Predicción
    auto_predict: bool = True


# ==============================================================================
# ORQUESTADOR PRINCIPAL
# ==============================================================================

class PistaInteligenteOrchestrator:
    """
    Orquestador que coordina ETL y ML en un flujo unificado.
    """
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self._etl_pipeline: Optional[ETLPipeline] = None
        self._predictor: Optional[RacePredictor] = None
    
    @property
    def etl(self) -> ETLPipeline:
        """Lazy loading del ETL pipeline."""
        if self._etl_pipeline is None:
            etl_config = ETLConfig(
                db_path=self.config.db_path,
                update_aggregations=self.config.update_aggregations
            )
            self._etl_pipeline = ETLPipeline(etl_config)
        return self._etl_pipeline
    
    @property
    def predictor(self) -> RacePredictor:
        """Lazy loading del predictor."""
        if self._predictor is None:
            ml_config = MLConfig(
                db_path=self.config.db_path,
                model_type=self.config.model_type,
                output_dir=self.config.output_dir
            )
            self._predictor = RacePredictor(ml_config)
        return self._predictor
    
    def close(self):
        """Cierra conexiones."""
        if self._etl_pipeline:
            self._etl_pipeline.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ==========================================================================
    # FLUJOS PRINCIPALES
    # ==========================================================================
    
    def run_full_pipeline(
        self,
        csv_path: Optional[str] = None,
        legacy_db_path: Optional[str] = None,
        force_retrain: bool = False
    ) -> Dict:
        """
        Ejecuta el pipeline completo: ETL → Train → Predict.
        
        Args:
            csv_path: Ruta a CSV con nuevos datos
            legacy_db_path: Ruta a BD legacy para migración inicial
            force_retrain: Forzar reentrenamiento aunque exista modelo
            
        Returns:
            Dict con resultados de cada etapa
        """
        self._print_header("PISTA INTELIGENTE - PIPELINE COMPLETO")
        
        results = {
            'etl': None,
            'training': None,
            'predictions': None,
            'status': 'started',
            'started_at': datetime.now().isoformat()
        }
        
        try:
            # ETAPA 1: ETL
            if legacy_db_path:
                results['etl'] = self.run_migration(legacy_db_path)
            elif csv_path:
                results['etl'] = self.run_etl(csv_path)
            else:
                logger.info("⏭️ Sin datos nuevos para cargar, usando BD existente")
            
            # ETAPA 2: Entrenamiento
            need_training = force_retrain or not self._model_exists()
            
            if need_training:
                results['training'] = self.run_training()
            else:
                logger.info(f"⏭️ Usando modelo existente: {self.config.model_path}")
                self._predictor = RacePredictor.load_model(self.config.model_path)
                results['training'] = {'status': 'skipped', 'model': self.config.model_path}
            
            # ETAPA 3: Predicción
            if self.config.auto_predict:
                results['predictions'] = self.run_predictions()
            
            results['status'] = 'completed'
            results['completed_at'] = datetime.now().isoformat()
            
            self._print_summary(results)
            return results
            
        except Exception as e:
            logger.exception(f"❌ Error en pipeline: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
            raise
    
    def run_migration(self, legacy_db_path: str) -> Dict:
        """
        Ejecuta migración desde BD legacy.
        """
        self._print_header("ETAPA 1: MIGRACIÓN DESDE BD LEGACY")
        
        logger.info(f"📂 Origen: {legacy_db_path}")
        logger.info(f"📂 Destino: {self.config.db_path}")
        
        # Usar función de migración que crea esquema + migra datos
        result = migrate_legacy_to_3fn(
            legacy_db_path=legacy_db_path,
            new_db_path=self.config.db_path
        )
        
        logger.info(f"✅ Migración completada:")
        logger.info(f"   - Registros procesados: {result.records_raw}")
        logger.info(f"   - Insertados: {result.records_inserted}")
        logger.info(f"   - Rechazados: {result.records_rejected}")
        
        return {
            'batch_id': result.batch_id,
            'records_raw': result.records_raw,
            'records_inserted': result.records_inserted,
            'records_rejected': result.records_rejected,
            'status': result.status
        }
    
    def run_etl(self, csv_path: str) -> Dict:
        """
        Ejecuta ETL desde archivo CSV.
        """
        self._print_header("ETAPA 1: ETL - CARGA DE DATOS")
        
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV no encontrado: {csv_path}")
        
        logger.info(f"📂 Procesando: {csv_path}")
        
        # Usar auto-detección (source_type=None)
        result = self.etl.process_csv(csv_path)
        
        logger.info(f"✅ ETL completado:")
        logger.info(f"   - Registros procesados: {result.records_raw}")
        logger.info(f"   - Insertados: {result.records_inserted}")
        logger.info(f"   - Actualizados: {result.records_updated}")
        logger.info(f"   - Rechazados: {result.records_rejected}")
        
        return {
            'batch_id': result.batch_id,
            'records_raw': result.records_raw,
            'records_inserted': result.records_inserted,
            'records_updated': result.records_updated,
            'records_rejected': result.records_rejected,
            'status': result.status
        }
    
    def run_training(self) -> Dict:
        """
        Ejecuta entrenamiento del modelo ML.
        """
        self._print_header("ETAPA 2: ENTRENAMIENTO ML")
        
        # Entrenar
        metrics = self.predictor.train()
        
        # Guardar modelo
        Path(self.config.model_path).parent.mkdir(parents=True, exist_ok=True)
        self.predictor.save_model(self.config.model_path)
        
        logger.info(f"✅ Modelo entrenado y guardado:")
        logger.info(f"   - R² Test: {metrics['r2_test']:.4f}")
        logger.info(f"   - RMSE Test: {metrics['rmse_test']:.4f}")
        logger.info(f"   - Samples: {metrics['n_samples_train']} train, {metrics['n_samples_test']} test")
        logger.info(f"   - Guardado en: {self.config.model_path}")
        
        return {
            'metrics': metrics,
            'model_path': self.config.model_path,
            'status': 'completed'
        }
    
    def run_predictions(self) -> Dict:
        """
        Ejecuta predicciones para carreras futuras.
        """
        self._print_header("ETAPA 3: PREDICCIONES")
        
        # Generar predicciones
        predictions = self.predictor.predict()
        
        if not predictions:
            logger.warning("⚠️ No hay carreras futuras para predecir")
            return {'count': 0, 'status': 'no_data'}
        
        # Detectar patrones
        patterns = self.predictor.detect_patterns(predictions)
        
        # Guardar resultados
        csv_path, json_path = self.predictor.save_predictions(predictions, patterns)
        
        logger.info(f"✅ Predicciones generadas:")
        logger.info(f"   - Carreras predichas: {len(predictions)}")
        logger.info(f"   - CSV: {csv_path}")
        logger.info(f"   - JSON: {json_path}")
        
        return {
            'count': len(predictions),
            'carreras': [
                {
                    'fecha': p.fecha,
                    'hipodromo': p.hipodromo,
                    'nro_carrera': p.nro_carrera,
                    'top4': p.top4_predicho,
                    'confianza': p.confianza
                }
                for p in predictions
            ],
            'patterns': patterns,
            'output_csv': csv_path,
            'output_json': json_path,
            'status': 'completed'
        }
    
    def predict_only(self) -> List[PredictionResult]:
        """
        Solo genera predicciones usando modelo existente.
        """
        self._print_header("PREDICCIÓN (MODELO EXISTENTE)")
        
        if not self._model_exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {self.config.model_path}\n"
                f"Ejecute primero con --retrain"
            )
        
        # Cargar modelo
        self._predictor = RacePredictor.load_model(self.config.model_path)
        
        # Predecir
        predictions = self.predictor.predict()
        
        if predictions:
            patterns = self.predictor.detect_patterns(predictions)
            self.predictor.save_predictions(predictions, patterns)
        
        return predictions
    
    # ==========================================================================
    # UTILIDADES
    # ==========================================================================
    
    def _model_exists(self) -> bool:
        """Verifica si existe un modelo guardado."""
        return Path(self.config.model_path).exists()
    
    def _print_header(self, title: str):
        """Imprime header de sección."""
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60 + "\n")
    
    def _print_summary(self, results: Dict):
        """Imprime resumen final."""
        print("\n" + "╔" + "═"*58 + "╗")
        print("║" + " "*20 + "RESUMEN FINAL" + " "*20 + "║")
        print("╠" + "═"*58 + "╣")
        
        # ETL
        if results.get('etl'):
            etl = results['etl']
            print(f"║  ETL: {etl.get('records_inserted', 0)} insertados, "
                  f"{etl.get('records_rejected', 0)} rechazados" + " "*10 + "║")
        
        # Training
        if results.get('training'):
            train = results['training']
            if train.get('metrics'):
                r2 = train['metrics'].get('r2_test', 0)
                print(f"║  ML:  R² = {r2:.4f}" + " "*35 + "║")
            else:
                print(f"║  ML:  Modelo existente" + " "*30 + "║")
        
        # Predictions
        if results.get('predictions'):
            pred = results['predictions']
            print(f"║  Predicciones: {pred.get('count', 0)} carreras" + " "*30 + "║")
        
        print("╠" + "═"*58 + "╣")
        print(f"║  Status: {results.get('status', 'unknown').upper()}" + " "*40 + "║")
        print("╚" + "═"*58 + "╝\n")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pista Inteligente - Orquestador de Pipeline"
    )
    
    parser.add_argument(
        '--csv', 
        type=str, 
        help='Ruta a CSV con datos nuevos'
    )
    parser.add_argument(
        '--migrate', 
        type=str, 
        help='Ruta a BD legacy para migración inicial'
    )
    parser.add_argument(
        '--retrain', 
        action='store_true',
        help='Forzar reentrenamiento del modelo'
    )
    parser.add_argument(
        '--predict-only', 
        action='store_true',
        help='Solo generar predicciones (requiere modelo existente)'
    )
    parser.add_argument(
        '--db', 
        type=str, 
        default='data/db/hipica_3fn.db',
        help='Ruta a la base de datos'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='models/predictor_latest.joblib',
        help='Ruta al archivo del modelo'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='output',
        help='Directorio de salida'
    )
    
    args = parser.parse_args()
    
    # Configurar
    config = OrchestratorConfig(
        db_path=args.db,
        model_path=args.model,
        output_dir=args.output
    )
    
    # Ejecutar
    with PistaInteligenteOrchestrator(config) as orchestrator:
        if args.predict_only:
            orchestrator.predict_only()
        else:
            orchestrator.run_full_pipeline(
                csv_path=args.csv,
                legacy_db_path=args.migrate,
                force_retrain=args.retrain
            )


if __name__ == "__main__":
    main()
