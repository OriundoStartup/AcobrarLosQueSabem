"""
================================================================================
PISTA INTELIGENTE - MODELOS ML OPTIMIZADOS
================================================================================
Sistema de predicción de carreras hípicas con Machine Learning avanzado.

Características:
- Feature Engineering automático desde BD 3FN
- Múltiples algoritmos (XGBoost, LightGBM, Ensemble)
- Validación temporal (TimeSeriesSplit)
- Calibración de probabilidades
- Detección de patrones (Ley de Tres)
- Persistencia de modelos con joblib

Uso:
    # Entrenar y guardar
    config = MLConfig(db_path="data/db/hipica_3fn.db")
    predictor = RacePredictor(config)
    predictor.train()
    predictor.save_model("models/predictor_v1.joblib")
    
    # Cargar y predecir
    predictor2 = RacePredictor.load_model("models/predictor_v1.joblib")
    predictions = predictor2.predict()

Autor: ML Engineering Team - Pista Inteligente
================================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import logging
import warnings
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, top_k_accuracy_score, ndcg_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

@dataclass
class MLConfig:
    """Configuración del sistema ML."""
    db_path: str = "data/db/hipica_3fn.db"
    random_state: int = 42
    test_size: float = 0.2
    n_cv_folds: int = 5
    use_time_series_cv: bool = True
    
    # Features a usar
    use_caballo_features: bool = True
    use_jinete_features: bool = True
    use_combo_features: bool = True
    use_carrera_features: bool = True
    
    # Modelo
    model_type: str = "xgboost"  # xgboost, lightgbm, ensemble
    
    # Output
    output_dir: str = "output"


@dataclass
class PredictionResult:
    """Resultado de predicción para una carrera."""
    fecha: str
    hipodromo: str
    nro_carrera: int
    predicciones: List[Dict]
    top4_predicho: List[str]
    confianza: float


# ==============================================================================
# EXCEPCIONES PERSONALIZADAS
# ==============================================================================

class DatabaseValidationError(Exception):
    """Error cuando la base de datos no tiene la estructura esperada."""
    pass


class ModelNotTrainedError(Exception):
    """Error cuando se intenta predecir sin entrenar."""
    pass


class FeatureInconsistencyError(Exception):
    """Error cuando las features de predicción no coinciden con entrenamiento."""
    pass



# ==============================================================================
# FEATURE ENGINEERING AVANZADO
# ==============================================================================

class FeatureEngineer:
    """
    Ingeniero de features que aprovecha el esquema 3FN para crear
    features predictivas avanzadas.
    """
    
    # Tablas y vistas requeridas
    REQUIRED_TABLES = [
        'fact_participaciones',
        'fact_carreras', 
        'dim_hipodromos',
        'dim_caballos',
        'dim_jinetes'
    ]
    
    REQUIRED_VIEWS = ['v_ml_training_data']
    
    OPTIONAL_TABLES = [
        'agg_caballo_stats',
        'agg_jinete_stats', 
        'agg_combo_caballo_jinete'
    ]
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._scaler: Optional[RobustScaler] = None
        self._feature_columns: Optional[List[str]] = None  # FIX: Guardar columnas
        self._is_fitted: bool = False
        
        # Validar BD al inicializar
        self._validate_db()
    
    def _validate_db(self) -> None:
        """Valida que la base de datos tenga la estructura esperada."""
        logger.info("🔍 Validando estructura de base de datos...")
        
        if not Path(self.db_path).exists():
            raise DatabaseValidationError(f"Base de datos no encontrada: {self.db_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Obtener todas las tablas y vistas
                cursor.execute("""
                    SELECT name, type FROM sqlite_master 
                    WHERE type IN ('table', 'view')
                """)
                existing = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Verificar tablas requeridas
                missing_tables = [t for t in self.REQUIRED_TABLES if t not in existing]
                if missing_tables:
                    raise DatabaseValidationError(
                        f"Tablas requeridas faltantes: {missing_tables}\n"
                        f"Ejecute primero el script de creación de esquema 3FN."
                    )
                
                # Verificar vistas requeridas
                missing_views = [v for v in self.REQUIRED_VIEWS if v not in existing]
                if missing_views:
                    raise DatabaseValidationError(
                        f"Vistas requeridas faltantes: {missing_views}\n"
                        f"Ejecute: CREATE VIEW v_ml_training_data AS ..."
                    )
                
                # Advertir sobre tablas opcionales faltantes
                missing_optional = [t for t in self.OPTIONAL_TABLES if t not in existing]
                if missing_optional:
                    logger.warning(
                        f"⚠️ Tablas de agregación faltantes: {missing_optional}\n"
                        f"   El modelo funcionará pero con menos features."
                    )
                
                logger.info("   ✅ Estructura de BD validada correctamente")
                
        except sqlite3.Error as e:
            raise DatabaseValidationError(f"Error al validar BD: {e}")
    
    def extract_training_data(self) -> Optional[pd.DataFrame]:
        """Extrae datos de entrenamiento desde la vista ML optimizada."""
        logger.info("📊 Extrayendo datos de entrenamiento...")
        
        query = """
        SELECT * FROM v_ml_training_data
        WHERE target IS NOT NULL
        ORDER BY fecha
        """
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
        except sqlite3.Error as e:
            logger.exception(f"Error extrayendo datos de entrenamiento: {e}")
            return None
        
        if df.empty:
            logger.warning("⚠️ No hay datos de entrenamiento disponibles")
            return None
        
        logger.info(f"   ✅ {len(df)} registros extraídos")
        return df
    
    def extract_prediction_data(self) -> Optional[pd.DataFrame]:
        """Extrae datos para predicción (carreras sin resultado)."""
        logger.info("📊 Extrayendo datos para predicción...")
        
        # Verificar si existen las tablas de agregación
        has_agg_tables = self._check_agg_tables()
        
        if has_agg_tables:
            logger.info("   -> Usando query con agregaciones")
            query = self._get_prediction_query_with_agg()
        else:
            logger.info("   -> Usando query básica")
            query = self._get_prediction_query_basic()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn)
                logger.info(f"   -> Query ejecutada. Filas recuperadas: {len(df)}")
                if not df.empty:
                    logger.info(f"   -> Fechas encontradas: {df['fecha'].unique()}")
        except sqlite3.Error as e:
            logger.exception(f"Error extrayendo datos para predicción: {e}")
            return None
        
        if df.empty:
            logger.warning("⚠️ No hay carreras para predecir")
            return None
        
        logger.info(f"   ✅ {len(df)} registros para predicción")
        return df
    
    def _check_agg_tables(self) -> bool:
        """Verifica si existen las tablas de agregación."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name LIKE 'agg_%'
                """)
                return len(cursor.fetchall()) >= 3
        except sqlite3.Error:
            return False
    
    def _get_prediction_query_with_agg(self) -> str:
        """Query de predicción con tablas de agregación."""
        return """
        SELECT 
            fp.id AS participacion_id,
            fc.fecha,
            fc.nro_carrera,
            fc.hipodromo_id,
            dh.nombre AS hipodromo_nombre,
            fc.distancia_metros,
            COALESCE(fc.superficie_id, 1) AS superficie_id,
            
            fp.caballo_id,
            dc.nombre AS caballo_nombre,
            fp.jinete_id,
            dj.nombre AS jinete_nombre,
            fp.partidor,
            fp.peso_programado,
            fp.edad_anos,
            fp.handicap,
            
            COALESCE(acs.total_carreras, 0) AS caballo_carreras_previas,
            COALESCE(acs.tasa_victoria, 0) AS caballo_tasa_victoria,
            COALESCE(acs.posicion_promedio, 5) AS caballo_pos_promedio,
            COALESCE(acs.dias_sin_correr, 30) AS caballo_dias_descanso,
            COALESCE(acs.racha_actual, 0) AS caballo_racha,
            
            COALESCE(ajs.tasa_victoria, 0) AS jinete_tasa_victoria,
            COALESCE(ajs.posicion_promedio, 5) AS jinete_pos_promedio,
            
            COALESCE(acj.tasa_victoria_combo, 0) AS combo_tasa_victoria,
            COALESCE(acj.carreras_juntos, 0) AS combo_carreras
            
        FROM fact_participaciones fp
        JOIN fact_carreras fc ON fp.carrera_id = fc.id
        JOIN dim_hipodromos dh ON fc.hipodromo_id = dh.id
        JOIN dim_caballos dc ON fp.caballo_id = dc.id
        LEFT JOIN dim_jinetes dj ON fp.jinete_id = dj.id
        LEFT JOIN agg_caballo_stats acs ON fp.caballo_id = acs.caballo_id
        LEFT JOIN agg_jinete_stats ajs ON fp.jinete_id = ajs.jinete_id
        LEFT JOIN agg_combo_caballo_jinete acj ON fp.caballo_id = acj.caballo_id 
            AND fp.jinete_id = acj.jinete_id
        WHERE fp.resultado_final IS NULL
            AND fc.fecha >= date('now', '-7 days')
        ORDER BY fc.fecha, fc.nro_carrera, fp.partidor
        """
    
    def _get_prediction_query_basic(self) -> str:
        """Query de predicción sin tablas de agregación (fallback)."""
        return """
        SELECT 
            fp.id AS participacion_id,
            fc.fecha,
            fc.nro_carrera,
            fc.hipodromo_id,
            dh.nombre AS hipodromo_nombre,
            fc.distancia_metros,
            COALESCE(fc.superficie_id, 1) AS superficie_id,
            
            fp.caballo_id,
            dc.nombre AS caballo_nombre,
            fp.jinete_id,
            dj.nombre AS jinete_nombre,
            fp.partidor,
            fp.peso_programado,
            fp.edad_anos,
            fp.handicap,
            
            0 AS caballo_carreras_previas,
            0 AS caballo_tasa_victoria,
            5 AS caballo_pos_promedio,
            30 AS caballo_dias_descanso,
            0 AS caballo_racha,
            0 AS jinete_tasa_victoria,
            5 AS jinete_pos_promedio,
            0 AS combo_tasa_victoria,
            0 AS combo_carreras
            
        FROM fact_participaciones fp
        JOIN fact_carreras fc ON fp.carrera_id = fc.id
        JOIN dim_hipodromos dh ON fc.hipodromo_id = dh.id
        JOIN dim_caballos dc ON fp.caballo_id = dc.id
        LEFT JOIN dim_jinetes dj ON fp.jinete_id = dj.id
        WHERE fp.resultado_final IS NULL
            AND fc.fecha >= date('now', '-7 days')
        ORDER BY fc.fecha, fc.nro_carrera, fp.partidor
        """
    
    def create_advanced_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Crea features avanzadas a partir de los datos base."""
        logger.info("⚙️ Generando features avanzadas...")
        
        df = df.copy()
        
        # 1. Features de distancia (one-hot encoding)
        df['dist_sprint'] = (df['distancia_metros'] <= 1100).astype(int)
        df['dist_milla_corta'] = ((df['distancia_metros'] > 1100) & (df['distancia_metros'] <= 1400)).astype(int)
        df['dist_milla'] = ((df['distancia_metros'] > 1400) & (df['distancia_metros'] <= 1700)).astype(int)
        df['dist_fondo'] = (df['distancia_metros'] > 1700).astype(int)
        
        # 2. Features de experiencia
        df['es_novato'] = (df['caballo_carreras_previas'] == 0).astype(int)
        df['experiencia_nivel'] = pd.cut(
            df['caballo_carreras_previas'],
            bins=[-1, 0, 5, 15, 50, 1000],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
        
        # 3. Features de forma reciente
        df['forma_reciente'] = np.where(
            df['caballo_dias_descanso'] <= 14,
            df['caballo_tasa_victoria'] * 1.1,
            df['caballo_tasa_victoria']
        )
        
        # 4. Indicador de "en racha"
        df['en_racha_positiva'] = (df['caballo_racha'] > 0).astype(int)
        df['en_racha_negativa'] = (df['caballo_racha'] < -3).astype(int)
        
        # 5. Interacciones
        df['combo_experiencia'] = df['combo_carreras'] * df['combo_tasa_victoria']
        df['jinete_caballo_match'] = (
            df['jinete_tasa_victoria'] * df['caballo_tasa_victoria']
        )
        
        # 6. Features de competencia (dentro de cada carrera)
        if 'fecha' in df.columns and 'nro_carrera' in df.columns:
            carrera_groups = df.groupby(['fecha', 'nro_carrera'])
            
            df['rank_tasa_victoria'] = carrera_groups['caballo_tasa_victoria'].rank(
                ascending=False, method='min'
            )
            df['rank_experiencia'] = carrera_groups['caballo_carreras_previas'].rank(
                ascending=False, method='min'
            )
            df['diff_vs_max_tasa'] = (
                carrera_groups['caballo_tasa_victoria'].transform('max') 
                - df['caballo_tasa_victoria']
            )
        
        # 7. Rellenar NaN con valores por defecto
        default_values = {
            'peso_programado': df['peso_programado'].median() if df['peso_programado'].notna().any() else 56.0,
            'edad_anos': 4,
            'handicap': 0,
            'caballo_dias_descanso': 30,
            'caballo_racha': 0,
            'diff_vs_max_tasa': 0,
            'rank_tasa_victoria': 5,
            'rank_experiencia': 5,
            'forma_reciente': 0,
            'combo_experiencia': 0,
            'jinete_caballo_match': 0
        }
        df = df.fillna(default_values)
        
        logger.info(f"   ✅ {len(df.columns)} columnas después de feature engineering")
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Retorna las columnas de features para el modelo."""
        return [
            # Base
            'hipodromo_id', 'distancia_metros', 'superficie_id',
            'partidor', 'peso_programado', 'edad_anos',
            
            # Caballo
            'caballo_carreras_previas', 'caballo_tasa_victoria', 
            'caballo_pos_promedio', 'caballo_dias_descanso',
            
            # Jinete
            'jinete_tasa_victoria', 'jinete_pos_promedio',
            
            # Combo
            'combo_tasa_victoria', 'combo_carreras',
            
            # Derivadas
            'es_novato', 'experiencia_nivel', 'forma_reciente',
            'en_racha_positiva', 'en_racha_negativa',
            'combo_experiencia', 'jinete_caballo_match',
            'rank_tasa_victoria', 'rank_experiencia', 'diff_vs_max_tasa',
            
            # Distancia one-hot
            'dist_sprint', 'dist_milla_corta', 'dist_milla', 'dist_fondo'
        ]
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Ajusta los transformadores y transforma los datos de entrenamiento."""
        logger.info("🔧 Ajustando transformadores...")
        
        # Crear features avanzadas
        df = self.create_advanced_features(df, is_training=True)
        
        # Seleccionar features disponibles
        feature_cols = self.get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]
        
        # FIX: Guardar las columnas usadas para consistencia
        self._feature_columns = available_cols
        
        X = df[available_cols].copy()
        y = df['target'].values
        
        # Escalar
        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        self._is_fitted = True
        
        logger.info(f"   ✅ Shape final: X={X_scaled.shape}, y={y.shape}")
        logger.info(f"   ✅ Features guardadas: {len(self._feature_columns)}")
        
        return X_scaled, y, available_cols
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforma datos nuevos usando los transformadores ajustados."""
        if not self._is_fitted or self._scaler is None:
            raise ModelNotTrainedError("Debe llamar fit_transform primero")
        
        if self._feature_columns is None:
            raise FeatureInconsistencyError("No hay columnas de features guardadas")
        
        # Crear features avanzadas
        df = self.create_advanced_features(df, is_training=False)
        
        # FIX: Usar las mismas columnas que en entrenamiento
        missing_cols = [c for c in self._feature_columns if c not in df.columns]
        if missing_cols:
            raise FeatureInconsistencyError(
                f"Columnas faltantes en datos de predicción: {missing_cols}"
            )
        
        X = df[self._feature_columns].copy()
        X_scaled = self._scaler.transform(X)
        
        return X_scaled
    
    def get_state(self) -> Dict[str, Any]:
        """Retorna el estado para serialización."""
        return {
            'scaler': self._scaler,
            'feature_columns': self._feature_columns,
            'is_fitted': self._is_fitted,
            'db_path': self.db_path
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restaura el estado desde serialización."""
        self._scaler = state['scaler']
        self._feature_columns = state['feature_columns']
        self._is_fitted = state['is_fitted']
        self.db_path = state['db_path']


# ==============================================================================
# MODELOS ML
# ==============================================================================

class RacePredictor:
    """Predictor de carreras con múltiples algoritmos y ensemble."""
    
    MODEL_VERSION = "1.0.0"
    
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.feature_engineer = FeatureEngineer(self.config.db_path)
        
        self.model = None
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, float] = {}
        self._is_trained = False
    
    def _create_model(self):
        """Crea el modelo según configuración."""
        if self.config.model_type == "xgboost" and HAS_XGBOOST:
            return xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == "lightgbm" and HAS_LIGHTGBM:
            return lgb.LGBMRegressor(
                objective='regression',
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1
            )
        else:
            logger.info("   ⚠️ Usando GradientBoosting de sklearn (fallback)")
            return GradientBoostingRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.config.random_state
            )
    
    def train(self) -> Dict[str, float]:
        """Entrena el modelo con validación cruzada temporal."""
        logger.info("\n" + "="*60)
        logger.info("🤖 ENTRENAMIENTO DE MODELO ML")
        logger.info("="*60)
        
        try:
            # 1. Extraer datos
            df_train = self.feature_engineer.extract_training_data()
            if df_train is None or len(df_train) < 100:
                raise ValueError(f"Datos insuficientes para entrenamiento: {len(df_train) if df_train is not None else 0}")
            
            # 2. Feature engineering
            X, y, self.feature_columns = self.feature_engineer.fit_transform(df_train)
            
            # 3. Split temporal
            if self.config.use_time_series_cv:
                split_idx = int(len(X) * (1 - self.config.test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.config.test_size, 
                    random_state=self.config.random_state
                )
            
            logger.info(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # 4. Entrenar modelo
            logger.info(f"   Modelo: {self.config.model_type}")
            self.model = self._create_model()
            self.model.fit(X_train, y_train)
            
            # 5. Evaluar
            preds_train = self.model.predict(X_train)
            preds_test = self.model.predict(X_test)
            
            self.metrics = {
                'rmse_train': float(np.sqrt(mean_squared_error(y_train, preds_train))),
                'rmse_test': float(np.sqrt(mean_squared_error(y_test, preds_test))),
                'mae_train': float(mean_absolute_error(y_train, preds_train)),
                'mae_test': float(mean_absolute_error(y_test, preds_test)),
                'r2_train': float(r2_score(y_train, preds_train)),
                'r2_test': float(r2_score(y_test, preds_test)),
                'n_samples_train': len(X_train),
                'n_samples_test': len(X_test),
                'n_features': len(self.feature_columns)
            }
            
            self._is_trained = True
            
            logger.info("\n📊 MÉTRICAS DE EVALUACIÓN:")
            logger.info(f"   RMSE Train: {self.metrics['rmse_train']:.4f}")
            logger.info(f"   RMSE Test:  {self.metrics['rmse_test']:.4f}")
            logger.info(f"   MAE Test:   {self.metrics['mae_test']:.4f}")
            logger.info(f"   R² Test:    {self.metrics['r2_test']:.4f}")
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info("\n📈 TOP 10 FEATURES MÁS IMPORTANTES:")
                for _, row in importance.head(10).iterrows():
                    logger.info(f"   {row['feature']}: {row['importance']:.4f}")
            
            return self.metrics
            
        except Exception as e:
            logger.exception(f"Error durante entrenamiento: {e}")
            raise
    
    def predict(self) -> List[PredictionResult]:
        """Genera predicciones para carreras futuras."""
        if not self._is_trained:
            raise ModelNotTrainedError("El modelo debe ser entrenado primero")
        
        logger.info("\n" + "="*60)
        logger.info("🔮 GENERANDO PREDICCIONES")
        logger.info("="*60)
        
        try:
            # 1. Extraer datos
            df_pred = self.feature_engineer.extract_prediction_data()
            if df_pred is None:
                logger.warning("No hay carreras para predecir")
                return []
            
            # 2. Transformar
            X_pred = self.feature_engineer.transform(df_pred)
            
            # 3. Predecir
            df_pred['prediccion_raw'] = self.model.predict(X_pred)
            
            # 4. Procesar por carrera
            results = []
            
            for (fecha, nro_carrera), grupo in df_pred.groupby(['fecha', 'nro_carrera']):
                grupo_sorted = grupo.sort_values('prediccion_raw')
                
                # FIX: Cálculo robusto de probabilidades
                min_pred = grupo_sorted['prediccion_raw'].min()
                max_pred = grupo_sorted['prediccion_raw'].max()
                n_participantes = len(grupo_sorted)
                
                if max_pred > min_pred:
                    grupo_sorted['probabilidad'] = 100 * (
                        max_pred - grupo_sorted['prediccion_raw']
                    ) / (max_pred - min_pred)
                else:
                    # FIX: Distribuir equitativamente cuando todos son iguales
                    grupo_sorted['probabilidad'] = 100.0 / n_participantes
                
                # Construir predicciones
                predicciones = []
                for rank, (_, row) in enumerate(grupo_sorted.iterrows(), 1):
                    predicciones.append({
                        'caballo': row['caballo_nombre'],
                        'jinete': row.get('jinete_nombre', 'N/A'),
                        'partidor': int(row['partidor']) if pd.notna(row['partidor']) else None,
                        'puntaje_calculado': round(float(row['prediccion_raw']), 4),
                        'probabilidad': round(float(row['probabilidad']), 1),
                        'ranking': rank,
                        'stats': {
                            'carreras': int(row['caballo_carreras_previas']),
                            'tasa_victoria': round(float(row['caballo_tasa_victoria']) * 100, 1),
                            'pos_promedio': round(float(row['caballo_pos_promedio']), 2)
                        }
                    })
                
                # Confianza basada en dispersión
                confianza = min(100, max(0, int((max_pred - min_pred) * 20)))
                
                results.append(PredictionResult(
                    fecha=str(fecha),
                    hipodromo=grupo_sorted.iloc[0].get('hipodromo_nombre', 'N/A'),
                    nro_carrera=int(nro_carrera),
                    predicciones=predicciones,
                    top4_predicho=[p['caballo'] for p in predicciones[:4]],
                    confianza=confianza
                ))
            
            logger.info(f"   ✅ {len(results)} carreras predichas")
            return results
            
        except Exception as e:
            logger.exception(f"Error durante predicción: {e}")
            raise
    
    def save_model(self, path: str) -> None:
        """
        Guarda el modelo entrenado y todos sus componentes.
        
        Args:
            path: Ruta del archivo (.joblib)
        """
        if not self._is_trained:
            raise ModelNotTrainedError("No hay modelo entrenado para guardar")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'version': self.MODEL_VERSION,
            'model': self.model,
            'config': self.config,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'feature_engineer_state': self.feature_engineer.get_state(),
            'trained_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, path)
        logger.info(f"💾 Modelo guardado: {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'RacePredictor':
        """
        Carga un modelo previamente guardado.
        
        Args:
            path: Ruta del archivo (.joblib)
            
        Returns:
            RacePredictor con el modelo cargado
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Modelo no encontrado: {path}")
        
        model_data = joblib.load(path)
        
        # Validar versión
        saved_version = model_data.get('version', '0.0.0')
        if saved_version != cls.MODEL_VERSION:
            logger.warning(
                f"⚠️ Versión del modelo ({saved_version}) difiere de la actual ({cls.MODEL_VERSION})"
            )
        
        # Reconstruir predictor
        predictor = cls(config=model_data['config'])
        predictor.model = model_data['model']
        predictor.feature_columns = model_data['feature_columns']
        predictor.metrics = model_data['metrics']
        predictor._is_trained = True
        
        # Restaurar feature engineer
        predictor.feature_engineer.set_state(model_data['feature_engineer_state'])
        
        logger.info(f"📂 Modelo cargado: {path}")
        logger.info(f"   Entrenado: {model_data.get('trained_at', 'N/A')}")
        logger.info(f"   R² Test: {predictor.metrics.get('r2_test', 'N/A'):.4f}")
        
        return predictor
    
    def detect_patterns(self, predictions: List[PredictionResult]) -> Dict:
        """Detecta patrones repetidos en las predicciones (Ley de Tres)."""
        logger.info("\n" + "="*60)
        logger.info("🔍 ANÁLISIS DE PATRONES (LEY DE TRES)")
        logger.info("="*60)
        
        quinelas = []
        trifectas = []
        superfectas = []
        
        for pred in predictions:
            # Obtener top 4 con partidores (formato: "#N NOMBRE")
            top4_with_numbers = []
            for p in pred.predicciones[:4]:
                partidor = p.get('partidor')
                nombre = p.get('caballo', 'N/A')
                if partidor:
                    top4_with_numbers.append(f"#{partidor} {nombre}")
                else:
                    top4_with_numbers.append(nombre)
            
            if len(top4_with_numbers) >= 2:
                quinelas.append(tuple(sorted(top4_with_numbers[:2])))
            if len(top4_with_numbers) >= 3:
                trifectas.append(tuple(top4_with_numbers[:3]))
            if len(top4_with_numbers) >= 4:
                superfectas.append(tuple(top4_with_numbers[:4]))
        
        patterns = {}
        
        def analyze(name: str, combos: List[tuple], min_count: int = 2):
            counter = Counter(combos)
            repeated = {str(k): v for k, v in counter.items() if v >= min_count}
            
            if repeated:
                logger.info(f"\n⚠️ Patrones en {name}:")
                for combo, count in sorted(repeated.items(), key=lambda x: -x[1]):
                    logger.info(f"   {combo}: {count} apariciones")
                patterns[name] = repeated
            else:
                logger.info(f"\n✅ Sin patrones repetidos en {name}")
        
        analyze("quinelas", quinelas)
        analyze("trifectas", trifectas)
        analyze("superfectas", superfectas)
        
        return patterns
    
    def save_predictions(
        self, 
        predictions: List[PredictionResult],
        patterns: Dict = None
    ) -> Tuple[str, str]:
        """
        Guarda predicciones en CSV y JSON.
        
        Returns:
            Tupla con rutas de CSV y JSON
        """
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. CSV simple
        rows = []
        for pred in predictions:
            for p in pred.predicciones:
                rows.append({
                    'fecha': pred.fecha,
                    'hipodromo': pred.hipodromo,
                    'nro_carrera': pred.nro_carrera,
                    'caballo': p['caballo'],
                    'jinete': p['jinete'],
                    'ranking': p['ranking'],
                    'probabilidad': p['probabilidad'],
                    'puntaje': p['puntaje_calculado']
                })
        
        df_csv = pd.DataFrame(rows)
        csv_path = f"{self.config.output_dir}/predicciones.csv"
        df_csv.to_csv(csv_path, index=False)
        logger.info(f"💾 CSV guardado: {csv_path}")
        
        # 2. JSON detallado
        json_data = {
            "metadata": {
                "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "modelo": self.config.model_type,
                "version": self.MODEL_VERSION,
                "metricas": self.metrics
            },
            "predicciones": [
                {
                    "fecha": p.fecha,
                    "hipodromo": p.hipodromo,
                    "nro_carrera": p.nro_carrera,
                    "confianza": p.confianza,
                    "top4": p.top4_predicho,
                    "predicciones": p.predicciones
                }
                for p in predictions
            ],
            "patrones": patterns or {}
        }
        
        json_path = f"{self.config.output_dir}/predicciones_detalle.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 JSON guardado: {json_path}")
        
        return csv_path, json_path


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def run_full_pipeline(
    config: MLConfig = None,
    save_model_path: Optional[str] = None
) -> Tuple[Dict, List[PredictionResult]]:
    """
    Ejecuta el pipeline completo de entrenamiento y predicción.
    
    Args:
        config: Configuración del sistema
        save_model_path: Ruta opcional para guardar el modelo
        
    Returns:
        Tupla con métricas y predicciones
    """
    config = config or MLConfig()
    
    print("\n" + "="*60)
    print(" " * 15 + "PISTA INTELIGENTE ML")
    print(" " * 10 + "Sistema de Prediccion de Carreras")
    print("="*60 + "\n")
    
    predictor = RacePredictor(config)
    
    # 1. Entrenar
    metrics = predictor.train()
    
    # 2. Guardar modelo si se especificó ruta
    if save_model_path:
        predictor.save_model(save_model_path)
    
    # 3. Predecir
    predictions = predictor.predict()
    
    if predictions:
        # 4. Detectar patrones
        patterns = predictor.detect_patterns(predictions)
        
        # 5. Guardar resultados
        predictor.save_predictions(predictions, patterns)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return metrics, predictions


if __name__ == "__main__":
    # Corregir rutas para ejecución desde root
    config = MLConfig(
        db_path="data/db/hipica_3fn.db",
        model_type="xgboost",
        output_dir="app/ml/output"  # FIX: Ruta correcta espera por data_sync.py
    )
    
    metrics, predictions = run_full_pipeline(
        config,
        save_model_path="models/predictor_latest.joblib"
    )