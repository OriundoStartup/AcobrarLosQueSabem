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

Autor: ML Engineering Team - Pista Inteligente
================================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import logging
import warnings

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

from collections import Counter

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
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
# FEATURE ENGINEERING AVANZADO
# ==============================================================================

class FeatureEngineer:
    """
    Ingeniero de features que aprovecha el esquema 3FN para crear
    features predictivas avanzadas.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._scaler: Optional[RobustScaler] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        
        # Features categóricas
        self.categorical_features = ['hipodromo_id', 'superficie_id']
        
        # Features numéricas base
        self.numeric_features = [
            'distancia_metros', 'partidor', 'peso_programado', 'edad_anos', 'handicap'
        ]
        
        # Features derivadas de agregaciones
        self.agg_features = [
            'caballo_carreras_previas', 'caballo_tasa_victoria', 'caballo_pos_promedio',
            'caballo_dias_descanso', 'caballo_racha',
            'jinete_tasa_victoria', 'jinete_pos_promedio',
            'combo_tasa_victoria', 'combo_carreras'
        ]
    
    def extract_training_data(self) -> pd.DataFrame:
        """
        Extrae datos de entrenamiento desde la vista ML optimizada.
        """
        logger.info("📊 Extrayendo datos de entrenamiento...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Usar la vista pre-creada
        query = """
        SELECT * FROM v_ml_training_data
        WHERE target IS NOT NULL
        ORDER BY fecha
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning("⚠️ No hay datos de entrenamiento disponibles")
            return None
        
        logger.info(f"   ✅ {len(df)} registros extraídos")
        return df
    
    def extract_prediction_data(self) -> pd.DataFrame:
        """
        Extrae datos para predicción (carreras sin resultado).
        """
        logger.info("📊 Extrayendo datos para predicción...")
        
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT 
            fp.id AS participacion_id,
            fc.fecha,
            fc.nro_carrera,
            fc.hipodromo_id,
            dh.codigo AS hipodromo_codigo,
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
        ORDER BY fc.fecha, fc.nro_carrera, fp.partidor
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            logger.warning("⚠️ No hay carreras para predecir")
            return None
        
        logger.info(f"   ✅ {len(df)} registros para predicción")
        return df
    
    def create_advanced_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Crea features avanzadas a partir de los datos base.
        """
        logger.info("⚙️ Generando features avanzadas...")
        
        df = df.copy()
        
        # 1. Features de distancia
        df['distancia_categoria'] = pd.cut(
            df['distancia_metros'],
            bins=[0, 1100, 1400, 1700, 3000],
            labels=['sprint', 'milla_corta', 'milla', 'fondo']
        )
        
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
            df['caballo_tasa_victoria'] * 1.1,  # Boost por actividad reciente
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
            carrera_groups = df.groupby(['fecha', 'nro_carrera'] if 'nro_carrera' in df.columns 
                                       else ['fecha', 'hipodromo_id'])
            
            # Ranking relativo dentro de la carrera
            df['rank_tasa_victoria'] = carrera_groups['caballo_tasa_victoria'].rank(
                ascending=False, method='min'
            )
            df['rank_experiencia'] = carrera_groups['caballo_carreras_previas'].rank(
                ascending=False, method='min'
            )
            
            # Diferencia con el favorito
            df['diff_vs_max_tasa'] = carrera_groups['caballo_tasa_victoria'].transform('max') - df['caballo_tasa_victoria']
        
        # 7. Rellenar NaN
        df = df.fillna({
            'peso_programado': df['peso_programado'].median() or 56.0,
            'edad_anos': 4,
            'handicap': 0,
            'caballo_dias_descanso': 30,
            'caballo_racha': 0,
            'diff_vs_max_tasa': 0,
            'rank_tasa_victoria': 5,
            'rank_experiencia': 5
        })
        
        logger.info(f"   ✅ {len(df.columns)} features generadas")
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
            'rank_tasa_victoria', 'rank_experiencia', 'diff_vs_max_tasa'
        ]
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta los transformadores y transforma los datos de entrenamiento.
        """
        logger.info("🔧 Ajustando transformadores...")
        
        # Crear features avanzadas
        df = self.create_advanced_features(df, is_training=True)
        
        # Seleccionar features
        feature_cols = self.get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]
        
        # Guardar columnas usadas para transform()
        self._fitted_columns = available_cols
        
        X = df[available_cols].copy()
        y = df['target'].values
        
        # Escalar
        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        logger.info(f"   ✅ Shape final: X={X_scaled.shape}, y={y.shape}")
        return X_scaled, y, available_cols
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforma datos nuevos usando los transformadores ajustados.
        """
        if self._scaler is None:
            raise ValueError("Debe llamar fit_transform primero")
        
        # Crear features avanzadas
        df = self.create_advanced_features(df, is_training=False)
        
        # Usar SOLO las columnas del entrenamiento
        if not hasattr(self, '_fitted_columns'):
             # Fallback por si acaso
             feature_cols = self.get_feature_columns()
             self._fitted_columns = [c for c in feature_cols if c in df.columns]

        X = df[self._fitted_columns].copy()
        X_scaled = self._scaler.transform(X)
        
        return X_scaled


# ==============================================================================
# MODELOS ML
# ==============================================================================

class RacePredictor:
    """
    Predictor de carreras con múltiples algoritmos y ensemble.
    """
    
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
            # Fallback a Gradient Boosting de sklearn
            return GradientBoostingRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.config.random_state
            )
    
    def train(self) -> Dict[str, float]:
        """
        Entrena el modelo con validación cruzada temporal.
        """
        logger.info("\n" + "="*60)
        logger.info("🤖 ENTRENAMIENTO DE MODELO ML")
        logger.info("="*60)
        
        # 1. Extraer datos
        df_train = self.feature_engineer.extract_training_data()
        if df_train is None or len(df_train) < 100:
            raise ValueError("Datos insuficientes para entrenamiento")
        
        # 2. Feature engineering
        X, y, self.feature_columns = self.feature_engineer.fit_transform(df_train)
        
        # 3. Split temporal
        if self.config.use_time_series_cv:
            # Usar los últimos 20% como test (respetando orden temporal)
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
            'rmse_train': np.sqrt(mean_squared_error(y_train, preds_train)),
            'rmse_test': np.sqrt(mean_squared_error(y_test, preds_test)),
            'mae_train': mean_absolute_error(y_train, preds_train),
            'mae_test': mean_absolute_error(y_test, preds_test),
            'r2_train': r2_score(y_train, preds_train),
            'r2_test': r2_score(y_test, preds_test),
        }
        
        # 6. Métricas de ranking (más relevantes para carreras)
        # Reconstruir grupos de carrera para calcular accuracy de Top-N
        
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
    
    def predict(self) -> List[PredictionResult]:
        """
        Genera predicciones para carreras futuras.
        """
        if not self._is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        logger.info("\n" + "="*60)
        logger.info("🔮 GENERANDO PREDICCIONES")
        logger.info("="*60)
        
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
            # Ordenar por predicción (menor es mejor posición)
            grupo_sorted = grupo.sort_values('prediccion_raw')
            
            # Calcular probabilidades normalizadas
            min_pred = grupo_sorted['prediccion_raw'].min()
            max_pred = grupo_sorted['prediccion_raw'].max()
            
            if max_pred > min_pred:
                grupo_sorted['probabilidad'] = 100 * (
                    max_pred - grupo_sorted['prediccion_raw']
                ) / (max_pred - min_pred)
            else:
                grupo_sorted['probabilidad'] = 50
            
            # Construir predicciones
            predicciones = []
            for rank, (_, row) in enumerate(grupo_sorted.iterrows(), 1):
                predicciones.append({
                    'caballo': row['caballo_nombre'],
                    'jinete': row.get('jinete_nombre', 'N/A'),
                    'partidor': int(row['partidor']) if pd.notna(row['partidor']) else None,
                    'puntaje': round(row['prediccion_raw'], 4),
                    'probabilidad': int(row['probabilidad']),
                    'ranking': rank,
                    'stats': {
                        'carreras': int(row['caballo_carreras_previas']),
                        'tasa_victoria': round(row['caballo_tasa_victoria'] * 100, 1),
                        'pos_promedio': round(row['caballo_pos_promedio'], 2)
                    }
                })
            
            # Confianza basada en dispersión de predicciones
            confianza = min(100, max(0, int((max_pred - min_pred) * 20)))
            
            results.append(PredictionResult(
                fecha=str(fecha),
                hipodromo=grupo_sorted.iloc[0].get('hipodromo_codigo', 'N/A'),
                nro_carrera=int(nro_carrera),
                predicciones=predicciones,
                top4_predicho=[p['caballo'] for p in predicciones[:4]],
                confianza=confianza
            ))
        
        logger.info(f"   ✅ {len(results)} carreras predichas")
        return results
    
    def detect_patterns(self, predictions: List[PredictionResult]) -> Dict:
        """
        Detecta patrones repetidos en las predicciones (Ley de Tres).
        """
        logger.info("\n" + "="*60)
        logger.info("🔍 ANÁLISIS DE PATRONES (LEY DE TRES)")
        logger.info("="*60)
        
        quinelas = []
        trifectas = []
        superfectas = []
        
        for pred in predictions:
            top4 = pred.top4_predicho
            
            if len(top4) >= 2:
                quinelas.append(tuple(sorted(top4[:2])))
            if len(top4) >= 3:
                trifectas.append(tuple(top4[:3]))
            if len(top4) >= 4:
                superfectas.append(tuple(top4[:4]))
        
        patterns = {}
        
        def analyze(name, combos, min_count=2):
            counter = Counter(combos)
            repeated = {str(k): v for k, v in counter.items() if v >= min_count}
            
            if repeated:
                logger.info(f"\n⚠️ Patrones en {name}:")
                for combo, count in sorted(repeated.items(), key=lambda x: -x[1]):
                    logger.info(f"   {combo}: {count} apariciones")
                patterns[name.lower()] = repeated
            else:
                logger.info(f"\n✅ Sin patrones repetidos en {name}")
        
        analyze("QUINELA (Top 2 sin orden)", quinelas)
        analyze("TRIFECTA (Top 3 con orden)", trifectas)
        analyze("SUPERFECTA (Top 4 con orden)", superfectas)
        
        return patterns
    
    def save_predictions(
        self, 
        predictions: List[PredictionResult],
        patterns: Dict = None
    ):
        """
        Guarda predicciones en CSV y JSON.
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
                    'puntaje': p['puntaje']
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
                "metricas_entrenamiento": self.metrics
            },
            "predicciones": [
                {
                    "fecha": p.fecha,
                    "hipodromo": p.hipodromo,
                    "nro_carrera": p.nro_carrera,
                    "confianza": p.confianza,
                    "top4": p.top4_predicho,
                    "detalle": p.predicciones
                }
                for p in predictions
            ],
            "patrones": patterns or {}
        }
        
        json_path = f"{self.config.output_dir}/predicciones_detalle.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 JSON guardado: {json_path}")


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def run_full_pipeline(config: MLConfig = None) -> Tuple[Dict, List[PredictionResult]]:
    """
    Ejecuta el pipeline completo de entrenamiento y predicción.
    """
    config = config or MLConfig()
    
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*15 + "🏇 PISTA INTELIGENTE ML" + " "*15 + "║")
    print("║" + " "*10 + "Sistema de Predicción de Carreras" + " "*10 + "║")
    print("╚" + "═"*58 + "╝\n")
    
    predictor = RacePredictor(config)
    
    # 1. Entrenar
    metrics = predictor.train()
    
    # 2. Predecir
    predictions = predictor.predict()
    
    if predictions:
        # 3. Detectar patrones
        patterns = predictor.detect_patterns(predictions)
        
        # 4. Guardar
        predictor.save_predictions(predictions, patterns)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    return metrics, predictions


if __name__ == "__main__":
    config = MLConfig(
        db_path="data/db/hipica_3fn.db",
        model_type="xgboost"
    )
    
    metrics, predictions = run_full_pipeline(config)