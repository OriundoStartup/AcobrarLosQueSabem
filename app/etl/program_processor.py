"""
================================================================================
PISTA INTELIGENTE - PROCESADOR DE PROGRAMAS HÍPICOS
================================================================================
Módulo: program_processor.py

Procesa archivos CSV de programas y resultados de carreras hípicas:
1. Lee archivos de exports/raw/
2. Estandariza nomenclatura (PROGRAMA/RESULTADOS)_HIPODROMO_FECHA.csv
3. Mueve a carpeta de procesamiento del ETL
4. Ejecuta el pipeline ETL para cargar a BD
5. Genera predicciones automáticamente

Nomenclatura de archivos:
- PROGRAMA_CHC_2025-12-05.csv (Club Hípico Chile)
- PROGRAMA_HC_2025-12-06.csv (Hipódromo Chile)
- PROGRAMA_VSC_2025-12-07.csv (Valparaíso Sporting Club)
- RESULTADOS_CHC_2025-12-05.csv

Autor: Pista Inteligente Team
================================================================================
"""

import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, List
import pandas as pd
import logging

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
APP_DIR = PROJECT_ROOT / "app"

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR / "etl"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

class Config:
    """Configuración del procesador."""
    # Directorios
    RAW_DIR = PROJECT_ROOT / "exports" / "raw"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "procesados"
    DB_PATH = PROJECT_ROOT / "data" / "db" / "hipica_3fn.db"
    
    # Mapeo de códigos de hipódromos
    HIPODROMO_CODES = {
        "CHC": "CHC",   # Club Hípico Chile
        "CHS": "CHC",   # Alternativo
        "HC": "HC",     # Hipódromo Chile
        "VSC": "VSC",   # Valparaíso Sporting Club
        "SPN": "VSC",   # Alternativo
    }
    
    # Patrones de nombres de archivo
    FILENAME_PATTERNS = {
        "programa": re.compile(
            r"PROGRAMA[_-]([A-Z]{2,3})[_-](\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2})\.csv",
            re.IGNORECASE
        ),
        "resultados": re.compile(
            r"resul(?:tados)?[_-]([A-Z]{2,3})[_-](\d{2}[-/.]\d{2}[-/.]\d{4}|\d{4}[-/.]\d{2}[-/.]\d{2})(?:\.csv)?",
            re.IGNORECASE
        ),
    }


# ==============================================================================
# UTILIDADES
# ==============================================================================

def parse_date(date_str: str) -> Optional[date]:
    """Parsea fecha desde string con múltiples formatos."""
    date_str = date_str.replace("/", "-").replace(".", "-")
    
    formats = [
        "%Y-%m-%d",  # 2025-12-05
        "%d-%m-%Y",  # 05-12-2025
        "%Y%m%d",    # 20251205
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    return None


def detect_file_type(filename: str) -> Tuple[Optional[str], Optional[str], Optional[date]]:
    """
    Detecta tipo de archivo, hipódromo y fecha desde el nombre.
    
    Returns:
        Tuple (tipo, hipodromo_code, fecha)
    """
    for file_type, pattern in Config.FILENAME_PATTERNS.items():
        match = pattern.match(filename)
        if match:
            hipodromo_raw = match.group(1).upper()
            date_str = match.group(2)
            
            hipodromo = Config.HIPODROMO_CODES.get(hipodromo_raw, hipodromo_raw)
            fecha = parse_date(date_str)
            
            return file_type, hipodromo, fecha
    
    return None, None, None


def standardize_filename(file_type: str, hipodromo: str, fecha: date) -> str:
    """Genera nombre de archivo estandarizado."""
    tipo_upper = file_type.upper()
    fecha_str = fecha.strftime("%Y-%m-%d")
    return f"{tipo_upper}_{hipodromo}_{fecha_str}.csv"


# ==============================================================================
# PROCESADOR DE ARCHIVOS
# ==============================================================================

class ProgramProcessor:
    """Procesa archivos de programas y resultados."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.processed_files: List[str] = []
        self.errors: List[str] = []
    
    def scan_raw_directory(self) -> List[Path]:
        """Escanea directorio raw y retorna archivos CSV pendientes."""
        raw_dir = self.config.RAW_DIR
        
        if not raw_dir.exists():
            logger.warning(f"Directorio raw no existe: {raw_dir}")
            return []
        
        csv_files = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("resul_*"))
        logger.info(f"📁 Encontrados {len(csv_files)} archivos en {raw_dir}")
        
        return csv_files
    
    def process_file(self, file_path: Path) -> bool:
        """
        Procesa un archivo individual.
        
        1. Detecta tipo y metadata
        2. Estandariza nombre si es necesario
        3. Valida formato CSV
        4. Mueve a carpeta de procesados
        """
        filename = file_path.name
        logger.info(f"📄 Procesando: {filename}")
        
        # Detectar tipo
        file_type, hipodromo, fecha = detect_file_type(filename)
        
        if not file_type:
            logger.warning(f"   ⚠️ No se pudo detectar tipo de archivo: {filename}")
            self.errors.append(f"Tipo no detectado: {filename}")
            return False
        
        if not fecha:
            logger.warning(f"   ⚠️ No se pudo detectar fecha: {filename}")
            self.errors.append(f"Fecha no detectada: {filename}")
            return False
        
        logger.info(f"   ✅ Tipo: {file_type}, Hipódromo: {hipodromo}, Fecha: {fecha}")
        
        # Estandarizar nombre
        new_filename = standardize_filename(file_type, hipodromo, fecha)
        target_path = self.config.PROCESSED_DIR / new_filename
        
        # Validar CSV
        try:
            df = pd.read_csv(file_path)
            logger.info(f"   📊 {len(df)} registros, columnas: {list(df.columns)[:5]}...")
        except Exception as e:
            logger.error(f"   ❌ Error leyendo CSV: {e}")
            self.errors.append(f"Error CSV {filename}: {e}")
            return False
        
        # Copiar a procesados (mantener original en raw)
        try:
            shutil.copy2(file_path, target_path)
            logger.info(f"   📦 Copiado a: {target_path.name}")
            self.processed_files.append(str(target_path))
            return True
        except Exception as e:
            logger.error(f"   ❌ Error copiando: {e}")
            self.errors.append(f"Error copiando {filename}: {e}")
            return False
    
    def process_all(self) -> int:
        """Procesa todos los archivos pendientes."""
        files = self.scan_raw_directory()
        processed_count = 0
        
        for file_path in files:
            if file_path.name.startswith("."):
                continue
            
            if self.process_file(file_path):
                processed_count += 1
        
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 RESUMEN: {processed_count}/{len(files)} archivos procesados")
        
        if self.errors:
            logger.warning(f"⚠️ Errores: {len(self.errors)}")
            for error in self.errors:
                logger.warning(f"   - {error}")
        
        return processed_count
    
    def run_etl_pipeline(self) -> bool:
        """Ejecuta el pipeline ETL para cargar datos a BD."""
        if not self.processed_files:
            logger.info("📭 No hay archivos nuevos para procesar")
            return True
        
        logger.info(f"\n🔄 Ejecutando ETL Pipeline...")
        
        try:
            from etl_pipeline import ETLPipeline, ETLConfig
            
            etl_config = ETLConfig(
                db_path=str(self.config.DB_PATH),
                auto_detect_csv=True,
                auto_predict=True
            )
            
            with ETLPipeline(etl_config) as pipeline:
                for csv_path in self.processed_files:
                    logger.info(f"   📥 Procesando: {Path(csv_path).name}")
                    result = pipeline.process_csv(csv_path)
                    
                    if result.status == "success":
                        logger.info(f"      ✅ Insertados: {result.records_inserted}, Actualizados: {result.records_updated}")
                    else:
                        logger.error(f"      ❌ Error: {result.errors}")
                
                # Actualizar agregaciones para ML
                pipeline.update_aggregations()
            
            logger.info("✅ ETL Pipeline completado")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Error importando ETL: {e}")
            logger.info("   Ejecuta manualmente: python -m app.etl.etl_pipeline")
            return False
        except Exception as e:
            logger.exception(f"❌ Error en ETL: {e}")
            return False
    
    def generate_predictions(self) -> bool:
        """Genera predicciones con el modelo ML."""
        logger.info(f"\n🧠 Generando predicciones...")
        
        try:
            from ml.predictor import RacePredictor, MLConfig
            
            ml_config = MLConfig(db_path=str(self.config.DB_PATH))
            predictor = RacePredictor(ml_config)
            
            # Cargar modelo existente
            model_path = PROJECT_ROOT / "models" / "predictor_latest.joblib"
            if model_path.exists():
                predictor.load(str(model_path))
                results = predictor.predict_upcoming_races()
                
                if results:
                    logger.info(f"   ✅ Generadas predicciones para {len(results)} carreras")
                else:
                    logger.info("   📭 No hay carreras pendientes de predicción")
            else:
                logger.warning(f"   ⚠️ Modelo no encontrado: {model_path}")
            
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️ Módulo ML no disponible: {e}")
            return False
        except Exception as e:
            logger.exception(f"❌ Error generando predicciones: {e}")
            return False


# ==============================================================================
# FUNCIONES DE CONVENIENCIA
# ==============================================================================

def process_and_update(run_predictions: bool = True) -> dict:
    """
    Función principal para procesar archivos y actualizar la BD.
    
    Args:
        run_predictions: Si True, genera predicciones después de cargar datos
    
    Returns:
        Dict con estadísticas del proceso
    """
    processor = ProgramProcessor()
    
    # 1. Procesar archivos CSV
    processed_count = processor.process_all()
    
    # 2. Ejecutar ETL
    etl_success = processor.run_etl_pipeline()
    
    # 3. Generar predicciones
    predictions_success = False
    if run_predictions and etl_success:
        predictions_success = processor.generate_predictions()
    
    return {
        "files_processed": processed_count,
        "files_with_errors": len(processor.errors),
        "etl_success": etl_success,
        "predictions_success": predictions_success,
        "errors": processor.errors
    }


def rename_to_standard(dry_run: bool = True) -> List[Tuple[str, str]]:
    """
    Renombra archivos en raw/ a nomenclatura estándar.
    
    Args:
        dry_run: Si True, solo muestra cambios sin ejecutar
    
    Returns:
        Lista de tuplas (nombre_original, nombre_nuevo)
    """
    raw_dir = Config.RAW_DIR
    renames = []
    
    for file_path in raw_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith("."):
            file_type, hipodromo, fecha = detect_file_type(file_path.name)
            
            if file_type and hipodromo and fecha:
                new_name = standardize_filename(file_type, hipodromo, fecha)
                
                if new_name != file_path.name:
                    renames.append((file_path.name, new_name))
                    
                    if not dry_run:
                        new_path = raw_dir / new_name
                        file_path.rename(new_path)
                        logger.info(f"Renombrado: {file_path.name} → {new_name}")
    
    if dry_run:
        logger.info("\n📋 Cambios propuestos (dry-run):")
        for old, new in renames:
            logger.info(f"   {old} → {new}")
    
    return renames


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Procesador de Programas Hípicos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python program_processor.py                    # Procesa todos los archivos
  python program_processor.py --no-predictions   # Sin generar predicciones
  python program_processor.py --rename           # Renombra archivos a formato estándar
  python program_processor.py --rename --execute # Ejecuta el renombramiento
        """
    )
    
    parser.add_argument(
        "--no-predictions", 
        action="store_true",
        help="No generar predicciones después de cargar datos"
    )
    parser.add_argument(
        "--rename",
        action="store_true", 
        help="Renombra archivos a nomenclatura estándar"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Ejecuta los cambios (por defecto es dry-run)"
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║         🏇 PISTA INTELIGENTE - Procesador de Programas 🏇    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if args.rename:
        rename_to_standard(dry_run=not args.execute)
    else:
        result = process_and_update(run_predictions=not args.no_predictions)
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                      📊 RESUMEN                              ║
╠══════════════════════════════════════════════════════════════╣
║  Archivos procesados: {result['files_processed']:>3}                                  ║
║  Archivos con error:  {result['files_with_errors']:>3}                                  ║
║  ETL exitoso:         {'✅ Sí' if result['etl_success'] else '❌ No':>6}                              ║
║  Predicciones:        {'✅ Sí' if result['predictions_success'] else '❌ No':>6}                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
