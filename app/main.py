"""
================================================================================
PISTA INTELIGENTE - ORQUESTADOR PRINCIPAL
================================================================================
Script principal que coordina la migración de datos, ETL y predicciones ML.

Uso:
    python main.py migrate --legacy-db path/to/old.db
    python main.py etl --csv path/to/data.csv
    python main.py predict
    python main.py full --legacy-db path/to/old.db

Autor: Pista Inteligente Team
================================================================================
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# Agregar paths del proyecto
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.schema_3fn import DatabaseManager
from etl.pipeline import ETLPipeline, ETLConfig, migrate_legacy_to_3fn
from ml.predictor import RacePredictor, MLConfig, run_full_pipeline

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/pista_{datetime.now():%Y%m%d}.log')
    ]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURACIÓN POR DEFECTO
# ==============================================================================

DEFAULT_CONFIG = {
    'db_path': 'data/db/hipica_3fn.db',
    'legacy_db_path': 'data/db/hipica_normalizada.db',
    'output_dir': 'output',
    'model_type': 'xgboost'
}


# ==============================================================================
# COMANDOS
# ==============================================================================

def cmd_init_db(args):
    """Inicializa una nueva base de datos con esquema 3FN."""
    db_path = args.db_path or DEFAULT_CONFIG['db_path']
    
    print(f"\n🆕 Inicializando base de datos: {db_path}")
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    with DatabaseManager(db_path) as db:
        db.initialize_schema()
    
    print(f"✅ Base de datos creada exitosamente")


def cmd_migrate(args):
    """Migra datos desde BD legacy a esquema 3FN."""
    legacy_path = args.legacy_db or DEFAULT_CONFIG['legacy_db_path']
    new_path = args.db_path or DEFAULT_CONFIG['db_path']
    
    print(f"\n🔄 Migrando datos:")
    print(f"   Origen: {legacy_path}")
    print(f"   Destino: {new_path}")
    
    result = migrate_legacy_to_3fn(legacy_path, new_path)
    
    print(f"\n📊 Resultado de migración:")
    print(f"   Registros procesados: {result.records_raw}")
    print(f"   Insertados: {result.records_inserted}")
    print(f"   Actualizados: {result.records_updated}")
    print(f"   Rechazados: {result.records_rejected}")
    print(f"   Estado: {result.status}")


def cmd_etl(args):
    """Procesa un archivo CSV."""
    csv_path = args.csv
    db_path = args.db_path or DEFAULT_CONFIG['db_path']
    
    if not Path(csv_path).exists():
        print(f"❌ Archivo no encontrado: {csv_path}")
        return
    
    print(f"\n📥 Procesando CSV: {csv_path}")
    
    config = ETLConfig(db_path=db_path)
    
    with ETLPipeline(config) as pipeline:
        result = pipeline.process_csv(csv_path)
    
    print(f"\n📊 Resultado ETL:")
    print(f"   Batch ID: {result.batch_id}")
    print(f"   Registros: {result.records_raw}")
    print(f"   Insertados: {result.records_inserted}")
    print(f"   Estado: {result.status}")


def cmd_predict(args):
    """Genera predicciones para carreras futuras."""
    db_path = args.db_path or DEFAULT_CONFIG['db_path']
    output_dir = args.output or DEFAULT_CONFIG['output_dir']
    model_type = args.model or DEFAULT_CONFIG['model_type']
    
    print(f"\n🔮 Generando predicciones...")
    
    config = MLConfig(
        db_path=db_path,
        output_dir=output_dir,
        model_type=model_type
    )
    
    metrics, predictions = run_full_pipeline(config)
    
    if predictions:
        print(f"\n📊 Resumen:")
        print(f"   Carreras predichas: {len(predictions)}")
        print(f"   RMSE modelo: {metrics.get('rmse_test', 'N/A'):.4f}")


def cmd_full(args):
    """Ejecuta pipeline completo: migración + predicción."""
    print("\n" + "="*60)
    print("🚀 EJECUTANDO PIPELINE COMPLETO")
    print("="*60)
    
    # 1. Migrar si hay BD legacy
    if args.legacy_db:
        cmd_migrate(args)
    
    # 2. Generar predicciones
    cmd_predict(args)


def cmd_stats(args):
    """Muestra estadísticas de la base de datos."""
    db_path = args.db_path or DEFAULT_CONFIG['db_path']
    
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"\n📊 Estadísticas de: {db_path}")
    print("-" * 40)
    
    # Conteos
    tables = [
        ('fact_carreras', 'Carreras'),
        ('fact_participaciones', 'Participaciones'),
        ('dim_caballos', 'Caballos'),
        ('dim_jinetes', 'Jinetes'),
        ('dim_studs', 'Studs'),
    ]
    
    for table, name in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   {name}: {count:,}")
        except:
            print(f"   {name}: N/A")
    
    # Rango de fechas
    try:
        cursor.execute("SELECT MIN(fecha), MAX(fecha) FROM fact_carreras")
        min_fecha, max_fecha = cursor.fetchone()
        print(f"\n   Período: {min_fecha} a {max_fecha}")
    except:
        pass
    
    # Carreras pendientes
    try:
        cursor.execute("""
            SELECT COUNT(DISTINCT fp.carrera_id) 
            FROM fact_participaciones fp 
            WHERE fp.resultado_final IS NULL
        """)
        pending = cursor.fetchone()[0]
        print(f"   Carreras sin resultado: {pending}")
    except:
        pass
    
    conn.close()


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='🏇 Pista Inteligente - Sistema de Predicción de Carreras',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # init-db
    p_init = subparsers.add_parser('init-db', help='Crear nueva BD con esquema 3FN')
    p_init.add_argument('--db-path', help='Ruta de la nueva BD')
    
    # migrate
    p_migrate = subparsers.add_parser('migrate', help='Migrar desde BD legacy')
    p_migrate.add_argument('--legacy-db', required=True, help='Ruta BD legacy')
    p_migrate.add_argument('--db-path', help='Ruta BD destino')
    
    # etl
    p_etl = subparsers.add_parser('etl', help='Procesar archivo CSV')
    p_etl.add_argument('--csv', required=True, help='Ruta al CSV')
    p_etl.add_argument('--db-path', help='Ruta BD')
    
    # predict
    p_predict = subparsers.add_parser('predict', help='Generar predicciones')
    p_predict.add_argument('--db-path', help='Ruta BD')
    p_predict.add_argument('--output', help='Directorio de salida')
    p_predict.add_argument('--model', choices=['xgboost', 'lightgbm', 'sklearn'],
                          default='xgboost', help='Tipo de modelo')
    
    # full
    p_full = subparsers.add_parser('full', help='Pipeline completo')
    p_full.add_argument('--legacy-db', help='Ruta BD legacy (opcional)')
    p_full.add_argument('--db-path', help='Ruta BD')
    p_full.add_argument('--output', help='Directorio de salida')
    p_full.add_argument('--model', default='xgboost')
    
    # stats
    p_stats = subparsers.add_parser('stats', help='Ver estadísticas de BD')
    p_stats.add_argument('--db-path', help='Ruta BD')
    
    args = parser.parse_args()
    
    # Crear directorio de logs
    Path('logs').mkdir(exist_ok=True)
    
    # Ejecutar comando
    commands = {
        'init-db': cmd_init_db,
        'migrate': cmd_migrate,
        'etl': cmd_etl,
        'predict': cmd_predict,
        'full': cmd_full,
        'stats': cmd_stats
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()