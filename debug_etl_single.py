from app.etl.etl_pipeline import ETLPipeline, ETLConfig
import logging

logging.basicConfig(level=logging.INFO)

config = ETLConfig(db_path='data/db/hipica_3fn.db')
etl = ETLPipeline(config)

try:
    result = etl.process_csv('exports/raw/PROGRAMA_CHC_2025-12-07.csv')
    print("Resultado:", result)
except Exception as e:
    print("ERROR FATAL:", e)
    import traceback
    traceback.print_exc()
