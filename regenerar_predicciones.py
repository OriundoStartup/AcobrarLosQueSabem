"""
Script para regenerar predicciones con la BD limpia.
"""

import sys
sys.path.append('app')

from ml.predictor import RacePredictor, MLConfig

print("=" * 80)
print("REGENERANDO PREDICCIONES CON BD LIMPIA")
print("=" * 80)

# Configurar y entrenar
config = MLConfig(db_path="data/db/hipica_3fn.db")
predictor = RacePredictor(config)

print("\n1️⃣ Entrenando modelo...")
predictor.train()

print("\n2️⃣ Generando predicciones...")
predictions = predictor.predict()

print(f"\n✅ Predicciones generadas para {len(predictions)} carreras")

# Guardar
print("\n3️⃣ Guardando predicciones...")
predictor.save_predictions(predictions)

print("\n✅ COMPLETADO")
print("\n📋 Predicciones guardadas en: app/ml/output/predicciones_detalle.json")
print("   Recarga la vista Streamlit para ver cambios")
