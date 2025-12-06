import json
import pandas as pd
from pathlib import Path

# Leer CSV
csv_path = Path("exports/raw/PROGRAMA_CHC_2025-12-05.csv")
df_csv = pd.read_csv(csv_path)
df_csv_c1 = df_csv[df_csv['Carrera'] == 1].sort_values('Numero')

# Leer JSON de predicciones
pred_path = Path("app/ml/output/predicciones_detalle.json")
with open(pred_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Encontrar la carrera 1 del CHC 2025-12-05
carrera_1_pred = None
for pred in data['predicciones']:
    if ('Club Hípico' in pred.get('hipodromo', '') and 
        '2025-12-05' in str(pred.get('fecha', '')) and
        pred.get('nro_carrera') == 1):
        carrera_1_pred = pred
        break

print("=" * 80)
print("COMPARACIÓN: CARRERA 1 - CHC 2025-12-05")
print("=" * 80)

print("\n📋 CABALLOS EN CSV (orden de partidor):")
print("-" * 80)
caballos_csv = set()
for idx, row in df_csv_c1.iterrows():
    print(f"  {row['Numero']:2d}. {row['Ejemplar']:<20s} | {row['Jinete']}")
    caballos_csv.add(row['Ejemplar'].strip().upper())

print(f"\nTotal en CSV: {len(caballos_csv)} caballos")

if carrera_1_pred:
    print("\n🤖 CABALLOS EN PREDICCIONES (orden de probabilidad IA):")
    print("-" * 80)
    
    detalle = carrera_1_pred.get('detalle', carrera_1_pred.get('predicciones', []))
    caballos_pred = set()
    
    for i, pick in enumerate(detalle, 1):
        cab = pick.get('caballo', '')
        jin = pick.get('jinete', '')
        prob = pick.get('probabilidad', 0)
        print(f"  {i:2d}. {cab:<20s} | {jin:<25s} | {prob}% prob")
        caballos_pred.add(cab.strip().upper())
    
    print(f"\nTotal en predicciones: {len(caballos_pred)} caballos")
    
    # Verificar coincidencias
    print("\n" + "=" * 80)
    print("ANÁLISIS DE COINCIDENCIAS")
    print("=" * 80)
    
    en_csv_no_pred = caballos_csv - caballos_pred
    en_pred_no_csv = caballos_pred - caballos_csv
    coinciden = caballos_csv & caballos_pred
    
    print(f"\n✅ Caballos que coinciden: {len(coinciden)}")
    
    if en_csv_no_pred:
        print(f"\n❌ En CSV pero NO en predicciones ({len(en_csv_no_pred)}):")
        for cab in sorted(en_csv_no_pred):
            print(f"   - {cab}")
    
    if en_pred_no_csv:
        print(f"\n⚠️  En predicciones pero NO en CSV ({len(en_pred_no_csv)}):")
        for cab in sorted(en_pred_no_csv):
            print(f"   - {cab}")
    
    if len(coinciden) == len(caballos_csv) == len(caballos_pred):
        print("\n✅ ¡PERFECTO! Todos los caballos coinciden.")
        print("   La diferencia que ve el usuario es solo el ORDEN:")
        print("   - CSV: Orden por número de partidor")
        print("   - Vista: Orden por probabilidad de ganar (ranking IA)")
    else:
        print("\n❌ PROBLEMA: Hay diferencias en los caballos.")
else:
    print("\n❌ No se encontró la carrera 1 en las predicciones")
