import pandas as pd
import json

# Verificar CSV
print("=" * 60)
print("VERIFICACIÓN DE PREDICCIONES CSV")
print("=" * 60)
df = pd.read_csv('output/predicciones.csv')
print("\nPrimeras 10 filas:")
print(df[['fecha', 'hipodromo', 'nro_carrera', 'caballo']].head(10))
print(f"\nTotal registros: {len(df)}")
print(f"Hipódromos únicos: {df['hipodromo'].unique()}")
print(f"Registros con hipodromo vacío: {df['hipodromo'].isna().sum()}")

# Verificar JSON
print("\n" + "=" * 60)
print("VERIFICACIÓN DE PREDICCIONES JSON")
print("=" * 60)
with open('output/predicciones_detalle.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"\nTotal de carreras predichas: {len(data['predicciones'])}")
print("\nPrimeras 3 predicciones:")
for i, pred in enumerate(data['predicciones'][:3]):
    print(f"\n{i+1}. Fecha: {pred['fecha']}, Hipódromo: {pred['hipodromo']}, Carrera: {pred['nro_carrera']}")
    print(f"   Top 4: {', '.join(pred['top4'])}")
