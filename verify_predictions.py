import pandas as pd

df = pd.read_csv('output/predicciones.csv')

print("=" * 60)
print("VERIFICACIÓN DE CÓDIGOS DE HIPÓDROMOS EN PREDICCIONES")
print("=" * 60)

print("\nHipódromos en predicciones:")
print(df['hipodromo'].value_counts())

print(f"\nTotal registros: {len(df)}")
print(f"Registros con CHC: {(df['hipodromo'] == 'CHC').sum()}")
print(f"Registros con HC: {(df['hipodromo'] == 'HC').sum()}")

print("\nPrimeras 5 predicciones:")
print(df[['fecha', 'hipodromo', 'nro_carrera', 'caballo']].head())

print("\n" + "=" * 60)
print("✅ VERIFICACIÓN COMPLETADA")
print("=" * 60)
