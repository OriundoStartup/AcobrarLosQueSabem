from app.etl.detector import CSVDetector
import os

files = [
    'PROGRAMA_CHC_2025-12-07.csv', 
    'RESULTADOS_CHC_2025-12-05.csv',
    'PROGRAMA_HC_2025-12-06.csv'
]

print("Probando detección de archivos:")
print("=" * 60)

for f in files:
    path = f'exports/raw/{f}'
    if os.path.exists(path):
        result = CSVDetector.detect(path)
        print(f"{f}:")
        print(f"  Tipo: {result.csv_type.value}")
        print(f"  Hipódromo: {result.hipodromo.value}")
        print(f"  Fecha: {result.fecha}")
        print(f"  Confianza: {result.confidence:.0%}")
        print()
    else:
        print(f"{f}: NO EXISTE")
