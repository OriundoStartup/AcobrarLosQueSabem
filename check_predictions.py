import json

with open('output/predicciones_detalle.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total predicciones: {len(data['predicciones'])}")
print("\nPrimera predicción:")
p = data['predicciones'][0]
print(f"  Fecha: {p['fecha']}")
print(f"  Hipódromo: {p['hipodromo']}")
print(f"  Carrera: {p['nro_carrera']}")
print(f"  Top 4: {p['top4']}")

print("\nEstructura de una predicción:")
print(f"  Keys: {list(p.keys())}")
