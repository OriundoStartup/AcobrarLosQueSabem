import json

with open('output/predicciones_detalle.json', encoding='utf-8') as f:
    data = json.load(f)

pred = data['predicciones'][1]
print(f"Carrera 2:")
print(f"  Hipódromo: {pred['hipodromo']}")
print(f"  Número: {pred['nro_carrera']}")
print(f"  Confianza: {pred['confianza']}%")
print(f"  Detalles: {len(pred['detalle'])} caballos")
print(f"\nPrimeros 3 caballos:")
for i, p in enumerate(pred['detalle'][:3], 1):
    print(f"  {i}. {p['caballo']} - {p['jinete']} - {p['probabilidad']}%")
