import json

with open('data/procesados/predicciones_v2.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

races_with_many = [r for r in d['predicciones'] if len(r['predicciones']) > 4]
print(f'Carreras totales: {len(d["predicciones"])}')
print(f'Carreras con >4 caballos: {len(races_with_many)}')

if races_with_many:
    ejemplo = races_with_many[0]
    print(f'\nEjemplo: {ejemplo["hipodromo"]} C{ejemplo["nro_carrera"]} - {len(ejemplo["predicciones"])} caballos')
    print(f'Top4: {ejemplo["top4_predicho"]}')

# Ver patrones
print(f'\nPatrones Quinela (total): {len(d["patrones"]["quinelas"])} combinaciones')
primer_patron = list(d["patrones"]["quinelas"].items())[0]
print(f'Primera combinación: {primer_patron}')
