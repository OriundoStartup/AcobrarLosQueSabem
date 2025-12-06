"""
Script de diagnóstico para revisar cómo se generan y muestran los patrones repetidos.
"""

import json
import sys
from pathlib import Path
from collections import Counter

print("=" * 80)
print("ANÁLISIS DE PATRONES - DIAGNÓSTICO COMPLETO")
print("=" * 80)

# 1. VERIFICAR ARCHIVO DE PREDICCIONES
json_path = Path("app/output/predicciones_detalle.json")
if not json_path.exists():
    print(f"\n❌ ERROR: No se encuentra el archivo: {json_path}")
    sys.exit(1)

print(f"\n✅ Archivo encontrado: {json_path}")

# 2. CARGAR Y ANALIZAR DATOS
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"\n📊 Estructura del JSON:")
print(f"   - Claves principales: {list(data.keys())}")

if 'predicciones' in data:
    print(f"   - Total predicciones: {len(data['predicciones'])}")
else:
    print("   ❌ No hay clave 'predicciones'")
    sys.exit(1)

# 3. ANALIZAR CADA PREDICCIÓN Y SU top4_predicho
print(f"\n{'='*80}")
print("ANÁLISIS DETALLADO DE PREDICCIONES")
print(f"{'='*80}")

quinelas_generadas = []
trifectas_generadas = []
superfectas_generadas = []
caballos_vacios = 0
total_participantes = 0

for idx, pred in enumerate(data['predicciones'], 1):
    fecha = pred.get('fecha', 'N/A')
    hipodromo = pred.get('hipodromo', 'N/A')
    nro = pred.get('nro_carrera', '?')
    top4 = pred.get('top4', pred.get('top4_predicho', []))
    detalle = pred.get('predicciones', pred.get('detalle', []))
    
    print(f"\n{idx}. {fecha} | {hipodromo} | Carrera {nro}")
    print(f"   Top4: {top4}")
    print(f"   Cantidad en top4: {len(top4)}")
    print(f"   Total participantes: {len(detalle)}")
    
    total_participantes += len(detalle)
    
    # Verificar si hay nombres vacíos
    if any(not c or c.strip() == '' for c in top4):
        caballos_vacios += 1
        print(f"   ⚠️  PROBLEMA: Hay nombres vacíos en top4")
        
        # Mostrar los primeros 3 del detalle para comparar
        print(f"   Top 3 del detalle:")
        for i, p in enumerate(detalle[:3], 1):
            cab = p.get('caballo', '')
            jin = p.get('jinete', '')
            prob = p.get('probabilidad', 0)
            print(f"      {i}. '{cab}' ({jin}) - {prob}%")
    
    # Generar patrones según la lógica del método detect_patterns
    if len(top4) >= 2:
        quinela = tuple(sorted(top4[:2]))
        quinelas_generadas.append(quinela)
        print(f"   Quinela generada: {quinela}")
    
    if len(top4) >= 3:
        trifecta = tuple(top4[:3])
        trifectas_generadas.append(trifecta)
        print(f"   Trifecta generada: {trifecta}")
    
    if len(top4) >= 4:
        superfecta = tuple(top4[:4])
        superfectas_generadas.append(superfecta)
        print(f"   Superfecta generada: {superfecta}")

# 4. CONTEO DE PATRONES (como lo hace detect_patterns)
print(f"\n{'='*80}")
print("ANÁLISIS DE PATRONES REPETIDOS (min_count=2)")
print(f"{'='*80}")

def analizar_patrones(nombre, combos, min_count=2):
    print(f"\n{nombre}:")
    counter = Counter(combos)
    print(f"   Total combinaciones: {len(combos)}")
    print(f"   Combinaciones únicas: {len(counter)}")
    
    repetidas = {str(k): v for k, v in counter.items() if v >= min_count}
    
    if repetidas:
        print(f"   ⚠️  PATRONES REPETIDOS ENCONTRADOS:")
        for combo, count in sorted(repetidas.items(), key=lambda x: -x[1]):
            print(f"      {combo}: {count} veces")
    else:
        print(f"   ✅ Sin patrones repetidos (todas son únicas o solo 1 ocurrencia)")
    
    # Mostrar las 5 más comunes (aunque no sean repetidas)
    print(f"   Top 5 combinaciones más comunes:")
    for combo, count in counter.most_common(5):
        print(f"      {combo}: {count} veces")
    
    return repetidas

patrones_quinela = analizar_patrones("QUINELAS", quinelas_generadas)
patrones_trifecta = analizar_patrones("TRIFECTAS", trifectas_generadas)
patrones_superfecta = analizar_patrones("SUPERFECTAS", superfectas_generadas)

# 5. COMPARAR CON LO QUE ESTÁ EN EL JSON
print(f"\n{'='*80}")
print("COMPARACIÓN: GENERADO vs ALMACENADO EN JSON")
print(f"{'='*80}")

if 'patrones' in data:
    patrones_json = data['patrones']
    print(f"\nPatrones en JSON: {list(patrones_json.keys())}")
    
    # Quinelas
    print(f"\n🧩 QUINELAS:")
    print(f"   Generado ahora: {patrones_quinela}")
    print(f"   En JSON: {patrones_json.get('quinelas', {})}")
    print(f"   ¿COINCIDEN? {'✅' if patrones_quinela == patrones_json.get('quinelas', {}) else '❌'}")
    
    # Trifectas
    print(f"\n🎯 TRIFECTAS/TRIDECTAS:")
    json_trifectas = patrones_json.get('trifectas', patrones_json.get('tridectas', {}))
    print(f"   Generado ahora: {patrones_trifecta}")
    print(f"   En JSON: {json_trifectas}")
    print(f"   ¿COINCIDEN? {'✅' if patrones_trifecta == json_trifectas else '❌'}")
    
    # Superfectas
    print(f"\n🔥 SUPERFECTAS:")
    print(f"   Generado ahora: {patrones_superfecta}")
    print(f"   En JSON: {patrones_json.get('superfectas', {})}")
    print(f"   ¿COINCIDEN? {'✅' if patrones_superfecta == patrones_json.get('superfectas', {}) else '❌'}")
else:
    print("\n❌ No hay clave 'patrones' en el JSON")

# 6. RESUMEN FINAL
print(f"\n{'='*80}")
print("RESUMEN FINAL")
print(f"{'='*80}")
print(f"Total carreras analizadas: {len(data['predicciones'])}")
print(f"Total participantes: {total_participantes}")
print(f"Carreras con nombres vacíos en top4: {caballos_vacios}")
print(f"\nQuinelas únicas: {len(set(quinelas_generadas))}")
print(f"Trifectas únicas: {len(set(trifectas_generadas))}")
print(f"Superfectas únicas: {len(set(superfectas_generadas))}")

# Verificar si el problema es que hay demasiados nombres vacíos
if caballos_vacios > 0:
    print(f"\n⚠️  ADVERTENCIA: {caballos_vacios} carreras tienen nombres de caballos vacíos")
    print(f"   Esto podría estar causando el patrón repetido ('', '', ...) que se muestra")
    print(f"   Se debe corregir el método que genera las predicciones para asegurar")
    print(f"   que los nombres de caballos se extraen correctamente del DataFrame.")
    
print(f"\n{'='*80}")
print("FIN DEL ANÁLISIS")
print(f"{'='*80}")
