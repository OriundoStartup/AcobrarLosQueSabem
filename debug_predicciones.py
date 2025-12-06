import json
from pathlib import Path

# Leer el archivo de predicciones si existe
pred_path = Path("app/ml/output/predicciones_detalle.json")

if pred_path.exists():
    with open(pred_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("INFORMACIÓN DE PREDICCIONES")
    print("=" * 80)
    
    if 'metadata' in data:
        meta = data['metadata']
        print(f"\nFecha generación: {meta.get('fecha_generacion', 'N/A')}")
        
        if 'metricas_entrenamiento' in meta or 'metricas' in meta:
            metricas = meta.get('metricas_entrenamiento', meta.get('metricas', {}))
            print(f"RMSE Test: {metricas.get('RMSE_Test', metricas.get('rmse_test', 'N/A'))}")
            print(f"R2 Test: {metricas.get('R2_Test', metricas.get('r2_test', 'N/A'))}")
    
    print("\n" + "=" * 80)
    print("CARRERAS PREDICHAS")
    print("=" * 80)
    
    if 'predicciones' in data:
        print(f"\nTotal carreras predichas: {len(data['predicciones'])}")
        print("\nLista de carreras:")
        print("-" * 80)
        
        for i, pred in enumerate(data['predicciones'], 1):
            hip = pred.get('hipodromo', 'N/A')
            fecha = pred.get('fecha', 'N/A')
            nro = pred.get('nro_carrera', '?')
            detalle = pred.get('detalle', pred.get('predicciones', []))
            n_picks = len(detalle)
            
            print(f"{i}. {fecha} | {hip} | Carrera {nro} | {n_picks} caballos")
            
            # Si es del CHC y 05-12-2025, mostrar detalles
            if '05' in str(fecha) and 'CHC' in hip:
                print(f"   >>> Detalles de top 5:")
                for j, pick in enumerate(detalle[:5], 1):
                    cab = pick.get('caballo', 'N/A')
                    jin = pick.get('jinete', 'N/A')
                    prob = pick.get('probabilidad', 0)
                    print(f"       {j}. {cab} ({jin}) - {prob}%")
        
        # Buscar específicamente carreras del CHC 05-12-2025
        print("\n" + "=" * 80)
        print("FILTRANDO: CHC - 2025-12-05")
        print("=" * 80)
        
        chc_05_12 = [
            p for p in data['predicciones']
            if ('CHC' in p.get('hipodromo', '') or 'Club' in p.get('hipodromo', ''))
            and '2025-12-05' in str(p.get('fecha', ''))
        ]
        
        print(f"\nEncontradas: {len(chc_05_12)} carreras")
        
        if chc_05_12:
            for pred in chc_05_12:
                print(f"\nCarrera {pred.get('nro_carrera')}: {pred.get('hipodromo')} - {pred.get('fecha')}")
                detalle = pred.get('detalle', pred.get('predicciones', []))
                print(f"Caballos predichos: {len(detalle)}")
                
                # Mostrar los primeros 3
                for i, pick in enumerate(detalle[:3], 1):
                    print(f"  {i}. {pick.get('caballo')} - {pick.get('jinete')}")
        else:
            print("\n⚠️ NO SE ENCONTRARON PREDICCIONES PARA CHC 2025-12-05")
    
    else:
        print("\n❌ No hay clave 'predicciones' en el JSON")
        
else:
    print("\n❌ ARCHIVO NO EXISTE: app/ml/output/predicciones_detalle.json")
    print("\nEl pipeline de predicciones NO ha sido ejecutado o falló.")
