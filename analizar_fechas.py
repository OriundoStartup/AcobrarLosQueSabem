import json
from pathlib import Path
from collections import defaultdict

# Leer el archivo de predicciones
pred_path = Path("app/ml/output/predicciones_detalle.json")

if pred_path.exists():
    with open(pred_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("ANÁLISIS DE FECHAS EN PREDICCIONES")
    print("=" * 80)
    
    if 'predicciones' in data:
        # Agrupar por fecha
        por_fecha = defaultdict(list)
        
        for pred in data['predicciones']:
            fecha = pred.get('fecha', 'N/A')
            hip = pred.get('hipodromo', 'N/A')
            carrera = pred.get('nro_carrera', '?')
            n_caballos = len(pred.get('detalle', pred.get('predicciones', [])))
            
            por_fecha[fecha].append({
                'hipodromo': hip,
                'carrera': carrera,
                'n_caballos': n_caballos
            })
        
        # Mostrar resumen por fecha
        print(f"\nTotal de fechas diferentes: {len(por_fecha)}")
        print("\nDesglose por fecha:")
        print("-" * 80)
        
        for fecha in sorted(por_fecha.keys()):
            carreras = por_fecha[fecha]
            total_carreras = len(carreras)
            
            # Contar por hipódromo
            hipodromos = defaultdict(int)
            for c in carreras:
                hipodromos[c['hipodromo']] += 1
            
            print(f"\n📅 FECHA: {fecha}")
            print(f"   Total carreras: {total_carreras}")
            for hip, count in hipodromos.items():
                print(f"   - {hip}: {count} carreras")
            
            # Mostrar detalles solo para 2025-12-05
            if '2025-12-05' in fecha:
                print(f"\n   🔍 DETALLE CARRERAS DEL 05-12-2025:")
                for c in carreras:
                    print(f"      Carrera {c['carrera']:2d}: {c['hipodromo']:<30s} ({c['n_caballos']} caballos)")
        
        # Verificar si hay fechas pasadas (antes de hoy)
        print("\n" + "=" * 80)
        print("VERIFICACIÓN DE FECHAS PASADAS")
        print("=" * 80)
        
        from datetime import datetime
        hoy = datetime.now().date()
        
        fechas_pasadas = []
        fechas_futuras = []
        
        for fecha in sorted(por_fecha.keys()):
            try:
                fecha_obj = datetime.strptime(fecha, '%Y-%m-%d').date()
                if fecha_obj < hoy:
                    fechas_pasadas.append((fecha, len(por_fecha[fecha])))
                else:
                    fechas_futuras.append((fecha, len(por_fecha[fecha])))
            except:
                pass
        
        if fechas_pasadas:
            print(f"\n⚠️  FECHAS PASADAS (pueden ser resultados):")
            for fecha, count in fechas_pasadas:
                print(f"   - {fecha}: {count} carreras")
        else:
            print("\n✅ No hay fechas pasadas")
        
        if fechas_futuras:
            print(f"\n✅ FECHAS FUTURAS/HOY (programas):")
            for fecha, count in fechas_futuras:
                print(f"   - {fecha}: {count} carreras")
        
    else:
        print("\n❌ No hay clave 'predicciones' en el JSON")
        
else:
    print("\n❌ ARCHIVO NO EXISTE: app/ml/output/predicciones_detalle.json")
