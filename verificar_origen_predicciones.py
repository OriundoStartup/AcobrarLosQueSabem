import sqlite3
import pandas as pd
from pathlib import Path

# Conectar a BD
db_path = Path("data/db/hipica_3fn.db")
conn = sqlite3.connect(str(db_path))

print("=" * 80)
print("ANÁLISIS: PARTICIPACIONES POR TIPO (CON Y SIN RESULTADO)")
print("=" * 80)

# Query SIN resultado (programas)
query_sin_resultado = """
SELECT 
    c.fecha,
    h.codigo as hipodromo,
    c.nro_carrera,
    COUNT(p.id) as participantes
FROM fact_participaciones p
JOIN fact_carreras c ON p.carrera_id = c.id
JOIN dim_hipodromos h ON c.hipodromo_id = h.id
WHERE p.resultado_final IS NULL
GROUP BY c.fecha, h.codigo, c.nro_carrera
ORDER BY c.fecha DESC, c.nro_carrera
"""

# Query CON resultado (resultados)
query_con_resultado = """
SELECT 
    c.fecha,
    h.codigo as hipodromo,
    c.nro_carrera,
    COUNT(p.id) as participantes
FROM fact_participaciones p
JOIN fact_carreras c ON p.carrera_id = c.id
JOIN dim_hipodromos h ON c.hipodromo_id = h.id
WHERE p.resultado_final IS NOT NULL
GROUP BY c.fecha, h.codigo, c.nro_carrera
ORDER BY c.fecha DESC, c.nro_carrera
"""

df_sin = pd.read_sql_query(query_sin_resultado, conn)
df_con = pd.read_sql_query(query_con_resultado, conn)

print("\n📄 CARRERAS SIN RESULTADO (resultado_final IS NULL) - PROGRAMAS")
print("=" * 80)
print(f"Total: {len(df_sin)} carreras\n")

for fecha in sorted(df_sin['fecha'].unique(), reverse=True):
    df_fecha = df_sin[df_sin['fecha'] == fecha]
    print(f"📅 {fecha}: {len(df_fecha)} carreras")
    for hip in df_fecha['hipodromo'].unique():
        count = len(df_fecha[df_fecha['hipodromo'] == hip])
        print(f"   {hip}: {count} carreras")

print("\n📊 CARRERAS CON RESULTADO (resultado_final NOT NULL) - RESULTADOS")
print("=" * 80)
print(f"Total: {len(df_con)} carreras\n")

for fecha in sorted(df_con['fecha'].unique(), reverse=True)[:10]:  # Solo últimas 10 fechas
    df_fecha = df_con[df_con['fecha'] == fecha]
    print(f"📅 {fecha}: {len(df_fecha)} carreras")
    for hip in df_fecha['hipodromo'].unique():
        count = len(df_fecha[df_fecha['hipodromo'] == hip])
        print(f"   {hip}: {count} carreras")

# Verificar solapamiento
print("\n" + "=" * 80)
print("VERIFICACIÓN DE FECHAS DUPLICADAS")
print("=" * 80)

fechas_sin = set(df_sin['fecha'].unique())
fechas_con = set(df_con['fecha'].unique())
fechas_ambas = fechas_sin & fechas_con

if fechas_ambas:
    print(f"\n⚠️  PROBLEMA: Hay {len(fechas_ambas)} fechas con AMBOS tipos de registros:")
    for fecha in sorted(fechas_ambas, reverse=True):
        sin_car = len(df_sin[df_sin['fecha'] == fecha])
        con_car = len(df_con[df_con['fecha'] == fecha])
        print(f"   {fecha}: {sin_car} sin resultado, {con_car} con resultado")
        
        # Verificar si es la misma carrera
        carreras_sin = set(df_sin[df_sin['fecha'] == fecha]['nro_carrera'])
        carreras_con = set(df_con[df_con['fecha'] == fecha]['nro_carrera'])
        carreras_duplicadas = carreras_sin & carreras_con
        
        if carreras_duplicadas:
            print(f"      ⚠️  Carreras duplicadas: {sorted(carreras_duplicadas)}")
            print(f"      Esto indica que la misma carrera tiene datos CON y SIN resultado")
else:
    print("\n✅ Correcto: No hay fechas con ambos tipos de registros")
    print("   Las carreras con resultado y sin resultado están bien separadas")

# Verificar archivos CSV en la carpeta
print("\n" + "=" * 80)
print("ARCHIVOS CSV DISPONIBLES")
print("=" * 80)

csv_dir = Path("exports/raw")
if csv_dir.exists():
    programas = sorted([f.name for f in csv_dir.glob("PROGRAMA*.csv")])
    resultados = sorted([f.name for f in csv_dir.glob("resul*.csv")])
    
    print(f"\n📄 PROGRAMA ({len(programas)}):")
    for f in programas:
        print(f"   {f}")
    
    print(f"\n📊 RESUL ({len(resultados)}):")
    for f in resultados:
        print(f"   {f}")
    
    # Extraer fechas de archivos
    import re
    
    fechas_prog = set()
    for f in programas:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', f)
        if match:
            fechas_prog.add(match.group(1))
    
    fechas_resul = set()
    for f in resultados:
        match = re.search(r'(\d{2}-\d{2}-\d{4})', f)
        if match:
            # Convertir DD-MM-YYYY a YYYY-MM-DD
            parts = match.group(1).split('-')
            fecha = f"{parts[2]}-{parts[1]}-{parts[0]}"
            fechas_resul.add(fecha)
    
    print(f"\n📊 Análisis:")
    print(f"   Fechas en BD sin resultado: {sorted(fechas_sin)}")
    print(f"   Fechas en archivos PROGRAMA: {sorted(fechas_prog)}")
    print(f"   Fechas en archivos RESUL: {sorted(fechas_resul)}")
    
    # Verificar correspondencia
    if fechas_sin == fechas_prog:
        print(f"\n   ✅ Las fechas sin resultado coinciden con archivos PROGRAMA")
    else:
        print(f"\n   ⚠️  Discrepancia entre BD y archivos PROGRAMA")

conn.close()
