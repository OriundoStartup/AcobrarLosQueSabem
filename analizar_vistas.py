import sqlite3
from pathlib import Path

db_path = Path("data/db/hipica_3fn.db")
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

print("=" * 80)
print("ANÁLISIS DE VISTAS SQL EN LA BASE DE DATOS")
print("=" * 80)

# Listar todas las vistas
cursor.execute("""
    SELECT name, sql 
    FROM sqlite_master 
    WHERE type='view'
    ORDER BY name
""")

vistas = cursor.fetchall()

if vistas:
    print(f"\nTotal de vistas encontradas: {len(vistas)}\n")
    
    for nombre, sql in vistas:
        print("-" * 80)
        print(f"📊 VISTA: {nombre}")
        print("-" * 80)
        print("\nDefinición SQL:")
        print(sql)
        print()
        
        # Intentar consultar la vista para verificar si funciona
        try:
            cursor.execute(f"SELECT * FROM {nombre} LIMIT 1")
            row = cursor.fetchone()
            if row:
                print(f"✅ Vista funcional - Retorna {len(row)} columnas")
            else:
                print("⚠️  Vista vacía (no retorna datos)")
        except Exception as e:
            print(f"❌ ERROR al consultar vista: {e}")
        
        print()
else:
    print("\n⚠️  No se encontraron vistas en la base de datos")

# Verificar si existe v_ml_training_data específicamente
print("=" * 80)
print("VERIFICACIÓN DE v_ml_training_data")
print("=" * 80)

cursor.execute("""
    SELECT sql 
    FROM sqlite_master 
    WHERE type='view' AND name='v_ml_training_data'
""")

vista_ml = cursor.fetchone()

if vista_ml:
    print("\n✅ La vista v_ml_training_data EXISTE")
    print("\nDefinición:")
    print(vista_ml[0])
    
    # Verificar que funcione
    try:
        cursor.execute("SELECT COUNT(*) FROM v_ml_training_data")
        count = cursor.fetchone()[0]
        print(f"\n✅ Vista funcional - Contiene {count} registros")
        
        # Mostrar columnas
        cursor.execute("PRAGMA table_info(v_ml_training_data)")
        columnas = cursor.fetchall()
        print(f"\nColumnas ({len(columnas)}):")
        for col in columnas:
            print(f"   - {col[1]} ({col[2]})")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
else:
    print("\n❌ La vista v_ml_training_data NO EXISTE")
    print("   El predictor ML no podrá entrenar sin esta vista")

conn.close()
