"""
Script simple para corregir códigos de hipódromos
"""
import sqlite3

DB_PATH = "data/db/hipica_3fn.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=" * 80)
print("CORRECCIÓN DE CÓDIGOS DE HIPÓDROMOS")
print("=" * 80)

# Ver estado actual
print("\nEstado actual:")
cursor.execute("SELECT id, codigo, nombre FROM dim_hipodromos")
for row in cursor.fetchall():
    print(f"  ID {row[0]}: código='{row[1]}', nombre='{row[2]}'")

# Actualizar CHS a CHC si existe
cursor.execute("SELECT id FROM dim_hipodromos WHERE codigo = 'CHS'")
if cursor.fetchone():
    print("\n⚠️ Encontrado código 'CHS', actualizando a 'CHC'...")
    cursor.execute("UPDATE dim_hipodromos SET codigo = 'CHC', nombre = 'Club Hípico de Santiago' WHERE codigo = 'CHS'")
    print("✅ Actualizado")
else:
    print("\n✅ No se encontró 'CHS'")
    # Verificar si existe CHC
    cursor.execute("SELECT id FROM dim_hipodromos WHERE codigo = 'CHC'")
    if not cursor.fetchone():
        print("⚠️ Creando registro CHC...")
        cursor.execute("INSERT INTO dim_hipodromos (codigo, nombre) VALUES ('CHC', 'Club Hípico de Santiago')")
        print("✅ Creado")

# Actualizar nombres NULL
print("\nActualizando nombres NULL...")
cursor.execute("UPDATE dim_hipodromos SET nombre = 'Hipódromo Chile' WHERE codigo = 'HC' AND (nombre IS NULL OR nombre = '')")
cursor.execute("UPDATE dim_hipodromos SET nombre = 'Club Hípico de Santiago' WHERE codigo = 'CHC' AND (nombre IS NULL OR nombre = '')")
print("✅ Nombres actualizados")

conn.commit()

# Ver estado final
print("\nEstado final:")
cursor.execute("SELECT id, codigo, nombre FROM dim_hipodromos")
for row in cursor.fetchall():
    print(f"  ID {row[0]}: código='{row[1]}', nombre='{row[2]}'")

# Ver carreras por hipódromo
print("\nCarreras por hipódromo:")
cursor.execute("""
    SELECT dh.codigo, COUNT(*) as total
    FROM fact_carreras fc
    JOIN dim_hipodromos dh ON fc.hipodromo_id = dh.id
    GROUP BY dh.codigo
""")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]} carreras")

conn.close()

print("\n" + "=" * 80)
print("✅ CORRECCIÓN COMPLETADA")
print("=" * 80)
