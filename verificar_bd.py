import sqlite3

conn = sqlite3.connect('data/db/hipica_3fn.db')
cur = conn.cursor()

# Verificar carreras de mañana
cur.execute("SELECT COUNT(*) FROM fact_carreras WHERE fecha = '2025-12-07'")
print(f"Carreras para 2025-12-07: {cur.fetchone()[0]}")

# Ver todas las fechas
cur.execute("""
    SELECT fecha, COUNT(*) 
    FROM fact_carreras 
    GROUP BY fecha 
    ORDER BY fecha DESC
    LIMIT 10
""")
print("\nÚltimas 10 fechas en BD:")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]} carreras")

conn.close()
