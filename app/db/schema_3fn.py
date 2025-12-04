"""
================================================================================
PISTA INTELIGENTE - ESQUEMA DE BASE DE DATOS 3FN
================================================================================
Diseño normalizado a Tercera Forma Normal (3FN) para el sistema de predicción
de carreras de caballos.

Principios aplicados:
- 1FN: Valores atómicos, sin grupos repetitivos
- 2FN: Dependencia funcional completa de la clave primaria
- 3FN: Eliminación de dependencias transitivas

Autor: Data Engineering Team - Pista Inteligente
================================================================================
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# SQL DDL - ESQUEMA 3FN
# ==============================================================================

SCHEMA_3FN = """
-- ============================================================================
-- TABLAS DE DIMENSIONES (CATÁLOGOS)
-- ============================================================================

-- Hipódromos: Catálogo de recintos
CREATE TABLE IF NOT EXISTS dim_hipodromos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo TEXT UNIQUE NOT NULL,           -- 'HC', 'CHC', 'VSC'
    nombre TEXT NOT NULL,                   -- 'Hipódromo Chile'
    ciudad TEXT,
    pais TEXT DEFAULT 'Chile',
    latitud REAL,
    longitud REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Studs/Haras: Catálogo de propietarios/criadores
CREATE TABLE IF NOT EXISTS dim_studs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Jinetes: Catálogo de jockeys
CREATE TABLE IF NOT EXISTS dim_jinetes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT UNIQUE NOT NULL,
    nacionalidad TEXT,
    peso_base REAL,                         -- Peso promedio histórico
    licencia_activa BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Entrenadores: Catálogo de preparadores
CREATE TABLE IF NOT EXISTS dim_entrenadores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Caballos: Entidad principal de ejemplares
CREATE TABLE IF NOT EXISTS dim_caballos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre TEXT UNIQUE NOT NULL,
    sexo TEXT CHECK(sexo IN ('M', 'H', 'C')),  -- Macho, Hembra, Castrado
    pelaje TEXT,
    fecha_nacimiento DATE,
    pais_origen TEXT DEFAULT 'Chile',
    stud_id INTEGER REFERENCES dim_studs(id),
    padre TEXT,
    madre TEXT,
    abuelo_materno TEXT,
    activo BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Superficies: Tipos de pista
CREATE TABLE IF NOT EXISTS dim_superficies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo TEXT UNIQUE NOT NULL,            -- 'arena', 'pasto', 'sintetico'
    descripcion TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tipos de Carrera: Clasificación por condiciones
CREATE TABLE IF NOT EXISTS dim_tipos_carrera (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo TEXT UNIQUE NOT NULL,            -- 'HANDx', 'COND', 'CLASICO', 'LISTADA'
    nombre TEXT NOT NULL,
    descripcion TEXT,
    nivel_importancia INTEGER DEFAULT 1,    -- 1-5 para rankings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLAS DE HECHOS (TRANSACCIONALES)
-- ============================================================================

-- Carreras: Evento principal
CREATE TABLE IF NOT EXISTS fact_carreras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Identificadores naturales
    fecha DATE NOT NULL,
    hipodromo_id INTEGER NOT NULL REFERENCES dim_hipodromos(id),
    nro_carrera INTEGER NOT NULL,
    
    -- Características de la carrera
    distancia_metros INTEGER NOT NULL,
    superficie_id INTEGER REFERENCES dim_superficies(id),
    tipo_carrera_id INTEGER REFERENCES dim_tipos_carrera(id),
    condicion_texto TEXT,                   -- Descripción original
    
    -- Premios
    premio_primero REAL,
    premio_segundo REAL,
    premio_tercero REAL,
    moneda TEXT DEFAULT 'CLP',
    
    -- Metadata
    hora_programada TIME,
    estado TEXT DEFAULT 'programada' CHECK(estado IN ('programada', 'corrida', 'suspendida', 'anulada')),
    
    -- ETL tracking
    source_file TEXT,
    etl_batch_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraint de unicidad natural
    UNIQUE(fecha, hipodromo_id, nro_carrera)
);

-- Participaciones: Relación caballo-carrera (Tabla pivote enriquecida)
CREATE TABLE IF NOT EXISTS fact_participaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Foreign Keys
    carrera_id INTEGER NOT NULL REFERENCES fact_carreras(id),
    caballo_id INTEGER NOT NULL REFERENCES dim_caballos(id),
    jinete_id INTEGER REFERENCES dim_jinetes(id),
    entrenador_id INTEGER REFERENCES dim_entrenadores(id),
    
    -- Datos de inscripción
    partidor INTEGER,                       -- Número de partidor
    peso_programado REAL,                   -- Peso asignado (kg)
    handicap REAL,
    edad_anos INTEGER,                      -- Edad al momento de la carrera
    
    -- Resultados (NULL si aún no corre)
    resultado_final INTEGER,                -- Posición de llegada
    tiempo_oficial TEXT,                    -- '1:23.45'
    tiempo_centesimas INTEGER,              -- 8345 (para cálculos)
    distancia_ganador TEXT,                 -- 'cabeza', '1/2 cpo', '2 cpos'
    dividendo_ganador REAL,
    dividendo_place REAL,
    
    -- Métricas calculadas (denormalizadas para performance)
    velocidad_promedio REAL,                -- metros/segundo
    
    -- ETL tracking
    source_file TEXT,
    etl_batch_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(carrera_id, caballo_id)
);

-- ============================================================================
-- TABLAS DE MÉTRICAS PRE-CALCULADAS (PARA ML)
-- ============================================================================

-- Estadísticas históricas de caballos (actualizadas por ETL)
CREATE TABLE IF NOT EXISTS agg_caballo_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caballo_id INTEGER UNIQUE NOT NULL REFERENCES dim_caballos(id),
    
    -- Conteos
    total_carreras INTEGER DEFAULT 0,
    victorias INTEGER DEFAULT 0,
    segundos INTEGER DEFAULT 0,
    terceros INTEGER DEFAULT 0,
    
    -- Promedios
    posicion_promedio REAL,
    velocidad_promedio REAL,
    
    -- Tasas
    tasa_victoria REAL,
    tasa_podio REAL,
    
    -- Rachas
    racha_actual INTEGER DEFAULT 0,         -- Carreras sin ganar (negativo) o ganando (positivo)
    mejor_racha INTEGER DEFAULT 0,
    
    -- Por distancia
    mejor_distancia INTEGER,
    victorias_cortas INTEGER DEFAULT 0,     -- < 1200m
    victorias_medias INTEGER DEFAULT 0,     -- 1200-1600m
    victorias_largas INTEGER DEFAULT 0,     -- > 1600m
    
    -- Última actividad
    ultima_carrera DATE,
    ultimo_resultado INTEGER,
    dias_sin_correr INTEGER,
    
    -- Metadata
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Estadísticas históricas de jinetes
CREATE TABLE IF NOT EXISTS agg_jinete_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    jinete_id INTEGER UNIQUE NOT NULL REFERENCES dim_jinetes(id),
    
    total_carreras INTEGER DEFAULT 0,
    victorias INTEGER DEFAULT 0,
    tasa_victoria REAL,
    posicion_promedio REAL,
    
    -- Por hipódromo (JSON para flexibilidad)
    stats_por_hipodromo TEXT,               -- JSON
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Combinaciones caballo-jinete
CREATE TABLE IF NOT EXISTS agg_combo_caballo_jinete (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    caballo_id INTEGER NOT NULL REFERENCES dim_caballos(id),
    jinete_id INTEGER NOT NULL REFERENCES dim_jinetes(id),
    
    carreras_juntos INTEGER DEFAULT 0,
    victorias_juntos INTEGER DEFAULT 0,
    tasa_victoria_combo REAL,
    
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(caballo_id, jinete_id)
);

-- ============================================================================
-- TABLA DE CONTROL ETL
-- ============================================================================

CREATE TABLE IF NOT EXISTS etl_control (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT UNIQUE NOT NULL,
    source_type TEXT NOT NULL,              -- 'csv', 'pdf', 'api'
    source_file TEXT,
    records_raw INTEGER,
    records_inserted INTEGER,
    records_updated INTEGER,
    records_rejected INTEGER,
    errors TEXT,                            -- JSON con errores
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT CHECK(status IN ('running', 'completed', 'failed', 'partial'))
);

CREATE TABLE IF NOT EXISTS etl_rejected_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    source_row INTEGER,
    raw_data TEXT,                          -- JSON del registro original
    rejection_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- ÍNDICES PARA PERFORMANCE
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_carreras_fecha ON fact_carreras(fecha);
CREATE INDEX IF NOT EXISTS idx_carreras_hipodromo ON fact_carreras(hipodromo_id);
CREATE INDEX IF NOT EXISTS idx_carreras_fecha_hip ON fact_carreras(fecha, hipodromo_id);

CREATE INDEX IF NOT EXISTS idx_participaciones_carrera ON fact_participaciones(carrera_id);
CREATE INDEX IF NOT EXISTS idx_participaciones_caballo ON fact_participaciones(caballo_id);
CREATE INDEX IF NOT EXISTS idx_participaciones_jinete ON fact_participaciones(jinete_id);
CREATE INDEX IF NOT EXISTS idx_participaciones_resultado ON fact_participaciones(resultado_final);

CREATE INDEX IF NOT EXISTS idx_caballo_nombre ON dim_caballos(nombre);
CREATE INDEX IF NOT EXISTS idx_jinete_nombre ON dim_jinetes(nombre);

-- ============================================================================
-- TRIGGERS PARA UPDATED_AT
-- ============================================================================

CREATE TRIGGER IF NOT EXISTS trg_carreras_updated 
    AFTER UPDATE ON fact_carreras
BEGIN
    UPDATE fact_carreras SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_participaciones_updated 
    AFTER UPDATE ON fact_participaciones
BEGIN
    UPDATE fact_participaciones SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ============================================================================
-- VISTAS ÚTILES
-- ============================================================================

-- Vista completa de participaciones (para consultas y ML)
CREATE VIEW IF NOT EXISTS v_participaciones_completas AS
SELECT 
    fp.id,
    fc.fecha,
    fc.nro_carrera,
    dh.nombre AS hipodromo,
    dh.codigo AS hipodromo_codigo,
    fc.distancia_metros,
    ds.codigo AS superficie,
    dtc.nombre AS tipo_carrera,
    fc.condicion_texto,
    fc.premio_primero,
    
    dc.id AS caballo_id,
    dc.nombre AS caballo,
    dc.sexo AS caballo_sexo,
    
    dj.id AS jinete_id,
    dj.nombre AS jinete,
    
    de.nombre AS entrenador,
    dst.nombre AS stud,
    
    fp.partidor,
    fp.peso_programado,
    fp.handicap,
    fp.edad_anos,
    fp.resultado_final,
    fp.tiempo_oficial,
    fp.dividendo_ganador,
    
    -- Stats pre-calculados
    acs.total_carreras AS caballo_carreras,
    acs.tasa_victoria AS caballo_tasa_victoria,
    acs.posicion_promedio AS caballo_pos_promedio,
    acs.dias_sin_correr AS caballo_dias_descanso,
    
    ajs.tasa_victoria AS jinete_tasa_victoria,
    
    acj.tasa_victoria_combo AS combo_tasa_victoria

FROM fact_participaciones fp
JOIN fact_carreras fc ON fp.carrera_id = fc.id
JOIN dim_hipodromos dh ON fc.hipodromo_id = dh.id
JOIN dim_caballos dc ON fp.caballo_id = dc.id
LEFT JOIN dim_jinetes dj ON fp.jinete_id = dj.id
LEFT JOIN dim_entrenadores de ON fp.entrenador_id = de.id
LEFT JOIN dim_studs dst ON dc.stud_id = dst.id
LEFT JOIN dim_superficies ds ON fc.superficie_id = ds.id
LEFT JOIN dim_tipos_carrera dtc ON fc.tipo_carrera_id = dtc.id
LEFT JOIN agg_caballo_stats acs ON dc.id = acs.caballo_id
LEFT JOIN agg_jinete_stats ajs ON dj.id = ajs.jinete_id
LEFT JOIN agg_combo_caballo_jinete acj ON dc.id = acj.caballo_id AND dj.id = acj.jinete_id;

-- Vista para datos de entrenamiento ML
CREATE VIEW IF NOT EXISTS v_ml_training_data AS
SELECT 
    fp.id AS participacion_id,
    fc.fecha,
    fc.hipodromo_id,
    fc.distancia_metros,
    COALESCE(fc.superficie_id, 1) AS superficie_id,
    
    fp.caballo_id,
    fp.jinete_id,
    fp.partidor,
    fp.peso_programado,
    fp.edad_anos,
    
    COALESCE(acs.total_carreras, 0) AS caballo_carreras_previas,
    COALESCE(acs.tasa_victoria, 0) AS caballo_tasa_victoria,
    COALESCE(acs.posicion_promedio, 5) AS caballo_pos_promedio,
    COALESCE(acs.dias_sin_correr, 30) AS caballo_dias_descanso,
    COALESCE(acs.racha_actual, 0) AS caballo_racha,
    
    COALESCE(ajs.tasa_victoria, 0) AS jinete_tasa_victoria,
    COALESCE(ajs.posicion_promedio, 5) AS jinete_pos_promedio,
    
    COALESCE(acj.tasa_victoria_combo, 0) AS combo_tasa_victoria,
    COALESCE(acj.carreras_juntos, 0) AS combo_carreras,
    
    -- Target
    fp.resultado_final AS target
    
FROM fact_participaciones fp
JOIN fact_carreras fc ON fp.carrera_id = fc.id
LEFT JOIN agg_caballo_stats acs ON fp.caballo_id = acs.caballo_id
LEFT JOIN agg_jinete_stats ajs ON fp.jinete_id = ajs.jinete_id
LEFT JOIN agg_combo_caballo_jinete acj ON fp.caballo_id = acj.caballo_id AND fp.jinete_id = acj.jinete_id
WHERE fp.resultado_final IS NOT NULL;
"""


class DatabaseManager:
    """
    Gestor de base de datos con soporte para migraciones y control de versiones.
    """
    
    CURRENT_VERSION = 1
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
    
    @property
    def conn(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            self._connection.row_factory = sqlite3.Row
            # Optimizaciones SQLite
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA cache_size=10000")
            self._connection.execute("PRAGMA foreign_keys=ON")
        return self._connection
    
    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def initialize_schema(self) -> bool:
        """Crea el esquema completo de la base de datos."""
        try:
            logger.info(f"Inicializando esquema 3FN en {self.db_path}")
            self.conn.executescript(SCHEMA_3FN)
            self.conn.commit()
            
            # Insertar datos de catálogo iniciales
            self._seed_catalogs()
            
            logger.info("✅ Esquema inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando esquema: {e}")
            raise
    
    def _seed_catalogs(self):
        """Inserta datos iniciales en catálogos."""
        
        # Hipódromos
        hipodromos = [
            ('HC', 'Hipódromo Chile', 'Santiago'),
            ('CHS', 'Club Hípico de Santiago', 'Santiago'),
            ('VSC', 'Valparaíso Sporting Club', 'Viña del Mar'),
        ]
        self.conn.executemany(
            "INSERT OR IGNORE INTO dim_hipodromos (codigo, nombre, ciudad) VALUES (?, ?, ?)",
            hipodromos
        )
        
        # Superficies
        superficies = [
            ('arena', 'Pista de Arena'),
            ('pasto', 'Pista de Pasto'),
            ('sintetico', 'Pista Sintética'),
        ]
        self.conn.executemany(
            "INSERT OR IGNORE INTO dim_superficies (codigo, descripcion) VALUES (?, ?)",
            superficies
        )
        
        # Tipos de carrera básicos
        tipos = [
            ('HAND', 'Handicap', 'Carrera con pesos asignados por índice', 2),
            ('COND', 'Condicional', 'Carrera por condiciones específicas', 2),
            ('CLAS', 'Clásico', 'Carrera clásica de alto nivel', 4),
            ('LIST', 'Listada', 'Carrera listada internacionalmente', 5),
            ('GRP1', 'Grupo 1', 'Máximo nivel internacional', 5),
        ]
        self.conn.executemany(
            "INSERT OR IGNORE INTO dim_tipos_carrera (codigo, nombre, descripcion, nivel_importancia) VALUES (?, ?, ?, ?)",
            tipos
        )
        
        self.conn.commit()
    
    def get_or_create_dimension(
        self, 
        table: str, 
        lookup_column: str, 
        value: str,
        extra_columns: dict = None
    ) -> int:
        """
        Obtiene el ID de una dimensión o la crea si no existe.
        Patrón común en ETL para manejo de dimensiones.
        """
        if not value or value.strip() == '':
            return None
        
        value = value.strip()
        
        cursor = self.conn.execute(
            f"SELECT id FROM {table} WHERE {lookup_column} = ?",
            (value,)
        )
        row = cursor.fetchone()
        
        if row:
            return row['id']
        
        # Crear nuevo registro
        columns = [lookup_column]
        values = [value]
        
        if extra_columns:
            columns.extend(extra_columns.keys())
            values.extend(extra_columns.values())
        
        placeholders = ', '.join(['?'] * len(values))
        col_names = ', '.join(columns)
        
        cursor = self.conn.execute(
            f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})",
            values
        )
        self.conn.commit()
        return cursor.lastrowid


def create_fresh_database(db_path: str) -> DatabaseManager:
    """
    Crea una base de datos nueva con el esquema 3FN.
    """
    db = DatabaseManager(db_path)
    db.initialize_schema()
    return db


if __name__ == "__main__":
    # Test de creación
    logging.basicConfig(level=logging.INFO)
    
    test_db = "test_hipica_3fn.db"
    with DatabaseManager(test_db) as db:
        db.initialize_schema()
        print(f"✅ Base de datos creada: {test_db}")