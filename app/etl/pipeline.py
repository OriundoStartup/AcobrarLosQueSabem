"""
================================================================================
PISTA INTELIGENTE - ETL PIPELINE PROFESIONAL
================================================================================
Pipeline de Extracción, Transformación y Carga (ETL) optimizado para el
procesamiento de datos de carreras hípicas desde múltiples fuentes.

Características:
- Validación robusta de datos
- Manejo de errores con registro de rechazos
- Carga incremental (upsert)
- Tracking completo de batches
- Actualización automática de agregaciones

Autor: Data Engineering Team - Pista Inteligente
================================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ==============================================================================

class SourceType(Enum):
    CSV_PROGRAMA = "csv_programa"
    CSV_RESULTADOS = "csv_resultados"
    PDF_PROGRAMA = "pdf_programa"
    API_SCRAPING = "api_scraping"


@dataclass
class ETLConfig:
    """Configuración del pipeline ETL."""
    db_path: str = "data/db/hipica_3fn.db"
    batch_size: int = 1000
    validate_strict: bool = True
    reject_threshold: float = 0.3  # Si más del 30% son rechazados, abort
    update_aggregations: bool = True


@dataclass
class ValidationResult:
    """Resultado de validación de un registro."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cleaned_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ETLBatchResult:
    """Resultado de un batch de procesamiento."""
    batch_id: str
    source_type: str
    source_file: str
    records_raw: int
    records_inserted: int
    records_updated: int
    records_rejected: int
    errors: List[Dict]
    started_at: datetime
    completed_at: datetime
    status: str


# ==============================================================================
# VALIDADORES Y TRANSFORMADORES
# ==============================================================================

class DataCleaner:
    """Clase para limpieza y transformación de datos."""
    
    # Mapeo de hipódromos a códigos
    HIPODROMO_MAP = {
        'hipódromo chile': 'HC',
        'hipodromo chile': 'HC',
        'club hípico de santiago': 'CHC',
        'club hipico de santiago': 'CHC',
        'club hípico': 'CHC',
        'valparaíso sporting club': 'VSC',
        'valparaiso sporting club': 'VSC',
        'sporting': 'VSC',
    }
    
    # Regex para distancias
    DISTANCIA_PATTERN = re.compile(r'(\d+)\s*m?', re.IGNORECASE)
    
    # Regex para premios
    PREMIO_PATTERN = re.compile(r'[\d.,]+')
    
    @classmethod
    def clean_distancia(cls, value: Any) -> Optional[int]:
        """Convierte distancia a metros (entero)."""
        if pd.isna(value) or value == '':
            return None
        
        match = cls.DISTANCIA_PATTERN.search(str(value))
        if match:
            return int(match.group(1))
        return None
    
    @classmethod
    def clean_peso(cls, value: Any) -> Optional[float]:
        """Convierte peso a float en kg."""
        if pd.isna(value) or value == '':
            return None
        
        try:
            cleaned = str(value).replace('Kg', '').replace('kg', '')
            cleaned = cleaned.replace(',', '.').strip()
            peso = float(cleaned)
            # Validar rango razonable (45-65 kg típico)
            if 40 <= peso <= 70:
                return peso
            return None
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def clean_premio(cls, value: Any, moneda: str = 'CLP') -> Optional[float]:
        """Extrae monto numérico del premio."""
        if pd.isna(value) or value == '':
            return None
        
        try:
            # Remover texto y símbolos de moneda
            numeros = cls.PREMIO_PATTERN.findall(str(value))
            if numeros:
                # Tomar el número más grande (el premio total)
                cleaned = max(numeros, key=lambda x: float(x.replace('.', '').replace(',', '.')))
                # Normalizar separadores
                cleaned = cleaned.replace('.', '').replace(',', '.')
                return float(cleaned)
            return None
        except (ValueError, TypeError):
            return None
    
    @classmethod
    def normalize_hipodromo(cls, value: Any) -> str:
        """Normaliza nombre de hipódromo a código."""
        if pd.isna(value) or value == '':
            return 'HC'  # Default
        
        value_lower = str(value).lower().strip()
        
        for key, code in cls.HIPODROMO_MAP.items():
            if key in value_lower:
                return code
        
        return 'HC'  # Default si no match
    
    @classmethod
    def clean_nombre(cls, value: Any) -> Optional[str]:
        """Limpia y normaliza nombres (caballos, jinetes, etc)."""
        if pd.isna(value) or value == '':
            return None
        
        # Normalizar espacios y mayúsculas
        cleaned = ' '.join(str(value).split())
        cleaned = cleaned.strip().upper()
        
        # Validar longitud mínima
        if len(cleaned) < 2:
            return None
        
        return cleaned
    
    @classmethod
    def clean_fecha(cls, value: Any) -> Optional[date]:
        """Convierte a fecha."""
        if pd.isna(value):
            return None
        
        if isinstance(value, (datetime, date)):
            return value if isinstance(value, date) else value.date()
        
        # Intentar varios formatos
        formatos = ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y/%m/%d']
        
        for fmt in formatos:
            try:
                return datetime.strptime(str(value), fmt).date()
            except ValueError:
                continue
        
        return None
    
    @classmethod
    def clean_sexo(cls, value: Any) -> Optional[str]:
        """Normaliza sexo del caballo."""
        if pd.isna(value) or value == '':
            return None
        
        value_upper = str(value).upper().strip()
        
        if value_upper in ['M', 'MACHO', 'CABALLO']:
            return 'M'
        elif value_upper in ['H', 'HEMBRA', 'YEGUA']:
            return 'H'
        elif value_upper in ['C', 'CASTRADO']:
            return 'C'
        
        return None
    
    @classmethod
    def extract_edad(cls, value: Any) -> Optional[int]:
        """Extrae edad en años."""
        if pd.isna(value) or value == '':
            return None
        
        try:
            # Buscar número
            numeros = re.findall(r'\d+', str(value))
            if numeros:
                edad = int(numeros[0])
                if 2 <= edad <= 15:  # Rango típico
                    return edad
            return None
        except (ValueError, TypeError):
            return None


class DataValidator:
    """Validador de registros de entrada."""
    
    @classmethod
    def validate_carrera(cls, row: Dict) -> ValidationResult:
        """Valida un registro de carrera."""
        errors = []
        warnings = []
        cleaned = {}
        
        # Campo obligatorio: fecha
        fecha = DataCleaner.clean_fecha(row.get('fecha'))
        if not fecha:
            errors.append("Fecha inválida o faltante")
        else:
            cleaned['fecha'] = fecha
        
        # Campo obligatorio: hipódromo
        hipodromo_code = DataCleaner.normalize_hipodromo(row.get('hipodromo'))
        cleaned['hipodromo_codigo'] = hipodromo_code
        
        # Campo obligatorio: número de carrera
        try:
            nro = int(row.get('nro_carrera', 0))
            if 1 <= nro <= 20:
                cleaned['nro_carrera'] = nro
            else:
                errors.append(f"Número de carrera fuera de rango: {nro}")
        except (ValueError, TypeError):
            errors.append("Número de carrera inválido")
        
        # Campo obligatorio: distancia
        distancia = DataCleaner.clean_distancia(row.get('distancia'))
        if not distancia:
            warnings.append("Distancia no especificada, usando default")
            distancia = 1200
        cleaned['distancia_metros'] = distancia
        
        # Campos opcionales
        cleaned['condicion_texto'] = str(row.get('condicion', '')).strip()
        cleaned['premio_primero'] = DataCleaner.clean_premio(row.get('premio'))
        cleaned['hora_programada'] = str(row.get('hora', '')).strip() or None
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned
        )
    
    @classmethod
    def validate_participante(cls, row: Dict) -> ValidationResult:
        """Valida un registro de participante."""
        errors = []
        warnings = []
        cleaned = {}
        
        # Campo obligatorio: nombre del caballo
        nombre_caballo = DataCleaner.clean_nombre(row.get('nombre_caballo'))
        if not nombre_caballo:
            errors.append("Nombre de caballo inválido o faltante")
        else:
            cleaned['nombre_caballo'] = nombre_caballo
        
        # Partidor
        try:
            partidor = int(row.get('partidor', 0))
            if 1 <= partidor <= 20:
                cleaned['partidor'] = partidor
            else:
                warnings.append(f"Partidor fuera de rango: {partidor}")
                cleaned['partidor'] = None
        except (ValueError, TypeError):
            cleaned['partidor'] = None
        
        # Jinete
        jinete = DataCleaner.clean_nombre(row.get('jinete'))
        cleaned['nombre_jinete'] = jinete
        
        # Peso
        peso = DataCleaner.clean_peso(row.get('peso_jinete'))
        if not peso:
            warnings.append("Peso no especificado")
        cleaned['peso_programado'] = peso
        
        # Stud
        stud = DataCleaner.clean_nombre(row.get('stud'))
        cleaned['nombre_stud'] = stud
        
        # Edad y sexo
        cleaned['edad_anos'] = DataCleaner.extract_edad(row.get('edad'))
        cleaned['sexo'] = DataCleaner.clean_sexo(row.get('sexo'))
        
        # Handicap
        try:
            handicap = float(row.get('handicap', 0) or 0)
            cleaned['handicap'] = handicap if handicap > 0 else None
        except (ValueError, TypeError):
            cleaned['handicap'] = None
        
        # Resultado (puede ser NULL para carreras futuras)
        resultado = row.get('resultado_final')
        if resultado is not None and resultado != '':
            try:
                res = int(resultado)
                if 1 <= res <= 20:
                    cleaned['resultado_final'] = res
                else:
                    cleaned['resultado_final'] = None
            except (ValueError, TypeError):
                cleaned['resultado_final'] = None
        else:
            cleaned['resultado_final'] = None
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned
        )


# ==============================================================================
# PIPELINE ETL PRINCIPAL
# ==============================================================================

class ETLPipeline:
    """
    Pipeline ETL profesional para carga de datos hípicos.
    
    Flujo:
    1. Extract: Lee datos crudos de CSV/PDF/API
    2. Transform: Valida, limpia y normaliza
    3. Load: Inserta/actualiza en BD normalizada
    4. Post-process: Actualiza agregaciones
    """
    
    def __init__(self, config: ETLConfig = None):
        self.config = config or ETLConfig()
        self.db_path = self.config.db_path
        self._connection: Optional[sqlite3.Connection] = None
        
        # Caché de dimensiones para evitar queries repetidas
        self._cache_hipodromos: Dict[str, int] = {}
        self._cache_caballos: Dict[str, int] = {}
        self._cache_jinetes: Dict[str, int] = {}
        self._cache_studs: Dict[str, int] = {}
    
    @property
    def conn(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
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
    
    def generate_batch_id(self, source_file: str) -> str:
        """Genera un ID único para el batch."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(source_file.encode()).hexdigest()[:8]
        return f"batch_{timestamp}_{file_hash}"
    
    def _load_dimension_cache(self):
        """Pre-carga cachés de dimensiones."""
        cursor = self.conn.cursor()
        
        # Hipódromos
        cursor.execute("SELECT codigo, id FROM dim_hipodromos")
        self._cache_hipodromos = {row['codigo']: row['id'] for row in cursor.fetchall()}
        
        # Caballos
        cursor.execute("SELECT nombre, id FROM dim_caballos")
        self._cache_caballos = {row['nombre']: row['id'] for row in cursor.fetchall()}
        
        # Jinetes
        cursor.execute("SELECT nombre, id FROM dim_jinetes")
        self._cache_jinetes = {row['nombre']: row['id'] for row in cursor.fetchall()}
        
        # Studs
        cursor.execute("SELECT nombre, id FROM dim_studs")
        self._cache_studs = {row['nombre']: row['id'] for row in cursor.fetchall()}
        
        logger.info(f"Cache cargado: {len(self._cache_caballos)} caballos, "
                   f"{len(self._cache_jinetes)} jinetes, {len(self._cache_studs)} studs")
    
    def _get_or_create_hipodromo(self, codigo: str) -> int:
        """Obtiene o crea un hipódromo."""
        if codigo in self._cache_hipodromos:
            return self._cache_hipodromos[codigo]
        
        # Si no existe, usar default (no debería pasar con seed inicial)
        return self._cache_hipodromos.get('HC', 1)
    
    def _get_or_create_caballo(self, nombre: str, sexo: str = None, stud_id: int = None) -> int:
        """Obtiene o crea un caballo."""
        if not nombre:
            return None
        
        if nombre in self._cache_caballos:
            return self._cache_caballos[nombre]
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO dim_caballos (nombre, sexo, stud_id) VALUES (?, ?, ?)",
            (nombre, sexo, stud_id)
        )
        new_id = cursor.lastrowid
        self._cache_caballos[nombre] = new_id
        return new_id
    
    def _get_or_create_jinete(self, nombre: str) -> Optional[int]:
        """Obtiene o crea un jinete."""
        if not nombre:
            return None
        
        if nombre in self._cache_jinetes:
            return self._cache_jinetes[nombre]
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO dim_jinetes (nombre) VALUES (?)",
            (nombre,)
        )
        new_id = cursor.lastrowid
        self._cache_jinetes[nombre] = new_id
        return new_id
    
    def _get_or_create_stud(self, nombre: str) -> Optional[int]:
        """Obtiene o crea un stud."""
        if not nombre:
            return None
        
        if nombre in self._cache_studs:
            return self._cache_studs[nombre]
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO dim_studs (nombre) VALUES (?)",
            (nombre,)
        )
        new_id = cursor.lastrowid
        self._cache_studs[nombre] = new_id
        return new_id
    
    def _upsert_carrera(
        self, 
        fecha: date,
        hipodromo_id: int,
        nro_carrera: int,
        data: Dict,
        batch_id: str
    ) -> int:
        """Inserta o actualiza una carrera. Retorna el ID."""
        cursor = self.conn.cursor()
        
        # Buscar existente
        cursor.execute(
            """SELECT id FROM fact_carreras 
               WHERE fecha = ? AND hipodromo_id = ? AND nro_carrera = ?""",
            (fecha, hipodromo_id, nro_carrera)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update
            cursor.execute(
                """UPDATE fact_carreras SET
                   distancia_metros = ?,
                   condicion_texto = ?,
                   premio_primero = ?,
                   hora_programada = ?,
                   etl_batch_id = ?,
                   updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (
                    data.get('distancia_metros'),
                    data.get('condicion_texto'),
                    data.get('premio_primero'),
                    data.get('hora_programada'),
                    batch_id,
                    existing['id']
                )
            )
            return existing['id']
        else:
            # Insert
            cursor.execute(
                """INSERT INTO fact_carreras 
                   (fecha, hipodromo_id, nro_carrera, distancia_metros, 
                    condicion_texto, premio_primero, hora_programada, etl_batch_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fecha,
                    hipodromo_id,
                    nro_carrera,
                    data.get('distancia_metros'),
                    data.get('condicion_texto'),
                    data.get('premio_primero'),
                    data.get('hora_programada'),
                    batch_id
                )
            )
            return cursor.lastrowid
    
    def _upsert_participacion(
        self,
        carrera_id: int,
        caballo_id: int,
        data: Dict,
        batch_id: str
    ) -> Tuple[int, bool]:
        """Inserta o actualiza una participación. Retorna (ID, is_new)."""
        cursor = self.conn.cursor()
        
        jinete_id = self._get_or_create_jinete(data.get('nombre_jinete'))
        
        # Buscar existente
        cursor.execute(
            """SELECT id FROM fact_participaciones 
               WHERE carrera_id = ? AND caballo_id = ?""",
            (carrera_id, caballo_id)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update
            cursor.execute(
                """UPDATE fact_participaciones SET
                   jinete_id = ?,
                   partidor = ?,
                   peso_programado = ?,
                   handicap = ?,
                   edad_anos = ?,
                   resultado_final = ?,
                   etl_batch_id = ?,
                   updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (
                    jinete_id,
                    data.get('partidor'),
                    data.get('peso_programado'),
                    data.get('handicap'),
                    data.get('edad_anos'),
                    data.get('resultado_final'),
                    batch_id,
                    existing['id']
                )
            )
            return existing['id'], False
        else:
            # Insert
            cursor.execute(
                """INSERT INTO fact_participaciones 
                   (carrera_id, caballo_id, jinete_id, partidor, peso_programado,
                    handicap, edad_anos, resultado_final, etl_batch_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    carrera_id,
                    caballo_id,
                    jinete_id,
                    data.get('partidor'),
                    data.get('peso_programado'),
                    data.get('handicap'),
                    data.get('edad_anos'),
                    data.get('resultado_final'),
                    batch_id
                )
            )
            return cursor.lastrowid, True
    
    def process_legacy_db(self, legacy_db_path: str) -> ETLBatchResult:
        """
        Migra datos desde la base de datos legacy (esquema antiguo) al nuevo esquema 3FN.
        """
        # Resolver path absoluto y verificar existencia
        legacy_path_obj = Path(legacy_db_path).resolve()
        if not legacy_path_obj.exists():
            raise FileNotFoundError(f"No se encuentra la BD legacy en: {legacy_path_obj}")
            
        batch_id = self.generate_batch_id(str(legacy_path_obj))
        started_at = datetime.now()
        
        logger.info(f"🔄 Iniciando migración desde BD legacy: {legacy_path_obj}")
        logger.info(f"   Batch ID: {batch_id}")
        
        # Cargar caché
        self._load_dimension_cache()
        
        # Conectar a BD legacy
        legacy_conn = sqlite3.connect(str(legacy_path_obj))
        legacy_conn.row_factory = sqlite3.Row
        
        # Debug: Listar tablas
        cursor = legacy_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        logger.info(f"   Tablas encontradas en legacy: {tables}")
        
        if 'carreras' not in tables:
             raise ValueError(f"La tabla 'carreras' no existe en {legacy_path_obj}. Tablas disponibles: {tables}")

        stats = {
            'raw': 0,
            'inserted': 0,
            'updated': 0,
            'rejected': 0
        }
        errors = []
        
        try:
            # 1. Migrar carreras
            logger.info("📊 Migrando carreras...")
            cursor = legacy_conn.cursor()
            cursor.execute("""
                SELECT id, fecha, hipodromo, nro_carrera, distancia, 
                       condicion, premio, superficie, hora
                FROM carreras
            """)
            carreras_legacy = cursor.fetchall()
            
            carrera_id_map = {}  # legacy_id -> new_id
            
            for row in carreras_legacy:
                stats['raw'] += 1
                
                # Validar
                validation = DataValidator.validate_carrera(dict(row))
                
                if not validation.is_valid:
                    stats['rejected'] += 1
                    errors.append({
                        'type': 'carrera',
                        'legacy_id': row['id'],
                        'errors': validation.errors
                    })
                    continue
                
                # Obtener IDs de dimensiones
                hipodromo_id = self._get_or_create_hipodromo(
                    validation.cleaned_data['hipodromo_codigo']
                )
                
                # Upsert
                new_id = self._upsert_carrera(
                    fecha=validation.cleaned_data['fecha'],
                    hipodromo_id=hipodromo_id,
                    nro_carrera=validation.cleaned_data['nro_carrera'],
                    data=validation.cleaned_data,
                    batch_id=batch_id
                )
                
                carrera_id_map[row['id']] = new_id
                stats['inserted'] += 1
            
            self.conn.commit()
            logger.info(f"   ✅ {len(carrera_id_map)} carreras migradas")
            
            # 2. Migrar participantes
            logger.info("📊 Migrando participantes...")
            cursor.execute("""
                SELECT id, carrera_id, partidor, nombre_caballo, jinete,
                       peso_jinete, stud, edad, sexo, handicap, resultado_final
                FROM participantes
            """)
            participantes_legacy = cursor.fetchall()
            
            participantes_insertados = 0
            participantes_actualizados = 0
            
            for row in participantes_legacy:
                stats['raw'] += 1
                
                # Verificar que la carrera fue migrada
                if row['carrera_id'] not in carrera_id_map:
                    stats['rejected'] += 1
                    continue
                
                # Validar
                validation = DataValidator.validate_participante(dict(row))
                
                if not validation.is_valid:
                    stats['rejected'] += 1
                    errors.append({
                        'type': 'participante',
                        'legacy_id': row['id'],
                        'errors': validation.errors
                    })
                    continue
                
                # Crear/obtener dimensiones
                stud_id = self._get_or_create_stud(validation.cleaned_data.get('nombre_stud'))
                caballo_id = self._get_or_create_caballo(
                    nombre=validation.cleaned_data['nombre_caballo'],
                    sexo=validation.cleaned_data.get('sexo'),
                    stud_id=stud_id
                )
                
                # Upsert participación
                new_carrera_id = carrera_id_map[row['carrera_id']]
                _, is_new = self._upsert_participacion(
                    carrera_id=new_carrera_id,
                    caballo_id=caballo_id,
                    data=validation.cleaned_data,
                    batch_id=batch_id
                )
                
                if is_new:
                    participantes_insertados += 1
                else:
                    participantes_actualizados += 1
            
            self.conn.commit()
            stats['inserted'] += participantes_insertados
            stats['updated'] = participantes_actualizados
            
            logger.info(f"   ✅ {participantes_insertados} participantes insertados, "
                       f"{participantes_actualizados} actualizados")
            
            # 3. Registrar batch en control
            self._register_batch(batch_id, SourceType.CSV_PROGRAMA.value, 
                               legacy_db_path, stats, errors, started_at)
            
            # 4. Actualizar agregaciones si está habilitado
            if self.config.update_aggregations:
                self.update_aggregations()
            
            legacy_conn.close()
            
            return ETLBatchResult(
                batch_id=batch_id,
                source_type='legacy_migration',
                source_file=legacy_db_path,
                records_raw=stats['raw'],
                records_inserted=stats['inserted'],
                records_updated=stats['updated'],
                records_rejected=stats['rejected'],
                errors=errors,
                started_at=started_at,
                completed_at=datetime.now(),
                status='completed'
            )
            
        except Exception as e:
            logger.error(f"❌ Error en migración: {e}")
            self.conn.rollback()
            legacy_conn.close()
            raise
    
    def process_csv(
        self, 
        csv_path: str,
        source_type: SourceType = SourceType.CSV_PROGRAMA
    ) -> ETLBatchResult:
        """
        Procesa un archivo CSV con datos de carreras/participantes.
        
        Formato esperado del CSV:
        - Columnas de carrera: fecha, hipodromo, nro_carrera, distancia, condicion, premio
        - Columnas de participante: partidor, nombre_caballo, jinete, peso_jinete, stud, etc.
        """
        batch_id = self.generate_batch_id(csv_path)
        started_at = datetime.now()
        
        logger.info(f"🔄 Procesando CSV: {csv_path}")
        logger.info(f"   Batch ID: {batch_id}")
        
        # Cargar caché
        self._load_dimension_cache()
        
        # Leer CSV
        df = pd.read_csv(csv_path)
        logger.info(f"   {len(df)} registros en archivo")
        
        stats = {'raw': len(df), 'inserted': 0, 'updated': 0, 'rejected': 0}
        errors = []
        
        try:
            # Agrupar por carrera (fecha + hipódromo + nro_carrera)
            carrera_cols = ['fecha', 'hipodromo', 'nro_carrera']
            
            for carrera_key, grupo in df.groupby(carrera_cols, dropna=False):
                fecha, hipodromo, nro_carrera = carrera_key
                
                # Tomar primera fila para datos de carrera
                first_row = grupo.iloc[0].to_dict()
                
                # Validar carrera
                validation_carrera = DataValidator.validate_carrera({
                    'fecha': fecha,
                    'hipodromo': hipodromo,
                    'nro_carrera': nro_carrera,
                    'distancia': first_row.get('distancia'),
                    'condicion': first_row.get('condicion'),
                    'premio': first_row.get('premio'),
                    'hora': first_row.get('hora')
                })
                
                if not validation_carrera.is_valid:
                    stats['rejected'] += len(grupo)
                    errors.append({
                        'type': 'carrera',
                        'key': str(carrera_key),
                        'errors': validation_carrera.errors
                    })
                    continue
                
                # Upsert carrera
                hipodromo_id = self._get_or_create_hipodromo(
                    validation_carrera.cleaned_data['hipodromo_codigo']
                )
                
                carrera_id = self._upsert_carrera(
                    fecha=validation_carrera.cleaned_data['fecha'],
                    hipodromo_id=hipodromo_id,
                    nro_carrera=validation_carrera.cleaned_data['nro_carrera'],
                    data=validation_carrera.cleaned_data,
                    batch_id=batch_id
                )
                
                # Procesar participantes del grupo
                for _, row in grupo.iterrows():
                    validation_part = DataValidator.validate_participante(row.to_dict())
                    
                    if not validation_part.is_valid:
                        stats['rejected'] += 1
                        continue
                    
                    # Crear dimensiones
                    stud_id = self._get_or_create_stud(
                        validation_part.cleaned_data.get('nombre_stud')
                    )
                    caballo_id = self._get_or_create_caballo(
                        nombre=validation_part.cleaned_data['nombre_caballo'],
                        sexo=validation_part.cleaned_data.get('sexo'),
                        stud_id=stud_id
                    )
                    
                    # Upsert participación
                    _, is_new = self._upsert_participacion(
                        carrera_id=carrera_id,
                        caballo_id=caballo_id,
                        data=validation.cleaned_data,
                        batch_id=batch_id
                    )
                    
                    if is_new:
                        stats['inserted'] += 1
                    else:
                        stats['updated'] += 1
            
            self.conn.commit()
            
            # Registrar batch
            self._register_batch(batch_id, source_type.value, csv_path, 
                               stats, errors, started_at)
            
            # Actualizar agregaciones
            if self.config.update_aggregations:
                self.update_aggregations()
            
            return ETLBatchResult(
                batch_id=batch_id,
                source_type=source_type.value,
                source_file=csv_path,
                records_raw=stats['raw'],
                records_inserted=stats['inserted'],
                records_updated=stats['updated'],
                records_rejected=stats['rejected'],
                errors=errors,
                started_at=started_at,
                completed_at=datetime.now(),
                status='completed'
            )
            
        except Exception as e:
            logger.error(f"❌ Error procesando CSV: {e}")
            self.conn.rollback()
            raise
    
    def _register_batch(
        self, 
        batch_id: str,
        source_type: str,
        source_file: str,
        stats: Dict,
        errors: List,
        started_at: datetime
    ):
        """Registra el batch en la tabla de control."""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO etl_control 
               (batch_id, source_type, source_file, records_raw, records_inserted,
                records_updated, records_rejected, errors, started_at, completed_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                source_type,
                source_file,
                stats['raw'],
                stats['inserted'],
                stats.get('updated', 0),
                stats['rejected'],
                json.dumps(errors[:100]),  # Limitar a 100 errores
                started_at,
                datetime.now(),
                'completed'
            )
        )
        self.conn.commit()
    
    def update_aggregations(self):
        """Actualiza las tablas de agregaciones para ML."""
        logger.info("📊 Actualizando agregaciones...")
        
        cursor = self.conn.cursor()
        
        # 1. Stats de caballos
        cursor.execute("""
            INSERT OR REPLACE INTO agg_caballo_stats 
            (caballo_id, total_carreras, victorias, segundos, terceros,
             posicion_promedio, tasa_victoria, tasa_podio, ultima_carrera,
             ultimo_resultado, dias_sin_correr, updated_at)
            SELECT 
                fp.caballo_id,
                COUNT(*) as total_carreras,
                SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) as victorias,
                SUM(CASE WHEN fp.resultado_final = 2 THEN 1 ELSE 0 END) as segundos,
                SUM(CASE WHEN fp.resultado_final = 3 THEN 1 ELSE 0 END) as terceros,
                AVG(fp.resultado_final) as posicion_promedio,
                CAST(SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as tasa_victoria,
                CAST(SUM(CASE WHEN fp.resultado_final <= 3 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as tasa_podio,
                MAX(fc.fecha) as ultima_carrera,
                (SELECT fp2.resultado_final 
                 FROM fact_participaciones fp2 
                 JOIN fact_carreras fc2 ON fp2.carrera_id = fc2.id
                 WHERE fp2.caballo_id = fp.caballo_id AND fp2.resultado_final IS NOT NULL
                 ORDER BY fc2.fecha DESC LIMIT 1) as ultimo_resultado,
                CAST(julianday('now') - julianday(MAX(fc.fecha)) AS INTEGER) as dias_sin_correr,
                CURRENT_TIMESTAMP
            FROM fact_participaciones fp
            JOIN fact_carreras fc ON fp.carrera_id = fc.id
            WHERE fp.resultado_final IS NOT NULL
            GROUP BY fp.caballo_id
        """)
        
        # 2. Stats de jinetes
        cursor.execute("""
            INSERT OR REPLACE INTO agg_jinete_stats 
            (jinete_id, total_carreras, victorias, tasa_victoria, 
             posicion_promedio, updated_at)
            SELECT 
                fp.jinete_id,
                COUNT(*) as total_carreras,
                SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) as victorias,
                CAST(SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as tasa_victoria,
                AVG(fp.resultado_final) as posicion_promedio,
                CURRENT_TIMESTAMP
            FROM fact_participaciones fp
            WHERE fp.resultado_final IS NOT NULL AND fp.jinete_id IS NOT NULL
            GROUP BY fp.jinete_id
        """)
        
        # 3. Combos caballo-jinete
        cursor.execute("""
            INSERT OR REPLACE INTO agg_combo_caballo_jinete 
            (caballo_id, jinete_id, carreras_juntos, victorias_juntos, 
             tasa_victoria_combo, updated_at)
            SELECT 
                fp.caballo_id,
                fp.jinete_id,
                COUNT(*) as carreras_juntos,
                SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) as victorias_juntos,
                CAST(SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as tasa_victoria_combo,
                CURRENT_TIMESTAMP
            FROM fact_participaciones fp
            WHERE fp.resultado_final IS NOT NULL 
              AND fp.jinete_id IS NOT NULL
            GROUP BY fp.caballo_id, fp.jinete_id
        """)
        
        self.conn.commit()
        logger.info("   ✅ Agregaciones actualizadas")


# ==============================================================================
# FUNCIONES DE UTILIDAD
# ==============================================================================

def migrate_legacy_to_3fn(
    legacy_db_path: str,
    new_db_path: str
) -> ETLBatchResult:
    """
    Función de conveniencia para migrar una BD legacy completa al esquema 3FN.
    """
    from db.schema_3fn import DatabaseManager
    
    # Crear nueva BD con esquema
    logger.info(f"🆕 Creando nueva BD: {new_db_path}")
    with DatabaseManager(new_db_path) as db:
        db.initialize_schema()
    
    # Migrar datos
    config = ETLConfig(db_path=new_db_path)
    with ETLPipeline(config) as pipeline:
        result = pipeline.process_legacy_db(legacy_db_path)
    
    return result


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Ejemplo de migración
    # result = migrate_legacy_to_3fn(
    #     "data/db/hipica_normalizada.db",
    #     "data/db/hipica_3fn.db"
    # )
    # print(result)