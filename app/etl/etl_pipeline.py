"""
================================================================================
PISTA INTELIGENTE - ETL PIPELINE PROFESIONAL
================================================================================
Módulo: etl_pipeline.py

Pipeline de Extracción, Transformación y Carga (ETL) con auto-detección
de tipo de CSV y mapeo automático de columnas.

Características:
- Auto-detección de tipo de CSV (programa vs resultados)
- Auto-detección de hipódromo y fecha desde nombre de archivo
- Mapeo automático de columnas al esquema 3FN
- Validación robusta de datos
- Manejo de errores con registro de rechazos
- Carga incremental (upsert)
- Actualización automática de agregaciones

Formatos soportados:
- PROGRAMA_CHC_YYYY-MM-DD.csv
- PROGRAMA_HC_YYYY-MM-DD.csv  
- resul_chc_YYYY-MM-DD.csv
- resul_hc_YYYY-MM-DD.csv

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

# Importar detector de CSV
from detector import (
    CSVDetector, CSVMapper, CSVType, Hipodromo,
    CSVDetectionResult, process_csv_auto
)

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
    CSV_AUTO = "csv_auto"  # Auto-detectado
    PDF_PROGRAMA = "pdf_programa"
    API_SCRAPING = "api_scraping"
    LEGACY_MIGRATION = "legacy_migration"


@dataclass
class ETLConfig:
    """Configuración del pipeline ETL."""
    db_path: str = "data/db/hipica_3fn.db"
    batch_size: int = 1000
    validate_strict: bool = True
    reject_threshold: float = 0.3
    update_aggregations: bool = True
    auto_detect_csv: bool = True  # Nueva opción


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
    detection_info: Optional[Dict] = None  # Info de auto-detección


# ==============================================================================
# VALIDADORES
# ==============================================================================

class DataValidator:
    """Validador de registros de entrada."""
    
    @classmethod
    def validate_carrera(cls, row: Dict) -> ValidationResult:
        """Valida un registro de carrera."""
        errors = []
        warnings = []
        cleaned = {}
        
        # Campo obligatorio: fecha
        fecha = row.get('fecha')
        if isinstance(fecha, date):
            cleaned['fecha'] = fecha
        elif fecha:
            try:
                if isinstance(fecha, str):
                    cleaned['fecha'] = datetime.strptime(fecha, '%Y-%m-%d').date()
                else:
                    cleaned['fecha'] = fecha
            except:
                errors.append("Fecha inválida o faltante")
        else:
            errors.append("Fecha faltante")
        
        # Campo obligatorio: hipódromo
        hipodromo = row.get('hipodromo', 'HC')
        cleaned['hipodromo_codigo'] = hipodromo if hipodromo else 'HC'
        
        # Campo obligatorio: número de carrera
        try:
            nro = int(row.get('nro_carrera', 0))
            if 1 <= nro <= 20:
                cleaned['nro_carrera'] = nro
            else:
                errors.append(f"Número de carrera fuera de rango: {nro}")
        except (ValueError, TypeError):
            errors.append("Número de carrera inválido")
        
        # Campo opcional: distancia
        distancia = row.get('distancia')
        if distancia and not pd.isna(distancia):
            cleaned['distancia_metros'] = int(distancia)
        else:
            warnings.append("Distancia no especificada, usando default")
            cleaned['distancia_metros'] = 1200
        
        # Campos opcionales
        cleaned['condicion_texto'] = str(row.get('condicion_texto', '') or '').strip()
        cleaned['nombre_premio'] = str(row.get('nombre_premio', '') or '').strip()
        
        premio = row.get('premio')
        if premio and not pd.isna(premio):
            cleaned['premio_primero'] = float(premio)
        else:
            cleaned['premio_primero'] = None
        
        cleaned['hora_programada'] = str(row.get('hora_programada', '') or '').strip() or None
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned
        )
    
    @classmethod
    def validate_participante(cls, row: Dict, is_resultado: bool = False) -> ValidationResult:
        """Valida un registro de participante."""
        errors = []
        warnings = []
        cleaned = {}
        
        # Campo obligatorio: nombre del caballo
        nombre_caballo = row.get('nombre_caballo')
        if nombre_caballo and not pd.isna(nombre_caballo):
            cleaned['nombre_caballo'] = str(nombre_caballo).strip().upper()
        else:
            errors.append("Nombre de caballo inválido o faltante")
        
        # Partidor
        partidor = row.get('partidor')
        if partidor and not pd.isna(partidor):
            try:
                p = int(partidor)
                if 1 <= p <= 20:
                    cleaned['partidor'] = p
                else:
                    warnings.append(f"Partidor fuera de rango: {p}")
                    cleaned['partidor'] = None
            except (ValueError, TypeError):
                cleaned['partidor'] = None
        else:
            cleaned['partidor'] = None
        
        # Jinete
        jinete = row.get('jinete')
        if jinete and not pd.isna(jinete):
            cleaned['nombre_jinete'] = str(jinete).strip().upper()
        else:
            cleaned['nombre_jinete'] = None
        
        # Peso jinete
        peso = row.get('peso_jinete')
        if peso and not pd.isna(peso):
            try:
                cleaned['peso_programado'] = float(peso)
            except (ValueError, TypeError):
                cleaned['peso_programado'] = None
        else:
            cleaned['peso_programado'] = None
        
        # Stud
        stud = row.get('stud')
        if stud and not pd.isna(stud):
            cleaned['nombre_stud'] = str(stud).strip().upper()
        else:
            cleaned['nombre_stud'] = None
        
        # Preparador/Entrenador
        preparador = row.get('preparador')
        if preparador and not pd.isna(preparador):
            cleaned['preparador'] = str(preparador).strip().upper()
        else:
            cleaned['preparador'] = None
        
        # Padre/Padrillo
        padre = row.get('padre')
        if padre and not pd.isna(padre):
            cleaned['padre'] = str(padre).strip().upper()
        else:
            cleaned['padre'] = None
        
        # Handicap
        handicap = row.get('handicap')
        if handicap and not pd.isna(handicap):
            try:
                cleaned['handicap'] = float(handicap)
            except (ValueError, TypeError):
                cleaned['handicap'] = None
        else:
            cleaned['handicap'] = None
        
        # Resultado (solo para CSV de resultados)
        resultado = row.get('resultado_final')
        if resultado is not None and not pd.isna(resultado):
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
        
        # Dividendo (solo para resultados)
        dividendo = row.get('dividendo')
        if dividendo and not pd.isna(dividendo):
            try:
                cleaned['dividendo'] = float(dividendo)
            except (ValueError, TypeError):
                cleaned['dividendo'] = None
        else:
            cleaned['dividendo'] = None
        
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
    Pipeline ETL con auto-detección de CSV.
    
    Flujo:
    1. Detect: Auto-detecta tipo de CSV, hipódromo y fecha
    2. Extract: Lee y mapea columnas automáticamente
    3. Transform: Valida, limpia y normaliza
    4. Load: Inserta/actualiza en BD normalizada
    5. Post-process: Actualiza agregaciones
    """
    
    def __init__(self, config: ETLConfig = None):
        self.config = config or ETLConfig()
        self.db_path = self.config.db_path
        self._connection: Optional[sqlite3.Connection] = None
        
        # Caché de dimensiones
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
        
        cursor.execute("SELECT codigo, id FROM dim_hipodromos")
        self._cache_hipodromos = {row['codigo']: row['id'] for row in cursor.fetchall()}
        
        cursor.execute("SELECT nombre, id FROM dim_caballos")
        self._cache_caballos = {row['nombre']: row['id'] for row in cursor.fetchall()}
        
        cursor.execute("SELECT nombre, id FROM dim_jinetes")
        self._cache_jinetes = {row['nombre']: row['id'] for row in cursor.fetchall()}
        
        cursor.execute("SELECT nombre, id FROM dim_studs")
        self._cache_studs = {row['nombre']: row['id'] for row in cursor.fetchall()}
        
        logger.info(f"Cache: {len(self._cache_caballos)} caballos, "
                   f"{len(self._cache_jinetes)} jinetes")
    
    def _get_or_create_hipodromo(self, codigo: str) -> int:
        """Obtiene o crea un hipódromo."""
        if codigo in self._cache_hipodromos:
            return self._cache_hipodromos[codigo]
        return self._cache_hipodromos.get('HC', 1)

    def _get_or_create_caballo(self, nombre: str, sexo: str = None, stud_id: int = None) -> Optional[int]:
        """Obtiene o crea un caballo."""
        if not nombre:
            return None
        
        if nombre in self._cache_caballos:
            return self._cache_caballos[nombre]
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO dim_caballos (nombre) VALUES (?)",
            (nombre,)
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
        cursor.execute("INSERT INTO dim_jinetes (nombre) VALUES (?)", (nombre,))
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
        cursor.execute("INSERT INTO dim_studs (nombre) VALUES (?)", (nombre,))
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
        
        cursor.execute(
            """SELECT id FROM fact_carreras 
               WHERE fecha = ? AND hipodromo_id = ? AND nro_carrera = ?""",
            (fecha, hipodromo_id, nro_carrera)
        )
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute(
                """UPDATE fact_carreras SET
                   distancia_metros = COALESCE(?, distancia_metros)
                   WHERE id = ?""",
                (
                    data.get('distancia_metros'),
                    existing['id']
                )
            )
            return existing['id']
        else:
            cursor.execute(
                """INSERT INTO fact_carreras 
                   (fecha, hipodromo_id, nro_carrera, distancia_metros)
                   VALUES (?, ?, ?, ?)""",
                (
                    fecha,
                    hipodromo_id,
                    nro_carrera,
                    data.get('distancia_metros')
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
        
        cursor.execute(
            """SELECT id FROM fact_participaciones 
               WHERE carrera_id = ? AND caballo_id = ?""",
            (carrera_id, caballo_id)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update - preservar datos existentes si nuevos son NULL
            cursor.execute(
                """UPDATE fact_participaciones SET
                   jinete_id = COALESCE(?, jinete_id),
                   partidor = COALESCE(?, partidor),
                   peso_programado = COALESCE(?, peso_programado),
                   handicap = COALESCE(?, handicap),
                   resultado_final = COALESCE(?, resultado_final)
                   WHERE id = ?""",
                (
                    jinete_id,
                    data.get('partidor'),
                    data.get('peso_programado'),
                    data.get('handicap'),
                    data.get('resultado_final'),
                    existing['id']
                )
            )
            return existing['id'], False
        else:
            cursor.execute(
                """INSERT INTO fact_participaciones 
                   (carrera_id, caballo_id, jinete_id, partidor, peso_programado,
                    handicap, resultado_final)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    carrera_id,
                    caballo_id,
                    jinete_id,
                    data.get('partidor'),
                    data.get('peso_programado'),
                    data.get('handicap'),
                    data.get('resultado_final')
                )
            )
            return cursor.lastrowid, True

    def process_csv(self, csv_path: str, source_type: SourceType = None) -> ETLBatchResult:
        """
        Procesa un archivo CSV con auto-detección de tipo.
        
        Args:
            csv_path: Ruta al archivo CSV
            source_type: Tipo forzado (opcional, si es None se auto-detecta)
            
        Returns:
            ETLBatchResult con estadísticas del proceso
        """
        batch_id = self.generate_batch_id(csv_path)
        started_at = datetime.now()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 PROCESANDO CSV: {Path(csv_path).name}")
        logger.info(f"{'='*60}")
        logger.info(f"   Batch ID: {batch_id}")
        
        # 1. AUTO-DETECCIÓN
        if self.config.auto_detect_csv and source_type is None:
            df_mapped, detection = process_csv_auto(csv_path)
            
            # Determinar source_type desde detección
            if detection.csv_type == CSVType.PROGRAMA:
                source_type = SourceType.CSV_PROGRAMA
            elif detection.csv_type == CSVType.RESULTADOS:
                source_type = SourceType.CSV_RESULTADOS
            else:
                source_type = SourceType.CSV_AUTO
            
            logger.info(f"   ✅ Auto-detectado: {detection.csv_type.value}")
            logger.info(f"   📍 Hipódromo: {detection.hipodromo.value}")
            logger.info(f"   📅 Fecha: {detection.fecha}")
            
            detection_info = {
                'csv_type': detection.csv_type.value,
                'hipodromo': detection.hipodromo.value,
                'fecha': str(detection.fecha) if detection.fecha else None,
                'confidence': detection.confidence,
                'method': detection.detection_method
            }
        else:
            # Modo legacy: sin auto-detección
            df_mapped = pd.read_csv(csv_path)
            detection = None
            detection_info = None
        
        # Cargar caché de dimensiones
        self._load_dimension_cache()
        
        stats = {'raw': len(df_mapped), 'inserted': 0, 'updated': 0, 'rejected': 0}
        errors = []
        is_resultado = source_type == SourceType.CSV_RESULTADOS
        
        try:
            # 2. AGRUPAR POR CARRERA
            # Identificar columnas de agrupación disponibles
            group_cols = []
            if 'fecha' in df_mapped.columns:
                group_cols.append('fecha')
            if 'hipodromo' in df_mapped.columns:
                group_cols.append('hipodromo')
            if 'nro_carrera' in df_mapped.columns:
                group_cols.append('nro_carrera')
            
            if not group_cols or 'nro_carrera' not in group_cols:
                raise ValueError("CSV debe tener al menos columna 'nro_carrera'")
            
            # 3. PROCESAR CADA CARRERA
            for carrera_key, grupo in df_mapped.groupby(group_cols, dropna=False):
                # Extraer valores de la key
                if len(group_cols) == 1:
                    nro_carrera = carrera_key
                    fecha = detection.fecha if detection else date.today()
                    hipodromo_code = detection.hipodromo.value if detection else 'HC'
                else:
                    key_dict = dict(zip(group_cols, carrera_key if isinstance(carrera_key, tuple) else [carrera_key]))
                    nro_carrera = key_dict.get('nro_carrera')
                    fecha = key_dict.get('fecha', detection.fecha if detection else date.today())
                    hipodromo_code = key_dict.get('hipodromo', detection.hipodromo.value if detection else 'HC')
                
                # Construir datos de carrera
                first_row = grupo.iloc[0].to_dict()
                carrera_data = {
                    'fecha': fecha,
                    'hipodromo': hipodromo_code,
                    'nro_carrera': nro_carrera,
                    'distancia': first_row.get('distancia'),
                    'condicion_texto': first_row.get('condicion_texto', ''),
                    'nombre_premio': first_row.get('nombre_premio', ''),
                    'premio': first_row.get('premio'),
                    'hora_programada': first_row.get('hora_programada')
                }
                
                # Validar carrera
                validation_carrera = DataValidator.validate_carrera(carrera_data)
                
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
                
                # 4. PROCESAR PARTICIPANTES
                for _, row in grupo.iterrows():
                    row_dict = row.to_dict()
                    validation_part = DataValidator.validate_participante(row_dict, is_resultado)
                    
                    if not validation_part.is_valid:
                        stats['rejected'] += 1
                        errors.append({
                            'type': 'participante',
                            'caballo': row_dict.get('nombre_caballo', 'N/A'),
                            'errors': validation_part.errors
                        })
                        continue
                    
                    # Crear dimensiones
                    stud_id = self._get_or_create_stud(
                        validation_part.cleaned_data.get('nombre_stud')
                    )
                    caballo_id = self._get_or_create_caballo(
                        nombre=validation_part.cleaned_data['nombre_caballo'],
                        stud_id=stud_id
                    )
                    
                    if not caballo_id:
                        stats['rejected'] += 1
                        continue
                    
                    # Upsert participación
                    _, is_new = self._upsert_participacion(
                        carrera_id=carrera_id,
                        caballo_id=caballo_id,
                        data=validation_part.cleaned_data,
                        batch_id=batch_id
                    )
                    
                    if is_new:
                        stats['inserted'] += 1
                    else:
                        stats['updated'] += 1
            
            self.conn.commit()
            
            # 5. REGISTRAR BATCH
            self._register_batch(batch_id, source_type.value, csv_path, 
                               stats, errors, started_at)
            
            # 6. ACTUALIZAR AGREGACIONES
            if self.config.update_aggregations and is_resultado:
                self.update_aggregations()
            
            # Resumen
            logger.info(f"\n📊 RESUMEN:")
            logger.info(f"   Registros raw: {stats['raw']}")
            logger.info(f"   Insertados: {stats['inserted']}")
            logger.info(f"   Actualizados: {stats['updated']}")
            logger.info(f"   Rechazados: {stats['rejected']}")
            
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
                status='completed',
                detection_info=detection_info
            )
            
        except Exception as e:
            logger.error(f"❌ Error procesando CSV: {e}")
            self.conn.rollback()
            raise
    
    def process_directory(self, dir_path: str) -> List[ETLBatchResult]:
        """
        Procesa todos los CSV de un directorio.
        
        Args:
            dir_path: Ruta al directorio
            
        Returns:
            Lista de resultados por archivo
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise ValueError(f"No es un directorio: {dir_path}")
        
        results = []
        csv_files = sorted(path.glob("*.csv"))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📁 PROCESANDO DIRECTORIO: {path.name}")
        logger.info(f"   {len(csv_files)} archivos CSV encontrados")
        logger.info(f"{'='*60}")
        
        for csv_file in csv_files:
            try:
                result = self.process_csv(str(csv_file))
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Error en {csv_file.name}: {e}")
                results.append(ETLBatchResult(
                    batch_id=f"error_{csv_file.stem}",
                    source_type='error',
                    source_file=str(csv_file),
                    records_raw=0,
                    records_inserted=0,
                    records_updated=0,
                    records_rejected=0,
                    errors=[{'error': str(e)}],
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    status='failed'
                ))
        
        # Resumen final
        total_inserted = sum(r.records_inserted for r in results)
        total_updated = sum(r.records_updated for r in results)
        total_rejected = sum(r.records_rejected for r in results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 RESUMEN TOTAL DEL DIRECTORIO")
        logger.info(f"   Archivos procesados: {len(results)}")
        logger.info(f"   Total insertados: {total_inserted}")
        logger.info(f"   Total actualizados: {total_updated}")
        logger.info(f"   Total rechazados: {total_rejected}")
        logger.info(f"{'='*60}")
        
        return results
    
    def process_legacy_db(self, legacy_db_path: str) -> ETLBatchResult:
        """
        Migra datos desde una base de datos legacy al esquema 3FN.
        
        Args:
            legacy_db_path: Ruta a la BD legacy
            
        Returns:
            ETLBatchResult con estadísticas
        """
        batch_id = self.generate_batch_id(legacy_db_path)
        started_at = datetime.now()
        
        logger.info(f"🔄 Migrando desde BD legacy: {legacy_db_path}")
        logger.info(f"   Batch ID: {batch_id}")
        
        self._load_dimension_cache()
        
        legacy_conn = sqlite3.connect(legacy_db_path)
        legacy_conn.row_factory = sqlite3.Row
        
        stats = {'raw': 0, 'inserted': 0, 'updated': 0, 'rejected': 0}
        errors = []
        
        try:
            cursor = legacy_conn.cursor()
            
            # 1. Migrar carreras
            logger.info("📊 Migrando carreras...")
            cursor.execute("""
                SELECT id, fecha, hipodromo, nro_carrera, distancia, 
                       condicion, premio, superficie, hora
                FROM carreras
            """)
            carreras_legacy = cursor.fetchall()
            
            carrera_id_map = {}
            
            for row in carreras_legacy:
                stats['raw'] += 1
                
                validation = DataValidator.validate_carrera({
                    'fecha': row['fecha'],
                    'hipodromo': row['hipodromo'],
                    'nro_carrera': row['nro_carrera'],
                    'distancia': row['distancia'],
                    'condicion_texto': row['condicion'],
                    'premio': row['premio'],
                    'hora_programada': row['hora']
                })
                
                if not validation.is_valid:
                    stats['rejected'] += 1
                    errors.append({
                        'type': 'carrera',
                        'legacy_id': row['id'],
                        'errors': validation.errors
                    })
                    continue
                
                hipodromo_id = self._get_or_create_hipodromo(
                    validation.cleaned_data['hipodromo_codigo']
                )
                
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
                
                if row['carrera_id'] not in carrera_id_map:
                    stats['rejected'] += 1
                    continue
                
                validation = DataValidator.validate_participante({
                    'nombre_caballo': row['nombre_caballo'],
                    'partidor': row['partidor'],
                    'jinete': row['jinete'],
                    'peso_jinete': row['peso_jinete'],
                    'stud': row['stud'],
                    'handicap': row['handicap'],
                    'resultado_final': row['resultado_final']
                })
                
                if not validation.is_valid:
                    stats['rejected'] += 1
                    errors.append({
                        'type': 'participante',
                        'legacy_id': row['id'],
                        'errors': validation.errors
                    })
                    continue
                
                stud_id = self._get_or_create_stud(validation.cleaned_data.get('nombre_stud'))
                caballo_id = self._get_or_create_caballo(
                    nombre=validation.cleaned_data['nombre_caballo'],
                    stud_id=stud_id
                )
                
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
            
            self._register_batch(batch_id, SourceType.LEGACY_MIGRATION.value,
                               legacy_db_path, stats, errors, started_at)
            
            if self.config.update_aggregations:
                self.update_aggregations()
            
            legacy_conn.close()
            
            return ETLBatchResult(
                batch_id=batch_id,
                source_type=SourceType.LEGACY_MIGRATION.value,
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
                json.dumps(errors[:100]),
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
        
        # Stats de caballos
        cursor.execute("""
            INSERT OR REPLACE INTO agg_caballo_stats 
            (caballo_id, total_carreras, victorias, segundo_lugar, tercer_lugar,
             posicion_promedio, tasa_victoria, dias_sin_correr, racha_actual, fecha_actualizacion)
            SELECT 
                fp.caballo_id,
                COUNT(*) as total_carreras,
                SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) as victorias,
                SUM(CASE WHEN fp.resultado_final = 2 THEN 1 ELSE 0 END) as segundo_lugar,
                SUM(CASE WHEN fp.resultado_final = 3 THEN 1 ELSE 0 END) as tercer_lugar,
                AVG(fp.resultado_final) as posicion_promedio,
                CAST(SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as tasa_victoria,
                CAST(julianday('now') - julianday(MAX(fc.fecha)) AS INTEGER) as dias_sin_correr,
                0 as racha_actual,
                CURRENT_TIMESTAMP
            FROM fact_participaciones fp
            JOIN fact_carreras fc ON fp.carrera_id = fc.id
            WHERE fp.resultado_final IS NOT NULL
            GROUP BY fp.caballo_id
        """)
        
        # Stats de jinetes
        cursor.execute("""
            INSERT OR REPLACE INTO agg_jinete_stats 
            (jinete_id, total_carreras, victorias, tasa_victoria, 
             posicion_promedio, fecha_actualizacion)
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
        
        # Combos caballo-jinete
        cursor.execute("""
            INSERT OR REPLACE INTO agg_combo_caballo_jinete 
            (caballo_id, jinete_id, carreras_juntos, victorias_juntos, 
             tasa_victoria_combo, fecha_actualizacion)
            SELECT 
                fp.caballo_id,
                fp.jinete_id,
                COUNT(*) as carreras_juntos,
                SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) as victorias_juntos,
                CAST(SUM(CASE WHEN fp.resultado_final = 1 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as tasa_victoria_combo,
                CURRENT_TIMESTAMP
            FROM fact_participaciones fp
            WHERE fp.resultado_final IS NOT NULL AND fp.jinete_id IS NOT NULL
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
    Migra una BD legacy completa al esquema 3FN.
    
    Args:
        legacy_db_path: Ruta a la BD legacy
        new_db_path: Ruta donde crear la nueva BD
        
    Returns:
        ETLBatchResult con estadísticas
    """
    from db.schema_3fn import DatabaseManager
    
    logger.info(f"🆕 Creando nueva BD: {new_db_path}")
    with DatabaseManager(new_db_path) as db:
        db.initialize_schema()
    
    config = ETLConfig(db_path=new_db_path)
    with ETLPipeline(config) as pipeline:
        result = pipeline.process_legacy_db(legacy_db_path)
    
    return result


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ETL Pipeline - Pista Inteligente")
    parser.add_argument('input', help='Archivo CSV o directorio')
    parser.add_argument('--db', default='data/db/hipica_3fn.db', help='Base de datos')
    parser.add_argument('--no-agg', action='store_true', help='No actualizar agregaciones')
    
    args = parser.parse_args()
    
    config = ETLConfig(
        db_path=args.db,
        update_aggregations=not args.no_agg
    )
    
    with ETLPipeline(config) as pipeline:
        path = Path(args.input)
        
        if path.is_dir():
            pipeline.process_directory(str(path))
        else:
            pipeline.process_csv(str(path))