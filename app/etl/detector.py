"""
================================================================================
PISTA INTELIGENTE - DETECTOR DE CSV
================================================================================
Módulo: detector.py
Auto-detección de tipo de CSV y mapeo de columnas al esquema normalizado.

Formatos soportados:
- PROGRAMA_CHC_YYYY-MM-DD.csv (Club Hípico de Santiago)
- PROGRAMA_HC_YYYY-MM-DD.csv (Hipódromo Chile)
- resul_chc_YYYY-MM-DD.csv (Resultados Club Hípico)
- resul_hc_YYYY-MM-DD.csv (Resultados Hipódromo Chile)

Autor: Pista Inteligente Team
================================================================================
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# ENUMS Y CONFIGURACIÓN
# ==============================================================================

class CSVType(Enum):
    """Tipo de CSV detectado."""
    PROGRAMA = "programa"
    RESULTADOS = "resultados"
    UNKNOWN = "unknown"


class Hipodromo(Enum):
    """Hipódromos soportados."""
    CHS = "CHC"  # Club Hípico de Santiago (código en BD: CHC)
    HC = "HC"    # Hipódromo Chile
    VSC = "VSC"  # Valparaíso Sporting Club
    UNKNOWN = "UNK"


@dataclass
class CSVDetectionResult:
    """Resultado de la detección de CSV."""
    csv_type: CSVType
    hipodromo: Hipodromo
    fecha: Optional[date]
    confidence: float  # 0.0 a 1.0
    detection_method: str
    original_filename: str


# ==============================================================================
# MAPEO DE COLUMNAS
# ==============================================================================

# Columnas del CSV de PROGRAMA
PROGRAMA_COLUMN_MAP = {
    # CSV Column → Internal Field
    'carrera': 'nro_carrera',
    'carrera_nro': 'nro_carrera',  # NUEVO: formato con underscore
    'carrera nro': 'nro_carrera',
    'nro_carrera': 'nro_carrera',
    'fecha': 'fecha',  # NUEVO: mapeo directo
    'hora': 'hora_programada',
    'tiempo': 'hora_programada',  # NUEVO: formato alternativo
    'nombre premio': 'nombre_premio',
    'premio': 'nombre_premio',
    'distancia': 'distancia',
    'condición principal': 'condicion_texto',
    'condicion principal': 'condicion_texto',
    'condiciones': 'condicion_texto',  # NUEVO: formato plural
    'premio al ganador': 'premio',
    'bolsa_premios': 'premio',  # NUEVO: formato alternativo
    'cab. n°': 'partidor',
    'cab. nº': 'partidor',
    'cab n': 'partidor',
    'n°': 'partidor',
    'nro_caballo': 'partidor',  # NUEVO: formato con underscore
    'numero': 'partidor',       # NUEVO: formato usuario
    'nro': 'partidor',
    'nombre ejemplar': 'nombre_caballo',
    'ejemplar': 'nombre_caballo',  # NUEVO: formato usuario
    'caballo': 'nombre_caballo',
    'peso': 'peso_jinete',
    'jinete': 'jinete',
    'jockey': 'jinete',
}

# Columnas del CSV de RESULTADOS
RESULTADOS_COLUMN_MAP = {
    # CSV Column → Internal Field
    'carrera': 'nro_carrera',
    'lugar': 'resultado_final',
    'puesto': 'resultado_final',
    'posición': 'resultado_final',
    'posicion': 'resultado_final',
    'partida': 'partidor',
    'n°': 'partidor',
    'caballo': 'nombre_caballo',
    'nombre ejemplar': 'nombre_caballo',
    'ejemplar': 'nombre_caballo',
    'padrillo': 'padre',
    'padre': 'padre',
    'stud': 'stud',
    'pf': 'peso_final',
    'pf (kg)': 'peso_final',    # NUEVO: formato usuario
    'peso final': 'peso_final',
    'jinete': 'jinete',
    'jockey': 'jinete',
    'peso jinete': 'peso_jinete',
    'jinete_kg': 'peso_jinete', # NUEVO: formato usuario
    'peso': 'peso_jinete',
    'preparador': 'preparador',
    'entrenador': 'preparador',
    'distancia (cpos.)': 'distancia_cuerpos',
    'distancia': 'distancia_cuerpos',  # NUEVO: en resultados es cuerpos
    'cuerpos': 'distancia_cuerpos',
    'índice': 'handicap',
    'indice': 'handicap',
    'dividendo': 'dividendo',
    'div': 'dividendo',
    'div.': 'dividendo',        # NUEVO: formato usuario
}

# Columnas únicas para detección
PROGRAMA_UNIQUE_COLS = {'hora', 'nombre premio', 'premio al ganador', 'condición principal', 'ejemplar', 'numero'}
RESULTADOS_UNIQUE_COLS = {'lugar', 'dividendo', 'div.', 'padrillo', 'preparador', 'distancia (cpos.)', 'partida'}


# ==============================================================================
# DETECTOR DE CSV
# ==============================================================================

class CSVDetector:
    """
    Detecta automáticamente el tipo de CSV y extrae metadata.
    """
    
    # Patrones de nombre de archivo
    FILENAME_PATTERNS = {
        # PROGRAMA_CHC_2025-01-15.csv o PROGRAMA_HC_15-01-2025.csv
        'programa': re.compile(
            r'programa[_\-]?(chc|hc|vsc)?[_\-]?(\d{4}[-_]\d{2}[-_]\d{2}|\d{2}[-_]\d{2}[-_]\d{4})',
            re.IGNORECASE
        ),
        # resul_chc_2025-01-15.csv o resultados_hc_15-01-2025.csv
        'resultados': re.compile(
            r'resul(?:tados?)?[_\-]?(chc|hc|vsc)?[_\-]?(\d{4}[-_]\d{2}[-_]\d{2}|\d{2}[-_]\d{2}[-_]\d{4})',
            re.IGNORECASE
        ),
    }
    
    # Mapeo de códigos de hipódromo
    HIPODROMO_MAP = {
        'chc': Hipodromo.CHS,
        'chs': Hipodromo.CHS,
        'hc': Hipodromo.HC,
        'vsc': Hipodromo.VSC,
    }
    
    @classmethod
    def detect(cls, csv_path: str) -> CSVDetectionResult:
        """
        Detecta el tipo de CSV, hipódromo y fecha.
        
        Estrategia de detección:
        1. Primero intenta por nombre de archivo (más confiable)
        2. Si falla, analiza las columnas del CSV
        """
        path = Path(csv_path)
        filename = path.stem.lower()
        
        # Intento 1: Por nombre de archivo
        result = cls._detect_by_filename(filename, path.name)
        if result.confidence >= 0.8:
            logger.info(f"✅ CSV detectado por nombre: {result.csv_type.value}, "
                       f"hipódromo: {result.hipodromo.value}, fecha: {result.fecha}")
            return result
        
        # Intento 2: Por contenido de columnas
        result = cls._detect_by_columns(csv_path, path.name)
        logger.info(f"✅ CSV detectado por columnas: {result.csv_type.value}, "
                   f"confianza: {result.confidence:.0%}")
        
        return result
    
    @classmethod
    def _detect_by_filename(cls, filename: str, original: str) -> CSVDetectionResult:
        """Detecta tipo por patrón de nombre de archivo."""
        
        csv_type = CSVType.UNKNOWN
        hipodromo = Hipodromo.UNKNOWN
        fecha = None
        confidence = 0.0
        
        # Buscar patrón de programa
        match = cls.FILENAME_PATTERNS['programa'].search(filename)
        if match:
            csv_type = CSVType.PROGRAMA
            confidence = 0.9
            
            # Extraer hipódromo
            hip_code = match.group(1)
            if hip_code:
                hipodromo = cls.HIPODROMO_MAP.get(hip_code.lower(), Hipodromo.UNKNOWN)
            
            # Extraer fecha
            fecha = cls._parse_date(match.group(2))
            
            return CSVDetectionResult(
                csv_type=csv_type,
                hipodromo=hipodromo,
                fecha=fecha,
                confidence=confidence,
                detection_method='filename_pattern',
                original_filename=original
            )
        
        # Buscar patrón de resultados
        match = cls.FILENAME_PATTERNS['resultados'].search(filename)
        if match:
            csv_type = CSVType.RESULTADOS
            confidence = 0.9
            
            # Extraer hipódromo
            hip_code = match.group(1)
            if hip_code:
                hipodromo = cls.HIPODROMO_MAP.get(hip_code.lower(), Hipodromo.UNKNOWN)
            
            # Extraer fecha
            fecha = cls._parse_date(match.group(2))
            
            return CSVDetectionResult(
                csv_type=csv_type,
                hipodromo=hipodromo,
                fecha=fecha,
                confidence=confidence,
                detection_method='filename_pattern',
                original_filename=original
            )
        
        # No match - retornar con baja confianza
        return CSVDetectionResult(
            csv_type=CSVType.UNKNOWN,
            hipodromo=Hipodromo.UNKNOWN,
            fecha=None,
            confidence=0.0,
            detection_method='filename_no_match',
            original_filename=original
        )
    
    @classmethod
    def _detect_by_columns(cls, csv_path: str, original: str) -> CSVDetectionResult:
        """Detecta tipo analizando las columnas del CSV."""
        
        try:
            # Detectar separador automáticamente
            df = pd.read_csv(csv_path, nrows=5, sep=None, engine='python')
            columns_lower = set(col.lower().strip() for col in df.columns)
            
            # Contar columnas únicas de cada tipo
            programa_matches = len(columns_lower & PROGRAMA_UNIQUE_COLS)
            resultados_matches = len(columns_lower & RESULTADOS_UNIQUE_COLS)
            
            # También verificar si hay datos en "lugar" o "resultado"
            has_results = False
            for col in df.columns:
                if col.lower() in ['lugar', 'puesto', 'posición', 'posicion']:
                    has_results = df[col].notna().any()
                    break
            
            # Determinar tipo
            if resultados_matches > programa_matches or has_results:
                csv_type = CSVType.RESULTADOS
                confidence = min(0.7, 0.3 + (resultados_matches * 0.15))
            elif programa_matches > 0:
                csv_type = CSVType.PROGRAMA
                confidence = min(0.7, 0.3 + (programa_matches * 0.15))
            else:
                csv_type = CSVType.UNKNOWN
                confidence = 0.2
            
            return CSVDetectionResult(
                csv_type=csv_type,
                hipodromo=Hipodromo.UNKNOWN,  # No podemos detectar por columnas
                fecha=None,
                confidence=confidence,
                detection_method='column_analysis',
                original_filename=original
            )
            
        except Exception as e:
            logger.warning(f"Error analizando columnas: {e}")
            return CSVDetectionResult(
                csv_type=CSVType.UNKNOWN,
                hipodromo=Hipodromo.UNKNOWN,
                fecha=None,
                confidence=0.0,
                detection_method='error',
                original_filename=original
            )
    
    @classmethod
    def _parse_date(cls, date_str: str) -> Optional[date]:
        """Parsea fecha desde string."""
        if not date_str:
            return None
        
        # Normalizar separadores
        date_str = date_str.replace('_', '-')
        
        # Intentar varios formatos
        formats = [
            '%Y-%m-%d',  # 2025-01-15
            '%d-%m-%Y',  # 15-01-2025
            '%Y%m%d',    # 20250115
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        
        return None


# ==============================================================================
# MAPPER DE COLUMNAS
# ==============================================================================

class CSVMapper:
    """
    Mapea y transforma columnas del CSV al esquema interno.
    """
    
    @classmethod
    def map_dataframe(
        cls, 
        df: pd.DataFrame, 
        csv_type: CSVType,
        detection: CSVDetectionResult
    ) -> pd.DataFrame:
        """
        Mapea un DataFrame al esquema interno.
        
        Args:
            df: DataFrame original
            csv_type: Tipo de CSV detectado
            detection: Resultado de detección con metadata
            
        Returns:
            DataFrame con columnas mapeadas y datos limpios
        """
        # Seleccionar mapeo según tipo
        if csv_type == CSVType.PROGRAMA:
            column_map = PROGRAMA_COLUMN_MAP
        elif csv_type == CSVType.RESULTADOS:
            column_map = RESULTADOS_COLUMN_MAP
        else:
            raise ValueError(f"Tipo de CSV no soportado: {csv_type}")
        
        # Crear mapeo de columnas reales
        actual_map = {}
        columns_lower = {col.lower().strip(): col for col in df.columns}
        
        for csv_col, internal_col in column_map.items():
            if csv_col in columns_lower:
                actual_map[columns_lower[csv_col]] = internal_col
        
        # Renombrar columnas
        df_mapped = df.rename(columns=actual_map)
        
        # Agregar metadata de detección
        if detection.fecha:
            df_mapped['fecha'] = detection.fecha
        
        if detection.hipodromo != Hipodromo.UNKNOWN:
            df_mapped['hipodromo'] = detection.hipodromo.value
        
        # Aplicar transformaciones específicas
        df_mapped = cls._apply_transformations(df_mapped, csv_type)
        
        return df_mapped
    
    @classmethod
    def _apply_transformations(cls, df: pd.DataFrame, csv_type: CSVType) -> pd.DataFrame:
        """Aplica transformaciones de limpieza."""
        
        df = df.copy()
        
        # 1. Limpiar distancia (extraer metros)
        if 'distancia' in df.columns:
            df['distancia'] = df['distancia'].apply(cls._extract_distancia)
        
        # 2. Limpiar resultado (extraer número de posición)
        if 'resultado_final' in df.columns:
            df['resultado_final'] = df['resultado_final'].apply(cls._extract_posicion)
        
        # 3. Limpiar peso (extraer kg)
        for col in ['peso_jinete', 'peso_final']:
            if col in df.columns:
                df[col] = df[col].apply(cls._extract_peso)
        
        # 4. Limpiar premio (extraer monto)
        if 'premio' in df.columns:
            df['premio'] = df['premio'].apply(cls._extract_premio)
        
        # 5. Limpiar dividendo
        if 'dividendo' in df.columns:
            df['dividendo'] = df['dividendo'].apply(cls._extract_dividendo)
        
        # 6. Normalizar nombres (mayúsculas, sin espacios extras)
        for col in ['nombre_caballo', 'jinete', 'stud', 'preparador', 'padre']:
            if col in df.columns:
                df[col] = df[col].apply(cls._clean_nombre)
        
        # 7. Limpiar número de carrera (extraer número de '1ª', '2ª', etc.)
        if 'nro_carrera' in df.columns:
            df['nro_carrera'] = df['nro_carrera'].apply(cls._extract_posicion)
        
        # 8. Asegurar tipos numéricos
        numeric_cols = ['nro_carrera', 'partidor', 'resultado_final', 'handicap']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def _extract_distancia(value: Any) -> Optional[int]:
        """Extrae metros de distancia: '1000m' → 1000"""
        if pd.isna(value):
            return None
        match = re.search(r'(\d+)', str(value))
        return int(match.group(1)) if match else None
    
    @staticmethod
    def _extract_posicion(value: Any) -> Optional[int]:
        """Extrae posición: '1º' → 1, '2°' → 2"""
        if pd.isna(value):
            return None
        match = re.search(r'(\d+)', str(value))
        return int(match.group(1)) if match else None
    
    @staticmethod
    def _extract_peso(value: Any) -> Optional[float]:
        """Extrae peso: '58 Kg' → 58.0, '444Kg' → 444.0"""
        if pd.isna(value):
            return None
        # Remover "Kg" y espacios
        cleaned = re.sub(r'[Kk][Gg]', '', str(value)).strip()
        try:
            return float(cleaned.replace(',', '.'))
        except ValueError:
            return None
    
    @staticmethod
    def _extract_premio(value: Any) -> Optional[float]:
        """Extrae premio: 'CLP $1.560.000' → 1560000.0"""
        if pd.isna(value):
            return None
        # Buscar todos los números
        numbers = re.findall(r'[\d.,]+', str(value))
        if numbers:
            # Tomar el más largo (probablemente el monto)
            num_str = max(numbers, key=len)
            # Normalizar: remover puntos de miles, cambiar coma por punto
            num_str = num_str.replace('.', '').replace(',', '.')
            try:
                return float(num_str)
            except ValueError:
                pass
        return None
    
    @staticmethod
    def _extract_dividendo(value: Any) -> Optional[float]:
        """Extrae dividendo: '24.5' → 24.5"""
        if pd.isna(value):
            return None
        try:
            return float(str(value).replace(',', '.'))
        except ValueError:
            return None
    
    @staticmethod
    def _clean_nombre(value: Any) -> Optional[str]:
        """Limpia nombres: normaliza espacios y mayúsculas."""
        if pd.isna(value) or str(value).strip() == '':
            return None
        return ' '.join(str(value).split()).upper().strip()


# ==============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ==============================================================================

def process_csv_auto(csv_path: str) -> Tuple[pd.DataFrame, CSVDetectionResult]:
    """
    Procesa un CSV con auto-detección de tipo y mapeo.
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        Tupla (DataFrame mapeado, resultado de detección)
        
    Raises:
        ValueError: Si no se puede detectar el tipo de CSV
        FileNotFoundError: Si el archivo no existe
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {csv_path}")
    
    # 1. Detectar tipo
    detection = CSVDetector.detect(csv_path)
    
    if detection.csv_type == CSVType.UNKNOWN:
        raise ValueError(
            f"No se pudo detectar el tipo de CSV: {path.name}\n"
            f"Nombres esperados: PROGRAMA_CHC_FECHA.csv o resul_hc_FECHA.csv"
        )
    
    # 2. Leer CSV con detección automática de separador
    df = pd.read_csv(csv_path, sep=None, engine='python')
    logger.info(f"📄 Leídas {len(df)} filas de {path.name}")
    
    # 3. Mapear columnas
    df_mapped = CSVMapper.map_dataframe(df, detection.csv_type, detection)
    
    logger.info(f"✅ Columnas mapeadas: {list(df_mapped.columns)}")
    
    return df_mapped, detection


# ==============================================================================
# EJEMPLO DE USO
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        
        try:
            df, detection = process_csv_auto(csv_path)
            
            print(f"\n{'='*60}")
            print(f"RESULTADO DE DETECCIÓN")
            print(f"{'='*60}")
            print(f"Archivo: {detection.original_filename}")
            print(f"Tipo: {detection.csv_type.value}")
            print(f"Hipódromo: {detection.hipodromo.value}")
            print(f"Fecha: {detection.fecha}")
            print(f"Confianza: {detection.confidence:.0%}")
            print(f"Método: {detection.detection_method}")
            print(f"\nColumnas mapeadas:")
            for col in df.columns:
                print(f"  - {col}")
            print(f"\nPrimeras filas:")
            print(df.head())
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Uso: python csv_detector.py <ruta_csv>")