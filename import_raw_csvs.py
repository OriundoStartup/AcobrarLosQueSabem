"""
================================================================================
IMPORTADOR DE CSVs CRUDOS A BASE DE DATOS
================================================================================
Script para importar archivos CSV crudos (PROGRAMA y resul) desde exports/raw
a la base de datos normalizada.

Formatos soportados:
- PROGRAMA: Carreras futuras con formato especial (primera línea: "CHC 01-12-2025")
- resul: Resultados de carreras pasadas (separador: ";")
================================================================================
"""

import pandas as pd
import sqlite3
import re
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
import logging
import sys

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from app.db.schema_3fn import DatabaseManager, create_fresh_database
from app.etl.pipeline import DataCleaner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CSVImporter:
    """Importador de CSVs crudos a base de datos normalizada."""
    
    def __init__(self, db_path: str = "data/db/hipica_3fn.db"):
        self.db_path = db_path
        self.db_manager: Optional[DatabaseManager] = None
        self.stats = {
            'archivos_procesados': 0,
            'carreras_insertadas': 0,
            'participaciones_insertadas': 0,
            'errores': []
        }
    
    def __enter__(self):
        # Crear base de datos si no existe
        if not Path(self.db_path).exists():
            logger.info("📦 Base de datos no existe, creando...")
            self.db_manager = create_fresh_database(self.db_path)
        else:
            logger.info("📂 Conectando a base de datos existente...")
            self.db_manager = DatabaseManager(self.db_path)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.db_manager:
            self.db_manager.close()
    
    def detect_csv_type(self, filepath: Path) -> str:
        """Detecta el tipo de CSV por nombre de archivo."""
        filename = filepath.name.upper()
        if 'PROGRAMA' in filename:
            return 'PROGRAMA'
        elif 'RESUL' in filename:
            return 'RESULTADO'
        else:
            return 'DESCONOCIDO'
    
    def parse_programa_csv(self, filepath: Path) -> Tuple[Dict, pd.DataFrame]:
        """
        Parsea un CSV de tipo PROGRAMA.
        
        Formato:
        - Línea 1: "CHC 01-12-2025" (hipódromo y fecha)
        - Línea 2: vacía
        - Línea 3: encabezados
        - Resto: datos
        """
        logger.info(f"📄 Parseando PROGRAMA: {filepath.name}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            primera_linea = f.readline().strip()
        
        # Extraer hipódromo y fecha de la primera línea
        # Formato: "CHC 01-12-2025" o "SPN 03-12-2025"
        match = re.match(r'([A-Z]+)\s+(\d{2}-\d{2}-\d{4})', primera_linea)
        if not match:
            raise ValueError(f"Formato de primera línea inválido: {primera_linea}")
        
        hipodromo_codigo = match.group(1)
        fecha_str = match.group(2)
        
        # Convertir fecha (formato DD-MM-YYYY)
        fecha = datetime.strptime(fecha_str, '%d-%m-%Y').date()
        
        # Mapear códigos de hipódromo
        hipodromo_map = {
            'CHC': 'CHS',  # Club Hípico
            'SPN': 'HC',   # Hipódromo Chile (San Pedro)
            'HC': 'HC',
            'CHS': 'CHS'
        }
        hipodromo_codigo = hipodromo_map.get(hipodromo_codigo, hipodromo_codigo)
        
        # Leer CSV saltando las primeras 2 líneas
        df = pd.read_csv(filepath, skiprows=2, encoding='utf-8')
        
        metadata = {
            'hipodromo_codigo': hipodromo_codigo,
            'fecha': fecha,
            'tipo': 'PROGRAMA'
        }
        
        logger.info(f"  ✓ Hipódromo: {hipodromo_codigo}, Fecha: {fecha}, Registros: {len(df)}")
        
        return metadata, df
    
    def parse_resul_csv(self, filepath: Path) -> Tuple[Dict, pd.DataFrame]:
        """
        Parsea un CSV de tipo RESULTADO.
        
        Formato:
        - Separador: ";"
        - Primera línea: encabezados
        - Columnas incluyen: Carrera, Lugar, Partida, Caballo, etc.
        """
        logger.info(f"📄 Parseando RESULTADO: {filepath.name}")
        
        # Extraer hipódromo y fecha del nombre del archivo
        # Formato: "resul_CHC_28-11-2025.csv" o "resul_HC_22-11-2025.csv"
        match = re.search(r'resul_([A-Z]+)_(\d{2}-\d{2}[-.]?\d{4})', filepath.name)
        if not match:
            raise ValueError(f"No se pudo extraer hipódromo/fecha del nombre: {filepath.name}")
        
        hipodromo_codigo = match.group(1)
        fecha_str = match.group(2).replace('.', '-')
        
        # Convertir fecha
        fecha = datetime.strptime(fecha_str, '%d-%m-%Y').date()
        
        # Leer CSV con separador ";"
        df = pd.read_csv(filepath, sep=';', encoding='utf-8')
        
        metadata = {
            'hipodromo_codigo': hipodromo_codigo,
            'fecha': fecha,
            'tipo': 'RESULTADO'
        }
        
        logger.info(f"  ✓ Hipódromo: {hipodromo_codigo}, Fecha: {fecha}, Registros: {len(df)}")
        
        return metadata, df
    
    def import_programa(self, metadata: Dict, df: pd.DataFrame):
        """Importa datos de un CSV PROGRAMA."""
        conn = self.db_manager.conn
        
        # Obtener ID del hipódromo
        hipodromo_id = self.db_manager.get_or_create_dimension(
            'dim_hipodromos', 'codigo', metadata['hipodromo_codigo']
        )
        
        if not hipodromo_id:
            raise ValueError(f"Hipódromo no encontrado: {metadata['hipodromo_codigo']}")
        
        # Agrupar por carrera
        for nro_carrera, grupo in df.groupby('Carrera'):
            # Datos de la carrera (tomar primera fila del grupo)
            primera_fila = grupo.iloc[0]
            
            # Limpiar distancia
            distancia_str = str(primera_fila['Distancia']).replace('m', '').strip()
            distancia = DataCleaner.clean_distancia(distancia_str)
            
            # Limpiar premio
            premio_str = str(primera_fila.get('Premio al Ganador', 'N/D'))
            premio = DataCleaner.clean_premio(premio_str)
            
            # Insertar o actualizar carrera
            cursor = conn.execute("""
                INSERT OR REPLACE INTO fact_carreras 
                (fecha, hipodromo_id, nro_carrera, distancia_metros, condicion_texto, 
                 premio_primero, hora_programada, estado, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'programada', ?)
            """, (
                metadata['fecha'],
                hipodromo_id,
                int(nro_carrera),
                distancia,
                str(primera_fila.get('Condición Principal', '')),
                premio,
                str(primera_fila.get('Hora', '')),
                str(Path(metadata.get('source_file', '')).name)
            ))
            
            carrera_id = cursor.lastrowid
            if cursor.lastrowid == 0:
                # Ya existe, obtener ID
                cursor = conn.execute("""
                    SELECT id FROM fact_carreras 
                    WHERE fecha = ? AND hipodromo_id = ? AND nro_carrera = ?
                """, (metadata['fecha'], hipodromo_id, int(nro_carrera)))
                carrera_id = cursor.fetchone()[0]
            else:
                self.stats['carreras_insertadas'] += 1
            
            # Insertar participantes
            for _, row in grupo.iterrows():
                try:
                    # Obtener o crear dimensiones
                    caballo_nombre = DataCleaner.clean_nombre(row.get('Nombre Ejemplar', ''))
                    if not caballo_nombre:
                        continue
                    
                    caballo_id = self.db_manager.get_or_create_dimension(
                        'dim_caballos', 'nombre', caballo_nombre
                    )
                    
                    jinete_nombre = DataCleaner.clean_nombre(row.get('Jinete', ''))
                    jinete_id = None
                    if jinete_nombre:
                        jinete_id = self.db_manager.get_or_create_dimension(
                            'dim_jinetes', 'nombre', jinete_nombre
                        )
                    
                    # Peso
                    peso = DataCleaner.clean_peso(row.get('Peso', 0))
                    
                    # Insertar participación
                    conn.execute("""
                        INSERT OR REPLACE INTO fact_participaciones
                        (carrera_id, caballo_id, jinete_id, partidor, peso_programado, source_file)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        carrera_id,
                        caballo_id,
                        jinete_id,
                        int(row.get('Cab. N°', 0)),
                        peso,
                        str(Path(metadata.get('source_file', '')).name)
                    ))
                    
                    self.stats['participaciones_insertadas'] += 1
                    
                except Exception as e:
                    logger.warning(f"  ⚠️  Error en participante: {e}")
                    self.stats['errores'].append(str(e))
        
        conn.commit()
    
    def import_resultado(self, metadata: Dict, df: pd.DataFrame):
        """Importa datos de un CSV RESULTADO."""
        conn = self.db_manager.conn
        
        # Obtener ID del hipódromo
        hipodromo_id = self.db_manager.get_or_create_dimension(
            'dim_hipodromos', 'codigo', metadata['hipodromo_codigo']
        )
        
        if not hipodromo_id:
            raise ValueError(f"Hipódromo no encontrado: {metadata['hipodromo_codigo']}")
        
        # Agrupar por carrera
        for nro_carrera, grupo in df.groupby('Carrera'):
            # Datos de la carrera
            primera_fila = grupo.iloc[0]
            
            # Limpiar distancia
            distancia_str = str(primera_fila.get('Distancia', '1000m'))
            distancia = DataCleaner.clean_distancia(distancia_str)
            
            # Insertar o actualizar carrera
            cursor = conn.execute("""
                INSERT OR REPLACE INTO fact_carreras 
                (fecha, hipodromo_id, nro_carrera, distancia_metros, estado, source_file)
                VALUES (?, ?, ?, ?, 'corrida', ?)
            """, (
                metadata['fecha'],
                hipodromo_id,
                int(nro_carrera),
                distancia,
                str(Path(metadata.get('source_file', '')).name)
            ))
            
            carrera_id = cursor.lastrowid
            if cursor.lastrowid == 0:
                # Ya existe, obtener ID
                cursor = conn.execute("""
                    SELECT id FROM fact_carreras 
                    WHERE fecha = ? AND hipodromo_id = ? AND nro_carrera = ?
                """, (metadata['fecha'], hipodromo_id, int(nro_carrera)))
                carrera_id = cursor.fetchone()[0]
            else:
                self.stats['carreras_insertadas'] += 1
            
            # Insertar participantes con resultados
            for _, row in grupo.iterrows():
                try:
                    # Verificar si es retirado
                    lugar = str(row.get('Lugar', '')).upper()
                    if lugar == 'RT':
                        continue  # Saltar retirados por ahora
                    
                    # Obtener o crear dimensiones
                    caballo_nombre = DataCleaner.clean_nombre(row.get('Caballo', ''))
                    if not caballo_nombre:
                        continue
                    
                    caballo_id = self.db_manager.get_or_create_dimension(
                        'dim_caballos', 'nombre', caballo_nombre,
                        extra_columns={'padre': str(row.get('Padrillo', ''))}
                    )
                    
                    jinete_nombre = DataCleaner.clean_nombre(row.get('Jinete', ''))
                    jinete_id = None
                    if jinete_nombre:
                        jinete_id = self.db_manager.get_or_create_dimension(
                            'dim_jinetes', 'nombre', jinete_nombre
                        )
                    
                    entrenador_nombre = DataCleaner.clean_nombre(row.get('Preparador', ''))
                    entrenador_id = None
                    if entrenador_nombre:
                        entrenador_id = self.db_manager.get_or_create_dimension(
                            'dim_entrenadores', 'nombre', entrenador_nombre
                        )
                    
                    stud_nombre = DataCleaner.clean_nombre(row.get('Stud', ''))
                    if stud_nombre:
                        stud_id = self.db_manager.get_or_create_dimension(
                            'dim_studs', 'nombre', stud_nombre
                        )
                        # Actualizar caballo con stud
                        conn.execute(
                            "UPDATE dim_caballos SET stud_id = ? WHERE id = ?",
                            (stud_id, caballo_id)
                        )
                    
                    # Peso
                    peso = DataCleaner.clean_peso(row.get('Jinete_Kg', 0))
                    
                    # Dividendo
                    dividendo = DataCleaner.clean_premio(row.get('Div.', '0'))
                    
                    # Resultado
                    resultado_final = int(lugar) if lugar.isdigit() else None
                    
                    # Insertar participación
                    conn.execute("""
                        INSERT OR REPLACE INTO fact_participaciones
                        (carrera_id, caballo_id, jinete_id, entrenador_id, partidor, 
                         peso_programado, resultado_final, dividendo_ganador, source_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        carrera_id,
                        caballo_id,
                        jinete_id,
                        entrenador_id,
                        int(row.get('Partida', 0)),
                        peso,
                        resultado_final,
                        dividendo,
                        str(Path(metadata.get('source_file', '')).name)
                    ))
                    
                    self.stats['participaciones_insertadas'] += 1
                    
                except Exception as e:
                    logger.warning(f"  ⚠️  Error en participante: {e}")
                    self.stats['errores'].append(str(e))
        
        conn.commit()
    
    def import_csv(self, filepath: Path):
        """Importa un archivo CSV (detecta tipo automáticamente)."""
        try:
            csv_type = self.detect_csv_type(filepath)
            
            if csv_type == 'PROGRAMA':
                metadata, df = self.parse_programa_csv(filepath)
                metadata['source_file'] = str(filepath)
                self.import_programa(metadata, df)
            
            elif csv_type == 'RESULTADO':
                metadata, df = self.parse_resul_csv(filepath)
                metadata['source_file'] = str(filepath)
                self.import_resultado(metadata, df)
            
            else:
                logger.warning(f"⚠️  Tipo de CSV desconocido: {filepath.name}")
                return
            
            self.stats['archivos_procesados'] += 1
            logger.info(f"✅ Importado: {filepath.name}")
            
        except Exception as e:
            logger.error(f"❌ Error procesando {filepath.name}: {e}")
            self.stats['errores'].append(f"{filepath.name}: {str(e)}")
    
    def import_all(self, directory: str = "exports/raw"):
        """Importa todos los CSVs de un directorio."""
        csv_dir = Path(directory)
        
        if not csv_dir.exists():
            logger.error(f"❌ Directorio no existe: {directory}")
            return
        
        csv_files = list(csv_dir.glob("*.csv"))
        logger.info(f"📁 Encontrados {len(csv_files)} archivos CSV en {directory}")
        
        for csv_file in csv_files:
            self.import_csv(csv_file)
        
        # Mostrar resumen
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMEN DE IMPORTACIÓN")
        logger.info("="*60)
        logger.info(f"Archivos procesados: {self.stats['archivos_procesados']}")
        logger.info(f"Carreras insertadas: {self.stats['carreras_insertadas']}")
        logger.info(f"Participaciones insertadas: {self.stats['participaciones_insertadas']}")
        logger.info(f"Errores: {len(self.stats['errores'])}")
        
        if self.stats['errores']:
            logger.info("\n⚠️  ERRORES ENCONTRADOS:")
            for error in self.stats['errores'][:10]:  # Mostrar primeros 10
                logger.info(f"  - {error}")


def main():
    """Función principal."""
    logger.info("🚀 Iniciando importación de CSVs crudos...")
    
    with CSVImporter() as importer:
        importer.import_all("exports/raw")
    
    logger.info("\n✅ Importación completada!")


if __name__ == "__main__":
    main()
