"""
================================================================================
PISTA INTELIGENTE - SCRAPER DE PROGRAMAS HÍPICOS
================================================================================
Módulo: scraper.py

Descarga programas de carreras en PDF desde sitios oficiales y los convierte a CSV.

Fuentes soportadas:
- Club Hípico de Santiago (clubhipico.cl)
- Hipódromo Chile (teletrak.cl)
- Valparaíso Sporting Club (teletrak.cl)

Uso:
    python scraper.py                    # Descarga programa más reciente
    python scraper.py --fecha 2025-12-07 # Descarga para fecha específica
    python scraper.py --hipodromo CHC    # Solo Club Hípico

Autor: Pista Inteligente Team
================================================================================
"""

import os
import sys
import re
import time
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple
import requests
from bs4 import BeautifulSoup

# Intentar importar pdfplumber para procesamiento de PDF
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("⚠️ pdfplumber no instalado. Instalar con: pip install pdfplumber")

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXPORTS_RAW = PROJECT_ROOT / "exports" / "raw"
EXPORTS_PDF = PROJECT_ROOT / "exports" / "pdf"

# Crear directorios si no existen
EXPORTS_RAW.mkdir(parents=True, exist_ok=True)
EXPORTS_PDF.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURACIÓN DE SITIOS
# ==============================================================================

SITES = {
    "CHC": {
        "name": "Club Hípico de Santiago",
        "base_url": "https://www.clubhipico.cl",
        "programa_url": "https://www.clubhipico.cl/carreras/programa-digital/",  # URL correcta con ?fecha=
        "type": "clubhipico"
    },
    "HC": {
        "name": "Hipódromo Chile", 
        "base_url": "https://www.hipodromochile.cl",
        "programa_url": "https://www.hipodromochile.cl/carreras/programa-digital/",  # Similar a CHC
        "type": "hipodromochile"
    },
    "VSC": {
        "name": "Valparaíso Sporting Club",
        "base_url": "https://www.teletrak.cl",
        "programa_url": "https://www.teletrak.cl/valparaiso-sporting-club",
        "type": "teletrak"
    }
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'es-CL,es;q=0.9,en;q=0.8',
}


# ==============================================================================
# SCRAPER PRINCIPAL
# ==============================================================================

class ProgramaScraper:
    """Scraper de programas hípicos."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
    
    def get_programa_pdf_url(self, hipodromo: str, fecha: date = None) -> Optional[str]:
        """
        Obtiene la URL del PDF del programa para un hipódromo y fecha.
        
        Args:
            hipodromo: Código del hipódromo (CHC, HC, VSC)
            fecha: Fecha del programa (por defecto: mañana)
        
        Returns:
            URL del PDF o None si no se encuentra
        """
        if hipodromo not in SITES:
            logger.error(f"Hipódromo no soportado: {hipodromo}")
            return None
        
        site = SITES[hipodromo]
        fecha = fecha or (date.today() + timedelta(days=1))
        
        logger.info(f"🔍 Buscando programa de {site['name']} para {fecha}")
        
        if site['type'] == 'clubhipico':
            return self._get_clubhipico_pdf(fecha)
        elif site['type'] == 'teletrak':
            return self._get_teletrak_pdf(hipodromo, fecha)
        
        return None
    
    def _get_clubhipico_pdf(self, fecha: date) -> Optional[str]:
        """Obtiene URL del PDF desde clubhipico.cl"""
        try:
            # URL del programa digital con parámetro de fecha
            fecha_str = fecha.strftime('%Y-%m-%d')
            url = f"{SITES['CHC']['programa_url']}?fecha={fecha_str}"
            
            logger.info(f"   Accediendo a: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Buscar links a PDFs
            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '.pdf' in href.lower():
                    pdf_links.append(href)
                    logger.debug(f"   Encontrado PDF: {href}")
            
            # Buscar el PDF que coincida con la fecha o sea del programa
            for pdf_url in pdf_links:
                if fecha_str in pdf_url or fecha.strftime('%d-%m-%Y') in pdf_url:
                    if not pdf_url.startswith('http'):
                        pdf_url = SITES["CHC"]["base_url"] + pdf_url
                    logger.info(f"   ✅ PDF encontrado: {pdf_url}")
                    return pdf_url
            
            # Si no hay PDF, intentar extraer datos directamente del HTML
            # (El programa digital puede mostrar los datos sin PDF)
            carreras_data = self._extract_programa_from_html(soup, fecha)
            if carreras_data:
                # Guardar directamente como CSV sin PDF
                csv_path = self._save_html_programa_to_csv(carreras_data, "CHC", fecha)
                if csv_path:
                    logger.info(f"   ✅ Datos extraídos del HTML: {csv_path}")
                    return f"HTML:{csv_path}"  # Marcador especial
            
            logger.warning(f"   ⚠️ No se encontró PDF para {fecha}")
            return None
            
        except Exception as e:
            logger.error(f"   ❌ Error accediendo a Club Hípico: {e}")
            return None
    
    def _extract_programa_from_html(self, soup: BeautifulSoup, fecha: date) -> List[Dict]:
        """Extrae datos del programa directamente del HTML."""
        carreras = []
        
        # Buscar tablas o divs con datos de carreras
        # Estructura típica: tabla con Carrera, Hora, Numero, Ejemplar, Peso, Jinete
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows[1:]:  # Saltar header
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:
                    carrera_data = {
                        'carrera': cells[0].get_text(strip=True),
                        'hora': cells[1].get_text(strip=True) if len(cells) > 1 else '',
                        'numero': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                        'ejemplar': cells[3].get_text(strip=True) if len(cells) > 3 else '',
                        'peso': cells[4].get_text(strip=True) if len(cells) > 4 else '',
                        'jinete': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                    }
                    if carrera_data['carrera'] and carrera_data['ejemplar']:
                        carreras.append(carrera_data)
        
        return carreras
    
    def _save_html_programa_to_csv(self, carreras: List[Dict], hipodromo: str, fecha: date) -> Optional[Path]:
        """Guarda los datos extraídos del HTML como CSV."""
        if not carreras:
            return None
        
        import csv
        csv_filename = f"PROGRAMA_{hipodromo}_{fecha.strftime('%Y-%m-%d')}.csv"
        csv_path = EXPORTS_RAW / csv_filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Carrera', 'Hora', 'Numero', 'Ejemplar', 'Peso', 'Jinete'])
            writer.writeheader()
            for carrera in carreras:
                writer.writerow({
                    'Carrera': carrera['carrera'],
                    'Hora': carrera['hora'],
                    'Numero': carrera['numero'],
                    'Ejemplar': carrera['ejemplar'],
                    'Peso': carrera['peso'],
                    'Jinete': carrera['jinete'],
                })
        
        logger.info(f"   ✅ CSV generado desde HTML: {csv_path.name} ({len(carreras)} participantes)")
        return csv_path
    
    def _get_teletrak_pdf(self, hipodromo: str, fecha: date) -> Optional[str]:
        """Obtiene URL del PDF desde teletrak.cl"""
        try:
            site = SITES[hipodromo]
            response = self.session.get(site["programa_url"], timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Buscar links a PDFs de programa
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text().lower()
                
                if '.pdf' in href.lower() and ('programa' in text or 'volante' in text):
                    if not href.startswith('http'):
                        href = site["base_url"] + href
                    logger.info(f"   ✅ PDF encontrado: {href}")
                    return href
            
            logger.warning(f"   ⚠️ No se encontró PDF en Teletrak para {hipodromo}")
            return None
            
        except Exception as e:
            logger.error(f"   ❌ Error accediendo a Teletrak: {e}")
            return None
    
    def download_pdf(self, url: str, output_path: Path) -> bool:
        """Descarga un PDF desde una URL."""
        try:
            logger.info(f"📥 Descargando PDF...")
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"   ✅ Guardado en: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Error descargando PDF: {e}")
            return False
    
    def pdf_to_csv(self, pdf_path: Path, hipodromo: str, fecha: date) -> Optional[Path]:
        """
        Convierte un PDF de programa a CSV.
        
        Args:
            pdf_path: Ruta al archivo PDF
            hipodromo: Código del hipódromo
            fecha: Fecha del programa
        
        Returns:
            Ruta al CSV generado o None si falla
        """
        if not HAS_PDFPLUMBER:
            logger.error("❌ pdfplumber no instalado")
            return None
        
        logger.info(f"📄 Convirtiendo PDF a CSV...")
        
        try:
            all_rows = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if table and len(table) > 1:
                            # Procesar cada fila de la tabla
                            for row in table[1:]:  # Saltar header
                                if row and any(cell for cell in row if cell):
                                    all_rows.append(row)
            
            if not all_rows:
                # Si no hay tablas, intentar extraer texto
                logger.warning("   ⚠️ No se encontraron tablas, extrayendo texto...")
                all_rows = self._extract_text_rows(pdf_path)
            
            if not all_rows:
                logger.error("   ❌ No se pudo extraer datos del PDF")
                return None
            
            # Generar CSV
            csv_filename = f"PROGRAMA_{hipodromo}_{fecha.strftime('%Y-%m-%d')}.csv"
            csv_path = EXPORTS_RAW / csv_filename
            
            # Detectar y escribir columnas
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header básico de programa
                header = ['Carrera', 'Hora', 'Numero', 'Ejemplar', 'Peso', 'Jinete']
                writer.writerow(header)
                
                # Procesar filas
                for row in all_rows:
                    cleaned = [str(cell).strip() if cell else '' for cell in row[:6]]
                    if len(cleaned) >= 2 and cleaned[0]:  # Al menos carrera y algo más
                        writer.writerow(cleaned)
            
            logger.info(f"   ✅ CSV generado: {csv_path.name} ({len(all_rows)} filas)")
            return csv_path
            
        except Exception as e:
            logger.exception(f"   ❌ Error convirtiendo PDF: {e}")
            return None
    
    def _extract_text_rows(self, pdf_path: Path) -> List[List[str]]:
        """Extrae filas de texto cuando no hay tablas estructuradas."""
        rows = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    for line in lines:
                        # Buscar patrones de carrera
                        # Ejemplo: "1 12:30 1 GRAN TOMAHAWK 56 PIERO REYES"
                        parts = line.split()
                        if len(parts) >= 4 and parts[0].isdigit():
                            rows.append(parts[:6])
        
        return rows
    
    def scrape_and_convert(
        self, 
        hipodromo: str, 
        fecha: date = None
    ) -> Optional[Path]:
        """
        Proceso completo: buscar PDF, descargar y convertir a CSV.
        
        Args:
            hipodromo: Código del hipódromo
            fecha: Fecha del programa
        
        Returns:
            Ruta al CSV generado o None
        """
        fecha = fecha or (date.today() + timedelta(days=1))
        
        # 1. Obtener URL del PDF
        pdf_url = self.get_programa_pdf_url(hipodromo, fecha)
        if not pdf_url:
            return None
        
        # 2. Descargar PDF
        pdf_filename = f"PROGRAMA_{hipodromo}_{fecha.strftime('%Y-%m-%d')}.pdf"
        pdf_path = EXPORTS_PDF / pdf_filename
        
        if not self.download_pdf(pdf_url, pdf_path):
            return None
        
        # 3. Convertir a CSV
        csv_path = self.pdf_to_csv(pdf_path, hipodromo, fecha)
        
        return csv_path


# ==============================================================================
# FUNCIONES DE CONVENIENCIA
# ==============================================================================

def download_all_programas(fecha: date = None) -> Dict[str, Optional[Path]]:
    """
    Descarga programas de todos los hipódromos para una fecha.
    
    Args:
        fecha: Fecha del programa (por defecto: mañana)
    
    Returns:
        Dict con hipodromo -> path del CSV (o None si falló)
    """
    fecha = fecha or (date.today() + timedelta(days=1))
    scraper = ProgramaScraper()
    
    results = {}
    for hipodromo in SITES.keys():
        logger.info(f"\n{'='*50}")
        logger.info(f"Procesando {SITES[hipodromo]['name']}...")
        
        csv_path = scraper.scrape_and_convert(hipodromo, fecha)
        results[hipodromo] = csv_path
        
        # Pequeña pausa entre sitios
        time.sleep(2)
    
    return results


def download_programa(hipodromo: str, fecha: date = None) -> Optional[Path]:
    """
    Descarga programa de un hipódromo específico.
    
    Args:
        hipodromo: Código (CHC, HC, VSC)
        fecha: Fecha del programa
    
    Returns:
        Path al CSV o None
    """
    scraper = ProgramaScraper()
    return scraper.scrape_and_convert(hipodromo, fecha)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Scraper de Programas Hípicos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scraper.py                         # Descarga todos para mañana
  python scraper.py --hipodromo CHC         # Solo Club Hípico
  python scraper.py --fecha 2025-12-07      # Fecha específica
  python scraper.py --hipodromo CHC --fecha 2025-12-07
        """
    )
    
    parser.add_argument(
        "--hipodromo", "-H",
        choices=["CHC", "HC", "VSC", "all"],
        default="all",
        help="Hipódromo a descargar (default: all)"
    )
    parser.add_argument(
        "--fecha", "-f",
        type=str,
        help="Fecha del programa (formato: YYYY-MM-DD)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Solo probar conexión sin descargar"
    )
    
    args = parser.parse_args()
    
    # Parsear fecha
    if args.fecha:
        try:
            fecha = datetime.strptime(args.fecha, "%Y-%m-%d").date()
        except ValueError:
            print(f"❌ Formato de fecha inválido: {args.fecha}")
            print("   Use: YYYY-MM-DD (ejemplo: 2025-12-07)")
            sys.exit(1)
    else:
        fecha = date.today() + timedelta(days=1)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         🏇 PISTA INTELIGENTE - Scraper de Programas 🏇       ║
╠══════════════════════════════════════════════════════════════╣
║  Fecha: {fecha}                                           ║
║  Hipódromo: {args.hipodromo if args.hipodromo != 'all' else 'Todos':10}                                   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if args.test:
        print("🔍 Modo test - verificando conexiones...")
        scraper = ProgramaScraper()
        for code, site in SITES.items():
            try:
                response = scraper.session.get(site["programa_url"], timeout=10)
                status = "✅" if response.status_code == 200 else f"⚠️ {response.status_code}"
                print(f"   {code} ({site['name']}): {status}")
            except Exception as e:
                print(f"   {code}: ❌ Error - {e}")
    else:
        if args.hipodromo == "all":
            results = download_all_programas(fecha)
            
            print(f"\n{'='*50}")
            print("📊 RESUMEN")
            print(f"{'='*50}")
            for hip, path in results.items():
                status = f"✅ {path.name}" if path else "❌ No disponible"
                print(f"   {hip}: {status}")
        else:
            csv_path = download_programa(args.hipodromo, fecha)
            if csv_path:
                print(f"\n✅ CSV generado: {csv_path}")
            else:
                print(f"\n❌ No se pudo obtener el programa")
