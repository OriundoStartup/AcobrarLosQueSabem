"""
Servicios de análisis de carreras y trifectas
Ubicación: src/services/domain/analysis_service.py
"""
from typing import List, Dict, Tuple, Optional
from datetime import date, timedelta
from collections import Counter
import sys
sys.path.append('src')

from repositories.implementations.repositories import (
    TrifectaRepository, ResultadoRepository, 
    CarreraRepository, CaballoRepository
)


class TrifectaAnalysisService:
    """Servicio para análisis de trifectas."""
    
    def __init__(
        self, 
        trifecta_repo: TrifectaRepository,
        resultado_repo: ResultadoRepository,
        carrera_repo: CarreraRepository,
        caballo_repo: CaballoRepository
    ):
        self.trifecta_repo = trifecta_repo
        self.resultado_repo = resultado_repo
        self.carrera_repo = carrera_repo
        self.caballo_repo = caballo_repo
    
    def analizar_patrones_repetidos(self, minimo_repeticiones: int = 2) -> List[Dict]:
        """
        Encuentra patrones de trifectas que se han repetido.
        
        Args:
            minimo_repeticiones: Mínimo de veces que debe aparecer un patrón
            
        Returns:
            Lista de patrones detectados con información detallada
        """
        patrones = self.trifecta_repo.detectar_patrones_repetidos(minimo_repeticiones)
        
        # Enriquecer con información adicional
        for patron in patrones:
            # Calcular probabilidad histórica
            total_trifectas = len(self.trifecta_repo.obtener_todas_trifectas())
            patron["probabilidad"] = patron["frecuencia"] / total_trifectas if total_trifectas > 0 else 0
            
            # Días desde última aparición
            if patron["carreras"]:
                ultima_carrera = self.carrera_repo.obtener_por_id(patron["carreras"][-1])
                if ultima_carrera:
                    dias = (date.today() - ultima_carrera.fecha).days
                    patron["dias_desde_ultima"] = dias
        
        return patrones
    
    def calcular_combinaciones_probables(
        self, 
        caballos_programados: List[int],
        top_n: int = 10
    ) -> List[Dict]:
        """
        Calcula las combinaciones más probables basadas en historial.
        
        Args:
            caballos_programados: IDs de caballos en la próxima carrera
            top_n: Cantidad de combinaciones a retornar
            
        Returns:
            Lista de combinaciones ordenadas por probabilidad
        """
        combinaciones = []
        
        # Generar todas las combinaciones posibles
        from itertools import permutations
        for perm in permutations(caballos_programados, 3):
            primero, segundo, tercero = perm
            
            # Calcular score basado en historial individual
            score = self._calcular_score_combinacion(primero, segundo, tercero)
            
            # Obtener nombres
            cab1 = self.caballo_repo.obtener_por_id(primero)
            cab2 = self.caballo_repo.obtener_por_id(segundo)
            cab3 = self.caballo_repo.obtener_por_id(tercero)
            
            combinaciones.append({
                "primero_id": primero,
                "segundo_id": segundo,
                "tercero_id": tercero,
                "primero": cab1.nombre if cab1 else "Desconocido",
                "segundo": cab2.nombre if cab2 else "Desconocido",
                "tercero": cab3.nombre if cab3 else "Desconocido",
                "score": score,
                "probabilidad": score / 100  # Normalizar
            })
        
        # Ordenar por score y retornar top N
        combinaciones.sort(key=lambda x: x["score"], reverse=True)
        return combinaciones[:top_n]
    
    def _calcular_score_combinacion(self, primero: int, segundo: int, tercero: int) -> float:
        """Calcula un score para una combinación de trifecta."""
        score = 0.0
        
        # Score por rendimiento individual
        stats1 = self.resultado_repo.obtener_estadisticas_caballo(primero)
        stats2 = self.resultado_repo.obtener_estadisticas_caballo(segundo)
        stats3 = self.resultado_repo.obtener_estadisticas_caballo(tercero)
        
        if stats1:
            score += stats1.get("tasa_victoria", 0) * 40
        if stats2:
            score += stats2.get("tasa_podio", 0) * 30
        if stats3:
            score += stats3.get("tasa_podio", 0) * 30
        
        return score
    
    def detectar_caballos_calientes(
        self, 
        ultimos_dias: int = 30,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Detecta caballos con buen rendimiento reciente.
        
        Args:
            ultimos_dias: Ventana de tiempo a analizar
            top_n: Cantidad de caballos a retornar
            
        Returns:
            Lista de caballos con mejor rendimiento reciente
        """
        fecha_limite = date.today() - timedelta(days=ultimos_dias)
        
        # Obtener todas las carreras recientes
        carreras_recientes = []
        todas_carreras = self.carrera_repo.obtener_todas(limit=500)
        
        for carrera in todas_carreras:
            if carrera.fecha >= fecha_limite:
                carreras_recientes.append(carrera.id)
        
        if not carreras_recientes:
            return []
        
        # Analizar rendimiento por caballo
        rendimiento = {}
        
        for carrera_id in carreras_recientes:
            resultados = self.resultado_repo.obtener_por_carrera(carrera_id)
            
            for resultado in resultados:
                cab_id = resultado.caballo_id
                
                if cab_id not in rendimiento:
                    rendimiento[cab_id] = {
                        "carreras": 0,
                        "victorias": 0,
                        "podios": 0,
                        "puntos": 0
                    }
                
                rendimiento[cab_id]["carreras"] += 1
                
                # Asignar puntos según posición
                if resultado.posicion_final == 1:
                    rendimiento[cab_id]["victorias"] += 1
                    rendimiento[cab_id]["puntos"] += 10
                elif resultado.posicion_final == 2:
                    rendimiento[cab_id]["puntos"] += 5
                    rendimiento[cab_id]["podios"] += 1
                elif resultado.posicion_final == 3:
                    rendimiento[cab_id]["puntos"] += 3
                    rendimiento[cab_id]["podios"] += 1
        
        # Convertir a lista y agregar nombres
        resultado_lista = []
        for cab_id, stats in rendimiento.items():
            caballo = self.caballo_repo.obtener_por_id(cab_id)
            if caballo:
                stats["id"] = cab_id
                stats["nombre"] = caballo.nombre
                stats["promedio"] = stats["puntos"] / stats["carreras"]
                resultado_lista.append(stats)
        
        # Ordenar por puntos y retornar top N
        resultado_lista.sort(key=lambda x: x["puntos"], reverse=True)
        return resultado_lista[:top_n]
    
    def generar_alerta_patron(
        self, 
        caballos_proxima_carrera: List[int]
    ) -> Optional[Dict]:
        """
        Verifica si los caballos programados forman un patrón conocido.
        
        Args:
            caballos_proxima_carrera: IDs de caballos en próxima carrera
            
        Returns:
            Información de alerta si hay coincidencia, None si no
        """
        patrones = self.analizar_patrones_repetidos(minimo_repeticiones=2)
        
        for patron in patrones:
            # Verificar si todos los caballos del patrón están en la carrera
            if (patron["primero_id"] in caballos_proxima_carrera and
                patron["segundo_id"] in caballos_proxima_carrera and
                patron["tercero_id"] in caballos_proxima_carrera):
                
                return {
                    "alerta": True,
                    "patron": patron,
                    "mensaje": f"¡ALERTA! Patrón detectado: {patron['primero']}-{patron['segundo']}-{patron['tercero']}",
                    "frecuencia": patron["frecuencia"],
                    "recomendacion": "Este patrón se ha repetido en el pasado. Considerar para apuesta."
                }
        
        return None
    
    def analizar_rendimiento_hipodromo(self, hipodromo_id: int) -> Dict:
        """
        Analiza estadísticas generales de un hipódromo.
        
        Args:
            hipodromo_id: ID del hipódromo
            
        Returns:
            Estadísticas del hipódromo
        """
        carreras = self.carrera_repo.obtener_por_hipodromo(hipodromo_id, limit=100)
        
        if not carreras:
            return {}
        
        total_carreras = len(carreras)
        distancias = [c.distancia for c in carreras]
        
        return {
            "total_carreras": total_carreras,
            "distancia_promedio": sum(distancias) / len(distancias),
            "distancia_min": min(distancias),
            "distancia_max": max(distancias),
            "fecha_primera": carreras[-1].fecha,
            "fecha_ultima": carreras[0].fecha
        }


class DataImportService:
    """Servicio para importar datos históricos."""
    
    def __init__(
        self,
        carrera_repo: CarreraRepository,
        caballo_repo: CaballoRepository,
        resultado_repo: ResultadoRepository,
        hipodromo_repo
    ):
        self.carrera_repo = carrera_repo
        self.caballo_repo = caballo_repo
        self.resultado_repo = resultado_repo
        self.hipodromo_repo = hipodromo_repo
    
    def importar_desde_csv(self, archivo_path: str) -> Dict[str, int]:
        """
        Importa datos desde un archivo CSV.
        
        Formato esperado del CSV:
        fecha,hipodromo,numero_carrera,distancia,caballo,posicion,jinete,entrenador
        
        Returns:
            Estadísticas de importación
        """
        import csv
        from datetime import datetime
        
        carreras_creadas = 0
        caballos_creados = 0
        resultados_creados = 0
        errores = []
        
        try:
            with open(archivo_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                carrera_cache = {}  # Cache para no crear carreras duplicadas
                
                for row in reader:
                    try:
                        # Procesar hipódromo
                        hipodromo = self.hipodromo_repo.obtener_o_crear(
                            nombre=row['hipodromo']
                        )
                        
                        # Procesar carrera
                        fecha = datetime.strptime(row['fecha'], '%Y-%m-%d').date()
                        carrera_key = f"{fecha}_{hipodromo.id}_{row['numero_carrera']}"
                        
                        if carrera_key not in carrera_cache:
                            from models.database_models import Carrera
                            carrera = Carrera(
                                fecha=fecha,
                                hipodromo_id=hipodromo.id,
                                numero_carrera=int(row['numero_carrera']),
                                distancia=int(row['distancia']),
                                tipo_pista=row.get('tipo_pista', 'arena')
                            )
                            carrera = self.carrera_repo.crear(carrera)
                            carrera_cache[carrera_key] = carrera
                            carreras_creadas += 1
                        else:
                            carrera = carrera_cache[carrera_key]
                        
                        # Procesar caballo
                        caballo = self.caballo_repo.obtener_o_crear(
                            nombre=row['caballo']
                        )
                        if caballo.id is None:
                            caballos_creados += 1
                        
                        # Crear resultado
                        from models.database_models import Resultado
                        resultado = Resultado(
                            carrera_id=carrera.id,
                            caballo_id=caballo.id,
                            numero_ejemplar=int(row.get('numero_ejemplar', 0)),
                            posicion_final=int(row['posicion'])
                        )
                        self.resultado_repo.crear(resultado)
                        resultados_creados += 1
                        
                    except Exception as e:
                        errores.append(f"Error en fila: {str(e)}")
                        continue
        
        except Exception as e:
            errores.append(f"Error general: {str(e)}")
        
        return {
            "carreras_creadas": carreras_creadas,
            "caballos_creados": caballos_creados,
            "resultados_creados": resultados_creados,
            "errores": len(errores),
            "detalles_errores": errores[:10]  # Primeros 10 errores
        }
    