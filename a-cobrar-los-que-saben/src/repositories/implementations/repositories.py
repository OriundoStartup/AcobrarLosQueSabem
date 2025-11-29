"""
Repositorios para acceso a datos
Ubicación: src/repositories/implementations/repositories.py
"""
from typing import List, Optional, Dict, Tuple
from datetime import datetime, date
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

import sys
sys.path.append('src')

from models.database_models import (
    Carrera, Caballo, Resultado, Hipodromo, 
    Jinete, Entrenador, PatronTrifecta
)


class CarreraRepository:
    """Repositorio para gestionar carreras."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def crear(self, carrera: Carrera) -> Carrera:
        """Crea una nueva carrera."""
        self.session.add(carrera)
        self.session.commit()
        self.session.refresh(carrera)
        return carrera
    
    def obtener_por_id(self, carrera_id: int) -> Optional[Carrera]:
        """Obtiene una carrera por ID."""
        return self.session.query(Carrera).filter(Carrera.id == carrera_id).first()
    
    def obtener_por_fecha(self, fecha: date) -> List[Carrera]:
        """Obtiene todas las carreras de una fecha."""
        return self.session.query(Carrera).filter(Carrera.fecha == fecha).all()
    
    def obtener_por_hipodromo(self, hipodromo_id: int, limit: int = 100) -> List[Carrera]:
        """Obtiene carreras de un hipódromo."""
        return (self.session.query(Carrera)
                .filter(Carrera.hipodromo_id == hipodromo_id)
                .order_by(desc(Carrera.fecha))
                .limit(limit)
                .all())
    
    def obtener_todas(self, limit: int = 1000) -> List[Carrera]:
        """Obtiene todas las carreras."""
        return (self.session.query(Carrera)
                .order_by(desc(Carrera.fecha))
                .limit(limit)
                .all())
    
    def obtener_con_resultados(self, carrera_id: int) -> Optional[Carrera]:
        """Obtiene una carrera con sus resultados cargados."""
        return (self.session.query(Carrera)
                .filter(Carrera.id == carrera_id)
                .first())


class CaballoRepository:
    """Repositorio para gestionar caballos."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def crear(self, caballo: Caballo) -> Caballo:
        """Crea un nuevo caballo."""
        self.session.add(caballo)
        self.session.commit()
        self.session.refresh(caballo)
        return caballo
    
    def obtener_por_id(self, caballo_id: int) -> Optional[Caballo]:
        """Obtiene un caballo por ID."""
        return self.session.query(Caballo).filter(Caballo.id == caballo_id).first()
    
    def obtener_por_nombre(self, nombre: str) -> Optional[Caballo]:
        """Obtiene un caballo por nombre."""
        return self.session.query(Caballo).filter(Caballo.nombre == nombre).first()
    
    def buscar(self, termino: str) -> List[Caballo]:
        """Busca caballos por nombre."""
        return (self.session.query(Caballo)
                .filter(Caballo.nombre.ilike(f"%{termino}%"))
                .all())
    
    def obtener_o_crear(self, nombre: str, **kwargs) -> Caballo:
        """Obtiene un caballo o lo crea si no existe."""
        caballo = self.obtener_por_nombre(nombre)
        if not caballo:
            caballo = Caballo(nombre=nombre, **kwargs)
            caballo = self.crear(caballo)
        return caballo


class ResultadoRepository:
    """Repositorio para gestionar resultados."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def crear(self, resultado: Resultado) -> Resultado:
        """Crea un nuevo resultado."""
        self.session.add(resultado)
        self.session.commit()
        self.session.refresh(resultado)
        return resultado
    
    def crear_multiples(self, resultados: List[Resultado]) -> List[Resultado]:
        """Crea múltiples resultados."""
        self.session.add_all(resultados)
        self.session.commit()
        return resultados
    
    def obtener_por_carrera(self, carrera_id: int) -> List[Resultado]:
        """Obtiene todos los resultados de una carrera."""
        return (self.session.query(Resultado)
                .filter(Resultado.carrera_id == carrera_id)
                .order_by(Resultado.posicion_final)
                .all())
    
    def obtener_trifecta(self, carrera_id: int) -> List[Resultado]:
        """Obtiene la trifecta de una carrera."""
        return (self.session.query(Resultado)
                .filter(
                    Resultado.carrera_id == carrera_id,
                    Resultado.posicion_final.in_([1, 2, 3])
                )
                .order_by(Resultado.posicion_final)
                .all())
    
    def obtener_historial_caballo(self, caballo_id: int, limit: int = 20) -> List[Resultado]:
        """Obtiene el historial de un caballo."""
        return (self.session.query(Resultado)
                .join(Carrera)
                .filter(Resultado.caballo_id == caballo_id)
                .order_by(desc(Carrera.fecha))
                .limit(limit)
                .all())
    
    def obtener_estadisticas_caballo(self, caballo_id: int) -> Dict:
        """Obtiene estadísticas de un caballo."""
        resultados = self.session.query(Resultado).filter(
            Resultado.caballo_id == caballo_id
        ).all()
        
        if not resultados:
            return {}
        
        total = len(resultados)
        victorias = sum(1 for r in resultados if r.posicion_final == 1)
        podios = sum(1 for r in resultados if r.posicion_final <= 3)
        
        return {
            "carreras": total,
            "victorias": victorias,
            "podios": podios,
            "tasa_victoria": victorias / total if total > 0 else 0,
            "tasa_podio": podios / total if total > 0 else 0
        }


class TrifectaRepository:
    """Repositorio para análisis de trifectas."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def obtener_todas_trifectas(self) -> List[Tuple[int, int, int, int]]:
        """
        Obtiene todas las trifectas históricas.
        Retorna: Lista de tuplas (primero_id, segundo_id, tercero_id, carrera_id)
        """
        query = """
        SELECT 
            r1.caballo_id as primero,
            r2.caballo_id as segundo,
            r3.caballo_id as tercero,
            r1.carrera_id
        FROM resultados r1
        JOIN resultados r2 ON r1.carrera_id = r2.carrera_id
        JOIN resultados r3 ON r1.carrera_id = r3.carrera_id
        WHERE r1.posicion_final = 1
          AND r2.posicion_final = 2
          AND r3.posicion_final = 3
        """
        result = self.session.execute(query)
        return result.fetchall()
    
    def detectar_patrones_repetidos(self, minimo: int = 2) -> List[Dict]:
        """
        Detecta patrones de trifectas que se han repetido.
        """
        trifectas = self.obtener_todas_trifectas()
        
        # Contar frecuencias
        patrones = {}
        for primero, segundo, tercero, carrera_id in trifectas:
            key = (primero, segundo, tercero)
            if key not in patrones:
                patrones[key] = {"count": 0, "carreras": []}
            patrones[key]["count"] += 1
            patrones[key]["carreras"].append(carrera_id)
        
        # Filtrar por frecuencia mínima
        resultado = []
        for (p1, p2, p3), data in patrones.items():
            if data["count"] >= minimo:
                # Obtener nombres de caballos
                cab1 = self.session.query(Caballo).get(p1)
                cab2 = self.session.query(Caballo).get(p2)
                cab3 = self.session.query(Caballo).get(p3)
                
                resultado.append({
                    "primero_id": p1,
                    "segundo_id": p2,
                    "tercero_id": p3,
                    "primero": cab1.nombre if cab1 else "Desconocido",
                    "segundo": cab2.nombre if cab2 else "Desconocido",
                    "tercero": cab3.nombre if cab3 else "Desconocido",
                    "frecuencia": data["count"],
                    "carreras": data["carreras"]
                })
        
        # Ordenar por frecuencia descendente
        resultado.sort(key=lambda x: x["frecuencia"], reverse=True)
        return resultado
    
    def guardar_patron(self, primero: int, segundo: int, tercero: int, 
                      hipodromo_id: Optional[int] = None) -> PatronTrifecta:
        """Guarda un patrón de trifecta detectado."""
        patron = PatronTrifecta(
            primero=primero,
            segundo=segundo,
            tercero=tercero,
            frecuencia=1,
            ultima_vez=date.today(),
            hipodromo_id=hipodromo_id
        )
        self.session.add(patron)
        self.session.commit()
        return patron


class HipodromoRepository:
    """Repositorio para gestionar hipódromos."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def crear(self, hipodromo: Hipodromo) -> Hipodromo:
        """Crea un nuevo hipódromo."""
        self.session.add(hipodromo)
        self.session.commit()
        self.session.refresh(hipodromo)
        return hipodromo
    
    def obtener_por_nombre(self, nombre: str) -> Optional[Hipodromo]:
        """Obtiene un hipódromo por nombre."""
        return self.session.query(Hipodromo).filter(Hipodromo.nombre == nombre).first()
    
    def obtener_o_crear(self, nombre: str, **kwargs) -> Hipodromo:
        """Obtiene un hipódromo o lo crea si no existe."""
        hipodromo = self.obtener_por_nombre(nombre)
        if not hipodromo:
            hipodromo = Hipodromo(nombre=nombre, **kwargs)
            hipodromo = self.crear(hipodromo)
        return hipodromo
    
    def obtener_todos(self) -> List[Hipodromo]:
        """Obtiene todos los hipódromos."""
        return self.session.query(Hipodromo).filter(Hipodromo.activo == True).all()