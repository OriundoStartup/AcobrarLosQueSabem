"""
Modelos de Base de Datos con SQLAlchemy
Ubicación: src/models/database_models.py
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, 
    DateTime, ForeignKey, Date, Boolean
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Hipodromo(Base):
    """Tabla de hipódromos."""
    __tablename__ = "hipodromos"
    
    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False, unique=True)
    ciudad = Column(String(100))
    pais = Column(String(50), default="Chile")
    activo = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relaciones
    carreras = relationship("Carrera", back_populates="hipodromo")
    
    def __repr__(self) -> str:
        return f"<Hipodromo {self.nombre}>"


class Carrera(Base):
    """Tabla de carreras."""
    __tablename__ = "carreras"
    
    id = Column(Integer, primary_key=True)
    fecha = Column(Date, nullable=False, index=True)
    hipodromo_id = Column(Integer, ForeignKey("hipodromos.id"), nullable=False)
    numero_carrera = Column(Integer, nullable=False)
    distancia = Column(Integer, nullable=False)  # en metros
    tipo_pista = Column(String(20))  # arena, pasto, sintética
    categoria = Column(String(50))  # Condicional, Handicap, etc.
    premio = Column(Float)  # premio en pesos
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relaciones
    hipodromo = relationship("Hipodromo", back_populates="carreras")
    resultados = relationship("Resultado", back_populates="carrera", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Carrera {self.fecha} - R{self.numero_carrera}>"


class Caballo(Base):
    """Tabla de caballos."""
    __tablename__ = "caballos"
    
    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False, unique=True, index=True)
    edad = Column(Integer)
    sexo = Column(String(1))  # M, H, C (Macho, Hembra, Castrado)
    padre = Column(String(100))
    madre = Column(String(100))
    criador = Column(String(100))
    propietario = Column(String(100))
    activo = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relaciones
    resultados = relationship("Resultado", back_populates="caballo")
    
    def __repr__(self) -> str:
        return f"<Caballo {self.nombre}>"


class Jinete(Base):
    """Tabla de jinetes."""
    __tablename__ = "jinetes"
    
    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False, unique=True, index=True)
    apellido = Column(String(100), nullable=False)
    nacionalidad = Column(String(50))
    peso = Column(Float)  # kg
    activo = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relaciones
    resultados = relationship("Resultado", back_populates="jinete")
    
    @property
    def nombre_completo(self) -> str:
        return f"{self.nombre} {self.apellido}"
    
    def __repr__(self) -> str:
        return f"<Jinete {self.nombre_completo}>"


class Entrenador(Base):
    """Tabla de entrenadores."""
    __tablename__ = "entrenadores"
    
    id = Column(Integer, primary_key=True)
    nombre = Column(String(100), nullable=False, unique=True, index=True)
    apellido = Column(String(100), nullable=False)
    nacionalidad = Column(String(50))
    activo = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relaciones
    resultados = relationship("Resultado", back_populates="entrenador")
    
    @property
    def nombre_completo(self) -> str:
        return f"{self.nombre} {self.apellido}"
    
    def __repr__(self) -> str:
        return f"<Entrenador {self.nombre_completo}>"


class Resultado(Base):
    """Tabla de resultados de carreras."""
    __tablename__ = "resultados"
    
    id = Column(Integer, primary_key=True)
    carrera_id = Column(Integer, ForeignKey("carreras.id"), nullable=False, index=True)
    caballo_id = Column(Integer, ForeignKey("caballos.id"), nullable=False, index=True)
    jinete_id = Column(Integer, ForeignKey("jinetes.id"))
    entrenador_id = Column(Integer, ForeignKey("entrenadores.id"))
    
    numero_ejemplar = Column(Integer, nullable=False)  # Número del caballo en programa
    posicion_final = Column(Integer, nullable=False, index=True)
    tiempo = Column(Float)  # tiempo en segundos
    peso_asignado = Column(Float)  # kilos
    dividendo_ganador = Column(Float)  # dividendo si ganó
    dividendo_place = Column(Float)  # dividendo place
    
    # Datos de apuesta
    favorito = Column(Boolean, default=False)
    cuota = Column(Float)
    
    created_at = Column(DateTime, default=datetime.now)
    
    # Relaciones
    carrera = relationship("Carrera", back_populates="resultados")
    caballo = relationship("Caballo", back_populates="resultados")
    jinete = relationship("Jinete", back_populates="resultados")
    entrenador = relationship("Entrenador", back_populates="resultados")
    
    def es_trifecta(self) -> bool:
        """Verifica si está en el podio."""
        return self.posicion_final in [1, 2, 3]
    
    def __repr__(self) -> str:
        return f"<Resultado Carrera:{self.carrera_id} Pos:{self.posicion_final}>"


class PatronTrifecta(Base):
    """Tabla para almacenar patrones de trifectas detectados."""
    __tablename__ = "patrones_trifecta"
    
    id = Column(Integer, primary_key=True)
    primero = Column(Integer, nullable=False)
    segundo = Column(Integer, nullable=False)
    tercero = Column(Integer, nullable=False)
    frecuencia = Column(Integer, default=1)
    ultima_vez = Column(Date)
    hipodromo_id = Column(Integer, ForeignKey("hipodromos.id"))
    activo = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relaciones
    hipodromo = relationship("Hipodromo")
    
    @property
    def patron(self) -> str:
        return f"{self.primero}-{self.segundo}-{self.tercero}"
    
    def __repr__(self) -> str:
        return f"<PatronTrifecta {self.patron} (x{self.frecuencia})>"


# Función para crear la base de datos
def create_database(database_url: str = "sqlite:///./data/a_cobrar_los_que_saben.db"):
    """Crea todas las tablas en la base de datos."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


# Función para obtener sesión
def get_session(engine):
    """Retorna una sesión de base de datos."""
    Session = sessionmaker(bind=engine)
    return Session()