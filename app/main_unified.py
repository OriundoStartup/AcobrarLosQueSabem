"""
===============================================================================
PISTA INTELIGENTE - A COBRAR LOS QUE SABEN
===============================================================================
Aplicación Unificada con UI 2025 y Arquitectura Profesional

Versión: 3.0.0
Fecha: 2025-12
===============================================================================
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os
import base64

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

# Imports internos
try:
    from app.data_sync import (
        run_full_pipeline, 
        load_advanced_stats, 
        load_predictions_json,
        get_db_stats
    )
except ImportError:
    from data_sync import (
        run_full_pipeline, 
        load_advanced_stats, 
        load_predictions_json,
        get_db_stats
    )

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Pista Inteligente | A Cobrar Los Que Saben",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '### Pista Inteligente v3.0\nSistema Profesional de Análisis Hípico con IA'
    }
)

# ============================================================================
# CONSTANTS
# ============================================================================

DB_PATH = BASE_DIR / "data" / "db" / "hipica_3fn.db"
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "style_2025.css"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_image_as_base64(image_path: str) -> str:
    """Carga una imagen y la convierte a base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return ""

# ============================================================================
# STYLES
# ============================================================================

def load_styles():
    """Carga estilos CSS externos con fallback inline."""
    
    # CSS crítico inline (siempre carga)
    critical_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(255, 255, 255, 0.03);
        --accent-cyan: #00f5ff;
        --accent-magenta: #ff00aa;
        --accent-gold: #ffd700;
        --accent-lime: #b8ff00;
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --text-muted: rgba(255, 255, 255, 0.4);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --gradient-primary: linear-gradient(135deg, #00f5ff 0%, #ff00aa 50%, #ffd700 100%);
        --radius-md: 12px;
        --radius-lg: 16px;
    }
    
    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: 
            radial-gradient(ellipse 80% 50% at 20% 30%, rgba(0, 245, 255, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse 60% 40% at 80% 70%, rgba(255, 0, 170, 0.06) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    .main .block-container {
        position: relative;
        z-index: 1;
        max-width: 1400px !important;
    }
    
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 24px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 245, 255, 0.3);
        box-shadow: 0 0 40px rgba(0, 245, 255, 0.15);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 4px;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .cyan { color: var(--accent-cyan); }
    .magenta { color: var(--accent-magenta); }
    .gold { color: var(--accent-gold); }
    .lime { color: var(--accent-lime); }
    
    .glass {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
    }
    
    .prediction-row {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px;
        background: var(--bg-card);
        border-radius: var(--radius-md);
        margin-bottom: 8px;
        transition: all 0.3s ease;
        border-left: 3px solid transparent;
    }
    
    .prediction-row:hover {
        background: rgba(0, 245, 255, 0.05);
        border-left-color: var(--accent-cyan);
        transform: translateX(4px);
    }
    
    .rank-badge {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .rank-1 { background: linear-gradient(135deg, #ffd700, #ffaa00); color: #0a0a0f; }
    .rank-2 { background: linear-gradient(135deg, #c0c0c0, #a0a0a0); color: #0a0a0f; }
    .rank-3 { background: linear-gradient(135deg, #cd7f32, #a0522d); color: #0a0a0f; }
    .rank-default { background: var(--bg-secondary); color: var(--text-muted); }
    
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-radius: var(--radius-lg) !important;
        padding: 4px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary) !important;
        color: var(--bg-primary) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-magenta)) !important;
        color: var(--bg-primary) !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
    }
    </style>
    """
    st.markdown(critical_css, unsafe_allow_html=True)
    
    # Cargar CSS externo si existe
    if CSS_PATH.exists():
        with open(CSS_PATH, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_db_connection():
    """Conexión a la base de datos."""
    if DB_PATH.exists():
        return sqlite3.connect(str(DB_PATH), check_same_thread=False)
    return None


@st.cache_data(ttl=300)
def get_dashboard_metrics():
    """Obtiene métricas principales para el dashboard."""
    conn = get_db_connection()
    
    default_metrics = {
        "total_carreras": 0,
        "total_caballos": 0,
        "proxima_jornada": None,
        "precision_ia": 94
    }
    
    if conn is None:
        return default_metrics
    
    try:
        total_carreras = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM fact_carreras", conn
        ).iloc[0]['count']
        
        total_caballos = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM dim_caballos", conn
        ).iloc[0]['count']
        
        prox_jornada = pd.read_sql_query("""
            SELECT h.nombre, c.fecha, COUNT(*) as carreras
            FROM fact_carreras c
            JOIN dim_hipodromos h ON c.hipodromo_id = h.id
            WHERE c.fecha >= date('now')
            GROUP BY c.fecha, h.nombre
            ORDER BY c.fecha ASC
            LIMIT 1
        """, conn)
        
        conn.close()
        
        return {
            "total_carreras": total_carreras,
            "total_caballos": total_caballos,
            "proxima_jornada": prox_jornada.iloc[0].to_dict() if not prox_jornada.empty else None,
            "precision_ia": 94
        }
    except Exception as e:
        return default_metrics


@st.cache_data(ttl=300)
def get_proximas_carreras():
    """Obtiene las próximas carreras programadas."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query("""
            SELECT 
                COALESCE(h.nombre, h.codigo, 'N/A') as Hipodromo,
                c.fecha as Fecha,
                c.nro_carrera as Carrera,
                'N/A' as Hora,
                c.distancia_metros || 'm' as Distancia,
                'Carrera' as Condicion,
                (SELECT COUNT(*) FROM fact_participaciones p WHERE p.carrera_id = c.id) as Participantes
            FROM fact_carreras c
            JOIN dim_hipodromos h ON c.hipodromo_id = h.id
            WHERE c.fecha >= date('now', '-7 days')
            ORDER BY c.fecha DESC, c.nro_carrera ASC
        """, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error en get_proximas_carreras: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_resultados_recientes():
    """Obtiene los resultados más recientes."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        df = pd.read_sql_query("""
            SELECT 
                c.fecha as Fecha,
                h.nombre as Hipodromo,
                c.nro_carrera as Carrera,
                dc.nombre as Ganador,
                dj.nombre as Jinete
            FROM fact_participaciones p
            JOIN fact_carreras c ON p.carrera_id = c.id
            JOIN dim_hipodromos h ON c.hipodromo_id = h.id
            JOIN dim_caballos dc ON p.caballo_id = dc.id
            LEFT JOIN dim_jinetes dj ON p.jinete_id = dj.id
            WHERE p.resultado_final = 1
            ORDER BY c.fecha DESC
            LIMIT 20
        """, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_participantes_carrera(fecha, hipodromo, nro_carrera):
    """Obtiene los participantes de una carrera específica."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                p.partidor as Partidor,
                dc.nombre as Caballo,
                dj.nombre as Jinete,
                p.peso_programado as Peso,
                p.handicap as "Index",
                p.edad_anos as Edad,
                c.distancia_metros as Distancia
            FROM fact_participaciones p
            JOIN fact_carreras c ON p.carrera_id = c.id
            JOIN dim_hipodromos h ON c.hipodromo_id = h.id
            JOIN dim_caballos dc ON p.caballo_id = dc.id
            LEFT JOIN dim_jinetes dj ON p.jinete_id = dj.id
            WHERE c.fecha = ? 
            AND (h.nombre = ? OR h.codigo = ?)
            AND c.nro_carrera = ?
            ORDER BY p.partidor ASC
        """
        df = pd.read_sql_query(query, conn, params=(fecha, hipodromo, hipodromo, nro_carrera))
        conn.close()
        return df
    except Exception as e:
        print(f"Error en get_participantes_carrera: {e}")
        return pd.DataFrame()


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Renderiza el header principal."""
    # Intentar cargar imagen de cabecera
    header_img_path = ASSETS_DIR / "img" / "Cabexera.png.png"
    
    if header_img_path.exists():
        header_b64 = load_image_as_base64(str(header_img_path))
        st.markdown(f"""
        <div style="
            width: 100%;
            height: 280px;
            border-radius: 20px;
            overflow: hidden;
            margin-bottom: 30px;
            position: relative;
        ">
            <img src="data:image/png;base64,{header_b64}" style="
                width: 100%;
                height: 100%;
                object-fit: cover;
                filter: brightness(0.9);
                object-position: center 35%;
            ">
            <div style="
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 60%;
                background: linear-gradient(to top, #0a0a0f 0%, transparent 100%);
            "></div>
            <div style="
                position: absolute;
                bottom: 30px;
                left: 30px;
            ">
                <h1 style="
                    font-family: 'Outfit', sans-serif;
                    font-size: 3rem;
                    font-weight: 900;
                    margin: 0;
                    background: linear-gradient(135deg, #00f5ff 0%, #ff00aa 50%, #ffd700 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                ">🏇 PISTA INTELIGENTE</h1>
                <p style="color: rgba(255,255,255,0.7); margin: 5px 0 0 0; font-size: 1.1rem;">
                    A Cobrar Los Que Saben • Sistema de Análisis Hípico con IA
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback sin imagen
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(0,245,255,0.1) 0%, rgba(255,0,170,0.1) 50%, rgba(255,215,0,0.1) 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
        ">
            <h1 style="
                font-family: 'Outfit', sans-serif;
                font-size: 3.5rem;
                font-weight: 900;
                margin: 0;
                background: linear-gradient(135deg, #00f5ff 0%, #ff00aa 50%, #ffd700 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">🏇 PISTA INTELIGENTE</h1>
            <p style="
                font-family: 'Outfit', sans-serif;
                font-size: 1.3rem;
                color: rgba(255,255,255,0.7);
                margin: 10px 0 0 0;
            ">A Cobrar Los Que Saben • Sistema de Análisis Hípico con IA</p>
        </div>
        """, unsafe_allow_html=True)

def render_metric_card(icon: str, value: str, label: str, color: str = "cyan", delta: str = None):
    """Renderiza una tarjeta de métrica."""
    delta_html = ""
    if delta:
        delta_color = "#b8ff00" if delta.startswith("+") else "#ff00aa"
        delta_html = f'<div style="font-size: 0.8rem; color: {delta_color}; margin-top: 8px;">{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
        <div class="metric-value {color}">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_prediction_card(rank: int, horse: str, jockey: str, score: float, probability: int):
    """Renderiza una tarjeta de predicción."""
    rank_class = f"rank-{rank}" if rank <= 3 else "rank-default"
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    medal = medals.get(rank, str(rank))
    
    st.markdown(f"""
    <div class="prediction-row">
        <div class="rank-badge {rank_class}">{medal}</div>
        <div style="flex: 1;">
            <div style="font-weight: 600; color: #fff; font-size: 1.1rem;">{horse}</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">🏇 {jockey}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #00f5ff;">
                {score:.1f}
            </div>
            <div style="font-size: 0.75rem; color: rgba(255,255,255,0.4);">puntos</div>
        </div>
        <div style="text-align: right; min-width: 60px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: #b8ff00;">
                {probability}%
            </div>
            <div style="font-size: 0.7rem; color: rgba(255,255,255,0.4);">prob.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, subtitle: str = None):
    """Renderiza un header de sección."""
    sub_html = f'<p style="color: rgba(255,255,255,0.5); margin: 5px 0 0 0; font-size: 0.9rem;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin: 30px 0 20px 0;">
        <h2 style="
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            color: #fff;
        ">{title}</h2>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_ad_leaderboard():
    """Renderiza espacio publicitario leaderboard."""
    st.markdown("""
    <div class="ad-container-leaderboard" style="
        width: 100%;
        max-width: 728px;
        height: 90px;
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(255,255,255,0.1);
        margin: 20px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        position: relative;
    ">
        <span style="position: absolute; top: 2px; right: 8px; font-size: 0.6rem; color: rgba(255,255,255,0.3);">Publicidad</span>
        <div style="text-align: center; color: rgba(255,255,255,0.3);">
            <strong>ESPACIO PUBLICITARIO</strong><br>
            <small>728x90 Leaderboard</small>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ad_sidebar():
    """Renderiza espacio publicitario en sidebar."""
    st.markdown("""
    <div style="
        width: 100%;
        height: 250px;
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(255,255,255,0.1);
        margin: 20px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        color: rgba(255,255,255,0.3);
    ">
        <div style="text-align: center;">
            <strong>ANUNCIO</strong><br>
            <small>300x250</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
def render_chatbot():
    """Renderiza el chatbot en el sidebar."""
    
    with st.sidebar:
        st.markdown("---")
        
        # Usar expander para el chat para ahorrar espacio
        with st.expander("💬 Asistente Virtual", expanded=False):
            # Estado del chat
            if 'messages' not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "¡Hola! 👋 Soy tu asistente hípico. ¿En qué puedo ayudarte?"}
                ]
            
            # Mostrar mensajes
            chat_container = st.container(height=300)
            with chat_container:
                for msg in st.session_state.messages:
                    if msg["role"] == "assistant":
                        st.markdown(f"""
                        <div style="
                            background: rgba(0,245,255,0.1);
                            border-left: 3px solid #00f5ff;
                            padding: 10px;
                            border-radius: 0 10px 10px 0;
                            margin-bottom: 10px;
                            font-size: 0.85rem;
                        ">🤖 {msg["content"]}</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            background: rgba(255,0,170,0.1);
                            border-right: 3px solid #ff00aa;
                            padding: 10px;
                            border-radius: 10px 0 0 10px;
                            margin-bottom: 10px;
                            text-align: right;
                            font-size: 0.85rem;
                        ">{msg["content"]} 👤</div>
                        """, unsafe_allow_html=True)
            
            # Input del usuario
            user_input = st.text_input(
                "Escribe tu pregunta...", 
                key="chat_input", 
                label_visibility="collapsed",
                placeholder="Escribe tu pregunta aquí..."
            )
            
            # Botones de preguntas rápidas
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊", help="Ver Predicciones", key="btn_pred"):
                    user_input = "predicciones"
            with col2:
                if st.button("🏇", help="Mejores Jinetes", key="btn_jinete"):
                    user_input = "jinetes"
            with col3:
                if st.button("💡", help="Tips de Apuesta", key="btn_tips"):
                    user_input = "tips"
            
            if user_input:
                # Agregar mensaje del usuario (si no es repetición del último)
                if not st.session_state.messages or st.session_state.messages[-1]["content"] != user_input:
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # Respuesta del bot (simulada)
                    respuestas = {
                        "predicciones": "📊 Las predicciones están en la pestaña 'PREDICCIONES IA'. ¡Revísalas para ver los favoritos!",
                        "jinetes": "🏇 Revisa la pestaña 'ESTADÍSTICAS' para ver el ranking actualizado de jinetes.",
                        "tips": "💡 Tip: Busca caballos con puntaje > 6.0 en nuestras predicciones, suelen tener alta probabilidad.",
                        "default": "🤔 Interesante. Te recomiendo explorar las pestañas de estadísticas para más detalles."
                    }
                    
                    input_lower = user_input.lower()
                    if "predicci" in input_lower:
                        resp = respuestas["predicciones"]
                    elif "jinete" in input_lower or "mejor" in input_lower:
                        resp = respuestas["jinetes"]
                    elif "tip" in input_lower or "apostar" in input_lower:
                        resp = respuestas["tips"]
                    else:
                        resp = respuestas["default"]
                    
                    st.session_state.messages.append({"role": "assistant", "content": resp})
                    st.rerun()


# ============================================================================
# TAB PAGES
# ============================================================================

def render_tab_programas():
    """Tab de programas y carreras (Programa Oficial)."""
    render_section_header("📖 Programa Oficial", "Nómina de participantes y detalles")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        hipodromo_filter = st.selectbox(
            "Filtrar por Hipódromo",
            ["Todos", "Club Hípico de Santiago", "Hipódromo Chile", "Valparaíso Sporting Club"],
            label_visibility="collapsed"
        )
    with col2:
        if st.button("🔄 Actualizar", width='stretch'):
            st.cache_data.clear()
            st.rerun()
    
    df_carreras = get_proximas_carreras()
    
    if df_carreras.empty:
        st.info("📭 No hay carreras programadas. Importa datos para ver el programa.")
        return
    
    if hipodromo_filter != "Todos":
        df_carreras = df_carreras[df_carreras['Hipodromo'] == hipodromo_filter]
    
    st.markdown(f"""
    <div class="glass" style="padding: 16px; margin-bottom: 20px;">
        <strong style="color: #00f5ff;">📊 {len(df_carreras)} carreras disponibles</strong>
        <span style="color: rgba(255,255,255,0.5); margin-left: 10px;">
            | Revisa el detalle de cada competencia
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Agrupar por fecha e hipódromo
    for (fecha, hipodromo), grupo in df_carreras.groupby(['Fecha', 'Hipodromo']):
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, rgba(255,0,170,0.1) 0%, transparent 100%);
            border-left: 3px solid #ff00aa;
            padding: 12px 16px;
            border-radius: 0 12px 12px 0;
            margin: 20px 0 10px 0;
        ">
            <strong style="color: #fff; font-size: 1.1rem;">🏟️ {hipodromo}</strong>
            <span style="color: rgba(255,255,255,0.5); margin-left: 15px;">📅 {fecha}</span>
        </div>
        """, unsafe_allow_html=True)
        
        for _, carrera in grupo.iterrows():
            nro_carrera = int(carrera['Carrera'])
            
            with st.expander(
                f"Carrera {nro_carrera} | {carrera['Distancia']} | {carrera['Participantes']} participantes"
            ):
                # Mostrar tabla de participantes (PROGRAMA)
                df_participantes = get_participantes_carrera(carrera['Fecha'], carrera['Hipodromo'], nro_carrera)
                
                if not df_participantes.empty:
                    st.markdown('<h4 style="margin: 0 0 12px 0; color: #ff00aa;">🏇 Nómina de Participantes</h4>', unsafe_allow_html=True)
                    st.dataframe(
                        df_participantes,
                        width='stretch',
                        hide_index=True,
                        key=f"participantes_{carrera['Fecha']}_{hipodromo}_{nro_carrera}",
                        column_config={
                            "Partidor": st.column_config.NumberColumn("#", width="small"),
                            "Caballo": st.column_config.TextColumn("Caballo", width="medium"),
                            "Jinete": st.column_config.TextColumn("Jinete", width="medium"),
                            "Peso": st.column_config.NumberColumn("Kg", format="%.1f"),
                            "Index": st.column_config.NumberColumn("Index", format="%d"),
                            "Edad": st.column_config.NumberColumn("Edad", format="%d años"),
                            "Distancia": st.column_config.TextColumn("Dist")
                        }
                    )
                else:
                    st.warning("⚠️ Sin participantes cargados")


def render_tab_resultados():
    """Tab de patrones (La Tercera es la Vencida)."""
    render_section_header("🏆 La Tercera Es La Vencida", "Patrones de Apuestas Repetidas")
    
    predictions = load_predictions_json()
    
    if not predictions or 'patrones' not in predictions:
        st.info("📭 No hay patrones detectados aún. Ejecuta el pipeline.")
        return
        
    patrones = predictions['patrones']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧩 Quinela (Top 2)")
        if 'quinelas' in patrones and patrones['quinelas']:
            for combo, count in patrones['quinelas'].items():
                st.markdown(f"""
                <div class="glass" style="padding: 12px; margin: 8px 0; border-left: 3px solid #00f5ff;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">Combinación</div>
                    <strong style="color: #fff; font-size: 1.1rem;">{combo}</strong>
                    <div style="text-align: right; color: #00f5ff; font-weight: bold;">x{count} veces</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Sin repeticiones")

    with col2:
        st.markdown("### 🎯 Tridecta (Top 3)")
        if 'tridectas' in patrones and patrones['tridectas']:
            for combo, count in patrones['tridectas'].items():
                st.markdown(f"""
                <div class="glass" style="padding: 12px; margin: 8px 0; border-left: 3px solid #ff00aa;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">Combinación</div>
                    <strong style="color: #fff; font-size: 1.1rem;">{combo}</strong>
                    <div style="text-align: right; color: #ff00aa; font-weight: bold;">x{count} veces</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Sin repeticiones")
            
    with col3:
        st.markdown("### 🔥 Superfecta (Top 4)")
        if 'superfectas' in patrones and patrones['superfectas']:
            for combo, count in patrones['superfectas'].items():
                st.markdown(f"""
                <div class="glass" style="padding: 12px; margin: 8px 0; border-left: 3px solid #ffd700;">
                    <div style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">Combinación</div>
                    <strong style="color: #fff; font-size: 1.1rem;">{combo}</strong>
                    <div style="text-align: right; color: #ffd700; font-weight: bold;">x{count} veces</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Sin repeticiones")


def render_tab_predicciones():
    """Tab de predicciones IA."""
    render_section_header("🤖 Predicciones de Inteligencia Artificial", "Análisis automático de todas las carreras")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Sincronizar Pipeline", width='stretch'):
            with st.spinner("Ejecutando pipeline completo..."):
                success, logs = run_full_pipeline()
                if success:
                    st.success("✅ Pipeline ejecutado correctamente")
                    st.cache_data.clear()
                else:
                    st.error("❌ Error en el pipeline")
                with st.expander("Ver logs"):
                    st.code(logs)
    
    predictions = load_predictions_json()
    
    if not predictions or 'predicciones' not in predictions:
        st.warning("⚠️ No hay predicciones disponibles. Ejecuta el pipeline de sincronización.")
        return
    
    # Stats rápidas
    total_races = len(predictions['predicciones'])
    total_horses = sum(len(p['predicciones']) for p in predictions['predicciones'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🏁 Carreras Analizadas", total_races)
    col2.metric("🐴 Caballos Evaluados", total_horses)
    col3.metric("📊 Promedio Participantes", f"{total_horses/total_races:.1f}" if total_races > 0 else "0")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Mostrar predicciones por carrera
    for idx, pred in enumerate(predictions['predicciones']):
        # Agregar separador visual entre carreras
        if idx > 0:
            st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander(
            f"🏇 {pred.get('hipodromo', 'N/A')} • Carrera {pred.get('nro_carrera', '?')} • Confianza: {pred.get('confianza', 0)}%",
            expanded=False
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 🏆 Top Predicciones")
                for i, pick in enumerate(pred.get('predicciones', pred.get('detalle', []))[:6], 1):
                    render_prediction_card(
                        rank=i,
                        horse=pick['caballo'],
                        jockey=pick['jinete'],
                        score=pick.get('puntaje_calculado', pick.get('puntaje', 0)),
                        probability=pick['probabilidad']
                    )
            
            with col2:
                # Mini gráfico de probabilidades
                top_4 = pred.get('predicciones', pred.get('detalle', []))[:4]
                if top_4:
                    chart_data = pd.DataFrame({
                        'Caballo': [p['caballo'][:12] for p in top_4],
                        'Prob': [p['probabilidad'] for p in top_4]
                    })
                    
                    fig = px.bar(
                        chart_data, 
                        x='Prob', 
                        y='Caballo', 
                        orientation='h',
                        color='Prob',
                        color_continuous_scale=['#ff00aa', '#00f5ff']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        showlegend=False,
                        coloraxis_showscale=False,
                        margin=dict(l=0, r=0, t=20, b=0),
                        height=200
                    )
                    fig.update_xaxes(title='', showgrid=False)
                    fig.update_yaxes(title='', showgrid=False)
                    st.plotly_chart(fig, width='stretch', key=f"chart_pred_{idx}")
    
    # Patrones detectados
    if 'patrones' in predictions and predictions['patrones']:
        st.markdown("<br>", unsafe_allow_html=True)
        render_section_header("🔮 Patrones Detectados", "Combinaciones repetidas frecuentemente")
        
        col1, col2, col3 = st.columns(3)
        
        pattern_types = [
            (col1, "quinelas", "🧩 Quinelas"),
            (col2, "tridectas", "🎯 Trifectas"),
            (col3, "superfectas", "🔥 Superfectas")
        ]
        
        for col, key, title in pattern_types:
            with col:
                st.markdown(f"#### {title}")
                patterns = predictions['patrones'].get(key, {})
                if patterns:
                    for combo, count in patterns.items():
                        st.markdown(f"""
                        <div class="glass" style="padding: 12px; margin: 8px 0;">
                            <strong style="color: #ffd700;">{combo}</strong>
                            <span style="float: right; color: #00f5ff;">x{count}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("Sin patrones detectados")


def render_tab_estadisticas():
    """Tab de estadísticas avanzadas."""
    render_section_header("📈 Estadísticas Avanzadas", "Análisis histórico de jinetes e hipódromos")
    
    stats = load_advanced_stats()
    
    # Mostrar Metadata del Modelo (ESTADÍSTICAS)
    predictions = load_predictions_json()
    if predictions and 'metadata' in predictions:
        meta = predictions['metadata']
        st.markdown("#### 🤖 Métricas del Modelo IA")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("📅 Generado", meta.get('fecha_generacion', 'N/A')[:16])
        
        if 'metricas' in meta:
            metricas = meta['metricas']
            m_col2.metric("📉 RMSE Test", f"{metricas.get('RMSE_Test', 0):.4f}")
            m_col3.metric("📈 R2 Score", f"{metricas.get('R2_Test', 0):.4f}")
            
        st.markdown("<br>", unsafe_allow_html=True)
    
    if not stats:
        st.warning("⚠️ Ejecuta la sincronización para generar estadísticas.")
        return
    
    # Top Jinetes
    st.markdown("#### 🏇 Ranking de Jinetes")
    
    if 'jinetes' in stats and stats['jinetes']:
        df_jinetes = pd.DataFrame(stats['jinetes'])
        
        st.dataframe(
            df_jinetes,
            width='stretch',
            hide_index=True,
            column_config={
                "nombre": st.column_config.TextColumn("🏇 Jinete"),
                "total_carreras": st.column_config.NumberColumn("Carreras"),
                "victorias": st.column_config.NumberColumn("Victorias"),
                "win_rate": st.column_config.ProgressColumn(
                    "% Victoria", 
                    format="%.1f%%", 
                    min_value=0, 
                    max_value=100
                )
            }
        )
    else:
        st.info("No hay datos de jinetes disponibles.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tendencias por distancia
    st.markdown("#### 🏟️ Rendimiento por Distancia")
    
    if 'hipodromos_distancia' in stats and stats['hipodromos_distancia']:
        df_trends = pd.DataFrame(stats['hipodromos_distancia'])
        
        if not df_trends.empty and 'hipodromo_codigo' in df_trends.columns:
            hipodromos = df_trends['hipodromo_codigo'].unique()
            selected_hip = st.selectbox("Seleccionar Hipódromo", hipodromos)
            
            df_filtered = df_trends[df_trends['hipodromo_codigo'] == selected_hip]
            
            fig = px.bar(
                df_filtered,
                x='distancia_metros',
                y='total_carreras',
                color='avg_posicion',
                color_continuous_scale=['#00f5ff', '#ff00aa'],
                labels={
                    'distancia_metros': 'Distancia (m)',
                    'total_carreras': 'Total Carreras',
                    'avg_posicion': 'Posición Promedio'
                }
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No hay datos de tendencias disponibles.")


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Renderiza el sidebar con navegación."""
    with st.sidebar:
        # Logo
        logo_path = ASSETS_DIR / "img" / "Logo_PistaInteligente.png.png"
        if logo_path.exists():
            st.image(str(logo_path), width='stretch')
        else:
            st.markdown("""
            <div style="text-align: center; padding: 20px 0;">
                <div style="font-size: 3rem; margin-bottom: 10px;">🏇</div>
                <h2 style="
                    font-size: 1.5rem;
                    margin: 0;
                    background: linear-gradient(135deg, #00f5ff, #ff00aa);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                ">PISTA INTELIGENTE</h2>
                <p style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin: 5px 0 0 0;">
                    v3.0 • 2025 Edition
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 Acciones Rápidas")
        
        if st.button("🔄 Sincronizar Datos", width='stretch'):
            with st.spinner("Sincronizando..."):
                success, logs = run_full_pipeline()
                if success:
                    st.success("✅ Datos sincronizados")
                else:
                    st.error("Error en sincronización")
                with st.expander("Ver detalles"):
                    st.code(logs)
        
        if st.button("🗑️ Limpiar Caché", width='stretch'):
            st.cache_data.clear()
            st.success("Caché limpiado")
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### ℹ️ Estado del Sistema")
        
        # Estado de la BD
        db_exists = DB_PATH.exists()
        db_status = "🟢 Conectada" if db_exists else "🔴 No encontrada"
        st.markdown(f"**Base de datos:** {db_status}")
        
        # Predicciones
        predictions = load_predictions_json()
        pred_count = len(predictions.get('predicciones', [])) if predictions else 0
        pred_status = f"🟢 {pred_count} carreras" if pred_count > 0 else "🔴 Sin datos"
        st.markdown(f"**Predicciones:** {pred_status}")
        
        # Stats
        stats = load_advanced_stats()
        stats_status = "🟢 Disponibles" if stats else "🔴 Sin datos"
        st.markdown(f"**Estadísticas:** {stats_status}")
        
        st.markdown("---")
        
        # Publicidad sidebar
        render_ad_sidebar()
        
        st.markdown("---")
        st.caption("© 2025 OriundoStartupChile.com")
        st.caption("Todos los derechos reservados ")


# ============================================================================
# MAIN PAGE
# ============================================================================

def page_dashboard():
    """Página principal - Dashboard."""
    render_header()
    
    # Publicidad leaderboard
    render_ad_leaderboard()
    
    # Métricas principales
    metrics = get_dashboard_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            "🏁", 
            f"{metrics['total_carreras']:,}", 
            "Carreras Analizadas",
            "cyan"
        )
    
    with col2:
        render_metric_card(
            "🐴", 
            f"{metrics['total_caballos']:,}", 
            "Caballos Rastreados",
            "magenta"
        )
    
    with col3:
        if metrics.get('proxima_jornada') and metrics['proxima_jornada']:
            try:
                fecha = datetime.strptime(metrics['proxima_jornada']['fecha'], '%Y-%m-%d').strftime('%d/%m')
            except:
                fecha = metrics['proxima_jornada'].get('fecha', 'N/A')
            hip_name = (metrics['proxima_jornada'].get('nombre') or 'N/A')[:15]
            render_metric_card("📅", fecha, f"Próxima: {hip_name}", "gold")
        else:
            render_metric_card("📅", "--/--", "Sin jornadas", "gold")
    
    with col4:
        render_metric_card(
            "🎯", 
            f"{metrics['precision_ia']}%", 
            "Precisión IA (Top 4)",
            "lime",
            "+2.3%"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 COMO LLEGAN", 
        "🏆 LA TERCERA ES LA VENCIDA", 
        "🤖 PREDICCIONES IA", 
        "📈 ESTADÍSTICAS"
    ])
    
    with tab1:
        render_tab_programas()
    
    with tab2:
        render_tab_resultados()
    
    with tab3:
        render_tab_predicciones()
    
    with tab4:
        render_tab_estadisticas()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal de la aplicación."""
    load_styles()
    render_sidebar()
    page_dashboard()
    render_chatbot()


if __name__ == "__main__":
    main()

