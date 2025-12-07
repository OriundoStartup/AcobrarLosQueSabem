
import os
from typing import List, Dict, Optional

class ChatService:
    def __init__(self):
        """Inicializa el servicio de chat con OpenAI."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        self.model = "gpt-3.5-turbo"
        
        if self.api_key:
            # Try OpenAI first
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.client_type = "openai"
            except Exception:
                # Fallback to Gemini if OpenAI fails or not installed
                gemini_key = os.getenv("GEMINI_API_KEY")
                if gemini_key:
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=gemini_key)
                        self.client = genai
                        self.client_type = "gemini"
                    except Exception as e:
                        print(f"Error initializing Gemini client: {e}")
                else:
                    print("⚠️ No valid API key for OpenAI or Gemini found.")

    def get_system_prompt(self, context_str: str = "") -> str:
        """Genera el prompt del sistema con el contexto o personalidad."""
        base_prompt = """Eres el Asistente Virtual Oficial de 'A Cobrar Los Que Saben', una plataforma de inteligencia artificial para pronósticos hípicos.
        
TU PERSONALIDAD:
- Eres experto, entusiasta y profesional en hípica (carreras de caballos).
- Usas terminología hípica correcta (ej: ejemplar, fusta, aprontes, figuraciones).
- Tus respuestas son concisas y directas, ideales para un chat lateral.
- SIEMPRE recuerdas que eres una IA y tus predicciones son probabilidades, no certezas.

TU CONOCIMIENTO:
- La plataforma usa modelos de Machine Learning (XGBoost/LightGBM) para predecir ganadores.
- Analiza factores como: velocidad, distancia, jinete, dividentos, y estado de la pista.
- Ofrece predicciones de Top 4, Quinelas, Trifectas y Superfectas.

INSTRUCCIONES DE RESPUESTA:
- Si te preguntan por "fijas" o "datos", sugiere revisar la pestaña 'PREDICCIONES IA'.
- Si te preguntan por jinetes, sugiere la pestaña 'ESTADÍSTICAS'.
- Si no sabes algo, admítelo y sugiere revisar los datos oficiales.
- Mantén un tono alentador pero responsable con las apuestas.
"""
        if context_str:
            base_prompt += f"\n\nCONTEXTO ACTUAL DE LA APP:\n{context_str}"
            
        return base_prompt

    def get_response(self, messages: List[Dict[str, str]], context: str = "") -> str:
        """
        Obtiene una respuesta del modelo.
        """
        if not self.client:
            return "⚠️ El módulo OpenAI no está disponible o no se configuró la API Key. Por favor verifica tu instalación (pip install openai) y tu archivo .env."

        try:
            # Preparar mensajes incluyendo el system prompt
            system_message = {"role": "system", "content": self.get_system_prompt(context)}
            full_messages = [system_message] + messages

            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Hubo un error al procesar tu mensaje: {str(e)}"
