
import os

def get_adsense_script() -> str:
    """
    Genera el script de Google AdSense utilizando el ID almacenado en variables de entorno.
    Retorna el HTML string para ser inyectado.
    """
    adsense_id = os.getenv("GOOGLE_ADSENSE_ID")
    
    if not adsense_id:
        return "<!-- Google AdSense ID not configured -->"
        
    script_html = f"""
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={adsense_id}"
     crossorigin="anonymous"></script>
    """
    return script_html
