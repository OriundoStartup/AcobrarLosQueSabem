
import os

def get_adsense_script() -> str:
    """
    Genera el script de Google AdSense utilizando el ID almacenado en variables de entorno.
    Retorna el HTML string para ser inyectado.
    """
    # Hardcoded script as requested for Google Authentication verification
    # Using the exact script provided to avoid any "obfuscation" issues
    script_html = """
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-5579178295407019"
     crossorigin="anonymous"></script>
    """
    return script_html
