
import os

def get_adsense_script() -> str:
    """
    Genera el script de Google AdSense utilizando el ID almacenado en variables de entorno.
    Retorna el HTML string para ser inyectado.
    """
    # Hardcoded script as requested for Google Authentication verification
    # Using the exact script provided to avoid any "obfuscation" issues
    # Script modificado para inyección directa en <HEAD> vía JavaScript
    # Esto cumple con el requisito estricto de estar en el HEAD en una SPA como Streamlit
    script_html = """
    <script>
        (function() {
            // 1. Inyectar Meta Tag de Verificación (NUEVO REQUISITO)
            var meta = document.createElement('meta');
            meta.name = "google-adsense-account";
            meta.content = "ca-pub-5579178295407019";
            document.head.appendChild(meta);
            console.log("AdSense Meta Tag injected");

            // 2. Inyectar Script Principal de AdSense
            var src = "https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-5579178295407019";
            if (!document.querySelector('script[src="' + src + '"]')) {
                var script = document.createElement('script');
                script.src = src;
                script.async = true;
                script.crossOrigin = "anonymous";
                document.head.appendChild(script);
                console.log("AdSense script injected into HEAD");
            }
        })();
    </script>
    """
    return script_html
