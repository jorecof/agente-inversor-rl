"""Exporta el informe a HTML autónomo. Dependencia opcional: pip install mistune."""
import argparse
import base64
import html
import mimetypes
import re
import unicodedata
from pathlib import Path
import mistune


STYLE = """
:root{color-scheme:light;--ink:#192831;--muted:#50616b;--accent:#245b72;--line:#dbe3e7;--paper:#fff;--bg:#f1f4f6}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.7 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
main{max-width:1100px;margin:40px auto;padding:58px 64px;background:var(--paper);border:1px solid var(--line);border-radius:12px}
h1,h2,h3{color:#17242d;line-height:1.25;text-wrap:balance}h1{font-size:2.25rem;font-weight:700;margin:0 0 28px;max-width:900px}
h2{font-size:1.5rem;margin:48px 0 18px;scroll-margin-top:24px}h3{font-size:1.15rem;margin:30px 0 14px;scroll-margin-top:24px}p{margin:0 0 18px}a{color:var(--accent);text-underline-offset:3px;overflow-wrap:anywhere}a:hover{color:#12394b}
ul,ol{padding-left:26px}li{margin:7px 0}code{font:0.86em/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;background:#edf2f5;border-radius:4px;padding:2px 5px;overflow-wrap:anywhere}
pre{white-space:pre-wrap;background:#edf2f5;padding:18px;border-radius:7px;overflow:auto}pre code{padding:0}
.table-wrap{overflow-x:auto;margin:24px 0;border:1px solid var(--line);border-radius:7px}table{width:100%;border-collapse:collapse;font-size:.87rem;line-height:1.5;font-variant-numeric:tabular-nums}
th,td{padding:12px 14px;text-align:right;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf3f6;font-weight:650;color:#183746}td:first-child,th:first-child{text-align:left}tbody tr:nth-child(even){background:#f8fafb}tbody tr:last-child td{border-bottom:0}
figure{margin:28px 0 32px}figure img{display:block;width:100%;height:auto;border:1px solid #e5eaed;border-radius:5px}figcaption{font-size:.88rem;color:var(--muted);margin:9px 0 0}
nav{border:1px solid var(--line);border-radius:7px;background:#f8fafb;padding:16px 20px;margin:28px 0 34px}summary{cursor:pointer;font-weight:600}nav ol{columns:2;column-gap:30px;font-size:.9rem;margin-bottom:0}nav li{break-inside:avoid;margin:5px 0}
.toolbar{display:flex;justify-content:flex-end;margin-bottom:20px}button{font:inherit;font-size:.9rem;background:white;border:1px solid #a9bac4;color:#244957;border-radius:6px;padding:8px 14px;cursor:pointer}button:hover{background:#edf3f6}
@media(max-width:760px){main{margin:0;padding:28px 22px;border:0;border-radius:0}body{font-size:16px}h1{font-size:1.8rem}h2{font-size:1.3rem}nav ol{columns:1}th,td{padding:9px}}
@media print{@page{size:A4;margin:18mm 16mm}body{background:white;font-size:10.5pt;line-height:1.5}main{max-width:none;margin:0;padding:0;border:0;border-radius:0}.toolbar,nav{display:none}h1{font-size:24pt}h2{font-size:15pt;margin-top:24pt;break-after:avoid}h3{font-size:12pt;break-after:avoid}p{orphans:3;widows:3}figure{break-inside:avoid;margin:16pt 0}img{max-height:210mm;object-fit:contain}table{font-size:8pt}tr{break-inside:avoid}thead{display:table-header-group}.table-wrap{overflow:visible}a{color:inherit}code{font-size:8.5pt}}
"""


def export(source, target):
    text = source.read_text(encoding="utf-8")
    body = mistune.create_markdown(escape=True, plugins=["table", "url"])(text)
    entries = []

    def heading(match):
        level, title = match.group(1), match.group(2)
        plain = html.unescape(re.sub("<[^>]+>", "", title))
        slug = unicodedata.normalize("NFKD", plain).encode("ascii", "ignore").decode().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
        identifier = f"seccion-{len(entries)+1}-{slug}"
        entries.append((level, title, identifier))
        return f'<h{level} id="{identifier}">{title}</h{level}>'

    body = re.sub(r"<h([23])>(.*?)</h\1>", heading, body, flags=re.S)

    def embed(match):
        attributes, name, rest = match.group(1), html.unescape(match.group(2)), match.group(3)
        path = (source.parent/name).resolve()
        if path.parent != source.parent.resolve() or not path.is_file():
            raise ValueError(f"Imagen local inválida: {name}")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img{attributes}src="data:{mime};base64,{data}"{rest}>'

    body = re.sub(r'<img([^>]*?)src="([^"]+)"([^>]*?)>', embed, body)
    body = re.sub(r'<p>(<img[^>]+>)</p>',
                  lambda m: '<figure>'+m.group(1)+'<figcaption>'+re.search(r'alt="([^"]*)"', m.group(1)).group(1)+'</figcaption></figure>', body)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    toc = '<nav aria-label="Contenido"><details><summary>Contenido del informe</summary><ol>'
    toc += ''.join(f'<li><a href="#{identifier}">{title}</a></li>' for level, title, identifier in entries if level == '2')
    toc += '</ol></details></nav>'
    first_end = body.find('</h1>')+len('</h1>')
    body = body[:first_end]+toc+body[first_end:]
    document = '<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n'
    document += '<title>Informe de inversión con Q-learning y DQN</title>\n<style>'+STYLE+'</style>\n</head>\n<body><main>'
    document += '<div class="toolbar"><button type="button" onclick="window.print()">Imprimir o guardar como PDF</button></div>'
    document += body+'</main></body></html>\n'
    target.write_text(document, encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("results/main/INFORME.md"))
    parser.add_argument("--output", type=Path, default=Path("results/main/INFORME.html"))
    args = parser.parse_args()
    print(export(args.source, args.output))
