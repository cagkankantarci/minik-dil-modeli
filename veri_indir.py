"""Vikikaynak'tan Ömer Seyfettin (ö. 1920, telifi düşmüş) hikâyelerini düz metin olarak indirir."""
import json, re, ssl, sys, time, urllib.error, urllib.parse, urllib.request

try:
    import certifi
    CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    CTX = ssl.create_default_context()
from pathlib import Path

API = "https://tr.wikisource.org/w/api.php"
UA = {"User-Agent": "cagkan-ai-egitim/1.0 (egitim amacli deneme)"}
OUT = Path(__file__).parent / "veri"

def api(**p):
    p["format"] = "json"
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(p), headers=UA)
    for deneme in range(6):
        try:
            return json.load(urllib.request.urlopen(req, timeout=30, context=CTX))
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(10 * (deneme + 1))  # sunucuyu yormamak için bekle
    raise RuntimeError("429 sürüyor")

# Yazar sayfasındaki eser bağlantıları
links, cont = [], {}
while True:
    d = api(action="query", prop="links", titles="Kişi:Ömer Seyfettin", plnamespace=0, pllimit="max", **cont)
    for page in d["query"]["pages"].values():
        links += [l["title"] for l in page.get("links", [])]
    if "continue" not in d: break
    cont = d["continue"]
print(len(links), "bağlantı")

metinler, kayit = [], []
for baslik in links:
    d = api(action="query", prop="extracts", explaintext=1, titles=baslik)
    for page in d["query"]["pages"].values():
        t = page.get("extract") or ""
        if len(t) < 1500: continue
        metinler.append(t.strip()); kayit.append(f"{page['title']}\t{len(t)}")
    time.sleep(1.5)

ham = "\n\n".join(metinler)
ham = re.sub(r"\n{3,}", "\n\n", ham)
OUT.mkdir(exist_ok=True)
(OUT / "omer_seyfettin.txt").write_text(ham, encoding="utf-8")
(OUT / "kaynaklar.tsv").write_text("\n".join(kayit), encoding="utf-8")
print(len(metinler), "eser,", len(ham), "karakter")
