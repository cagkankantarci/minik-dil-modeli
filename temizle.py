"""veri/omer_seyfettin.txt içinden dipnot, kaynakça ve vikikaynak işaretlerini atar → veri/temiz.txt"""
import re
from pathlib import Path

VERI = Path(__file__).parent / "veri"
ham = (VERI / "omer_seyfettin.txt").read_text(encoding="utf-8")

metin = ham
# Her eserin sonundaki "== Dipnotlar ==" / "== Kaynakça ==" bölümünü bir sonraki esere kadar sil
metin = re.sub(r"\n== (Dipnotlar|Kaynakça|Notlar) ==.*?(?=\n\n[^\n=]{200,}|\Z)", "\n", metin, flags=re.S)
metin = re.sub(r"^Kaynakça:.*$", "", metin, flags=re.M)
metin = re.sub(r"^=+ .* =+$", "", metin, flags=re.M)          # kalan başlık satırları
metin = metin.replace("­", "")                               # yumuşak tire
metin = re.sub(r"[🙝🙟*±&`´]", "", metin)
metin = metin.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("«", '"').replace("»", '"')
metin = re.sub(r"[ \t]+", " ", metin)
metin = re.sub(r"\n{3,}", "\n\n", metin).strip() + "\n"
(VERI / "temiz.txt").write_text(metin, encoding="utf-8")
print(len(ham), "→", len(metin), "karakter,", len(set(metin)), "farklı harf")
