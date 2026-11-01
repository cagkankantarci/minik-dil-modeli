"""Videonun bölümleri için ek ölçümler → sonuc/ek_deneyler.json ve ekrana özet.

1. Sayma modeli (bigram): yalnızca "bu harften sonra hangisi kaç kez gelmiş" sayılır.
2. Sıradaki harf tahmini: eğitilmiş model "Bir gü" sonrası hangi harfe ne olasılık veriyor.
3. Sıcaklık: aynı başlangıç, 0,2 / 0,8 / 1,5.
4. Token: aynı anlamdaki Türkçe ve İngilizce metin kaç token (o200k_base).
5. Dikkat: bir cümlede son harfin en çok baktığı önceki harfler.
"""
import argparse, json
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from model import MinikGPT

KOK = Path(__file__).parent
sonuc = {}

metin = (KOK / "veri/temiz.txt").read_text(encoding="utf-8")
harfler = sorted(set(metin))
stoi = {c: i for i, c in enumerate(harfler)}

# --- 1. sayma modeli: 89 × 89'luk tablo ---
sayac = np.zeros((len(harfler), len(harfler)))
for x, y in zip(metin, metin[1:]):
    sayac[stoi[x], stoi[y]] += 1
olas = (sayac + 1) / (sayac + 1).sum(1, keepdims=True)
rng = np.random.default_rng(1337)
idx, yazi = stoi["B"], ["B"]
for _ in range(200):
    idx = rng.choice(len(harfler), p=olas[idx]); yazi.append(harfler[idx])
sonuc["sayma_modeli"] = {"tablo": f"{len(harfler)}x{len(harfler)}", "ornek": "".join(yazi)}
# "a" harfinden sonra en sık gelenler
a_sonra = sorted(((sayac[stoi["a"], j] / sayac[stoi["a"]].sum(), harfler[j]) for j in range(len(harfler))), reverse=True)[:5]
sonuc["sayma_modeli"]["a_dan_sonra"] = [(h, round(100 * p, 1)) for p, h in a_sonra]

# --- eğitilmiş model ---
yuk = torch.load(KOK / "sonuc/mac/model.pt", map_location="cpu")
ay = yuk["ayarlar"]
c = argparse.Namespace(katman=ay["katman"], kafa=ay["kafa"], boyut=ay["boyut"], baglam=ay["baglam"], harf=len(yuk["harfler"]))
assert yuk["harfler"] == harfler
model = MinikGPT(c); model.load_state_dict(yuk["model"]); model.eval()
kodla = lambda s: torch.tensor([[stoi[ch] for ch in s]])
coz = lambda t: "".join(harfler[i] for i in t)

# --- 2. sıradaki harf ---
with torch.no_grad():
    for bas in ("Bir gü", "Paşa hazretl", "— Ne"):
        p = F.softmax(model(kodla(bas))[0][0, -1], dim=-1)
        top = torch.topk(p, 5)
        sonuc.setdefault("siradaki_harf", {})[bas] = [(harfler[i], round(100 * v.item(), 1)) for v, i in zip(top.values, top.indices)]

# --- 3. sıcaklık ---
for t in (0.2, 0.8, 1.5):
    torch.manual_seed(1337)
    sonuc.setdefault("sicaklik", {})[str(t)] = coz(model.uret(kodla("Bir gün"), 200, sicaklik=t)[0].tolist())

# --- 4. token ---
enc = tiktoken.get_encoding("o200k_base")
ciftler = [
    ("Yapay zekâ modelleri bir sonraki kelimeyi tahmin ederek metin üretir.",
     "Artificial intelligence models generate text by predicting the next word."),
    ("Bilgisayarımda sıfırdan küçük bir dil modeli eğittim ve sonuçlar beni şaşırttı.",
     "I trained a small language model from scratch on my computer and the results surprised me."),
    ("Gidemeyeceklerimizden misiniz?", "Are you one of those we will not be able to go to?"),
]
sonuc["token"] = []
for tr, en in ciftler:
    sonuc["token"].append({"tr": tr, "tr_token": len(enc.encode(tr)), "tr_parcalar": [enc.decode([t]) for t in enc.encode(tr)],
                           "en": en, "en_token": len(enc.encode(en))})
parca = metin[:20000]
sonuc["token_ortalama"] = {"omer_seyfettin_harf_basina_token": round(len(enc.encode(parca)) / len(parca), 3),
                           "omer_seyfettin_token_basina_harf": round(len(parca) / len(enc.encode(parca)), 2)}

# --- 5. dikkat ---
cumle = "Derviş Hasan birdenbire durdu. Derviş"
with torch.no_grad():
    model(kodla(cumle), harita_tut=True)
harita = model.bloklar[-1].dikkat.son_harita[0]          # kafa × T × T
son = harita[:, -1, :]                                     # son harfin baktıkları
dikkat_sonuc = []
for k in range(son.shape[0]):
    top = torch.topk(son[k], 5)
    dikkat_sonuc.append([(int(i), cumle[int(i)], round(100 * v.item(), 1)) for v, i in zip(top.values, top.indices)])
sonuc["dikkat"] = {"cumle": cumle, "son_harf": cumle[-1], "kafa_basina_en_cok_bakilan": dikkat_sonuc,
                   "harita": harita.mean(0).tolist()}

# --- 6. kopyalama: verinin hiç görmediği bir adı önceki cümleden alabiliyor mu? ---
assert "Cemşit" not in metin
for bas in ("Cemşit Ağa kapıyı açtı. Cemşit ", "Cemşit Ağa kapıyı açtı. Cemşit A"):
    with torch.no_grad():
        p = F.softmax(model(kodla(bas), harita_tut=True)[0][0, -1], dim=-1)
    top = torch.topk(p, 4)
    katmanlar = []
    for b in model.bloklar:
        hm = b.dikkat.son_harita[0][:, -1, :].mean(0); t2 = torch.topk(hm, 3)
        katmanlar.append([(int(i), bas[int(i)], round(100 * v.item(), 1)) for v, i in zip(t2.values, t2.indices)])
    sonuc.setdefault("kopyalama", {})[bas] = {"tahmin": [(harfler[i], round(100 * v.item(), 1)) for v, i in zip(top.values, top.indices)],
                                              "dogru_harf": "A" if bas.endswith(" ") else "ğ",
                                              "dogru_harf_olasiligi": round(100 * p[stoi["A" if bas.endswith(" ") else "ğ"]].item(), 1),
                                              "katman_basina_bakilan": katmanlar}

(KOK / "sonuc/ek_deneyler.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in sonuc.items():
    if k == "dikkat":
        v = {kk: vv for kk, vv in v.items() if kk != "harita"}
    print(f"\n### {k}\n{json.dumps(v, ensure_ascii=False)[:1500]}")
