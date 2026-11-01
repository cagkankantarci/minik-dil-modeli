"""Cemşit testi: minik modele aynı cümleyi tamamlatır → sonuc/cemsit/minik.json"""
import argparse, json
from pathlib import Path

import torch

from model import MinikGPT

KOK = Path(__file__).parent
BAS = "Cemşit Ağa kapıyı açtı. Cemşit"
yuk = torch.load(KOK / "sonuc/mac/model.pt", map_location="cpu")
h, a = yuk["harfler"], yuk["ayarlar"]
m = MinikGPT(argparse.Namespace(katman=a["katman"], kafa=a["kafa"], boyut=a["boyut"], baglam=a["baglam"], harf=len(h)))
m.load_state_dict(yuk["model"]); m.eval()
s = {c: i for i, c in enumerate(h)}
ciktilar = []
for tohum in (1, 2, 3):
    torch.manual_seed(tohum)
    t = m.uret(torch.tensor([[s[c] for c in BAS]]), 120, sicaklik=0.8)[0].tolist()
    ciktilar.append("".join(h[i] for i in t)[len(BAS):])
(KOK / "sonuc/cemsit/minik.json").write_text(json.dumps({"baslangic": BAS, "sicaklik": 0.8, "ciktilar": ciktilar}, ensure_ascii=False, indent=1), encoding="utf-8")
for c in ciktilar: print(repr(c))
