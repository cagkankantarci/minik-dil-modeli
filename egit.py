"""Minik, harf düzeyinde bir GPT'yi sıfırdan eğitir.

Aynı betik Mac'te (MPS), NVIDIA kartlı bir bilgisayarda (CUDA) ya da yalnızca işlemcide koşar. Çıktı klasörüne:
  kayit.jsonl   adım, eğitim/doğrulama kaybı, geçen süre
  ornekler.md   seçilen adımlarda modelin ürettiği metin
  ozet.json     cihaz, parametre sayısı, toplam süre, adım başı süre
  model.pt      son ağırlıklar

Örnek: python egit.py --veri veri/omer_seyfettin.txt --cikti sonuc/mac
"""
import argparse, json, math, platform, subprocess, time
from pathlib import Path

import torch

from model import MinikGPT

p = argparse.ArgumentParser()
p.add_argument("--veri", default="veri/omer_seyfettin.txt")
p.add_argument("--cikti", default="sonuc/deneme")
p.add_argument("--cihaz", default="otomatik")
p.add_argument("--adim", type=int, default=5000)
p.add_argument("--katman", type=int, default=4)
p.add_argument("--kafa", type=int, default=4)
p.add_argument("--boyut", type=int, default=256)
p.add_argument("--baglam", type=int, default=128)
p.add_argument("--yigin", type=int, default=32)
p.add_argument("--oran", type=float, default=3e-4)
p.add_argument("--ornek-adimlari", default="0,100,500,1000,2000,5000")
p.add_argument("--baslangic", default="Bir gün")
p.add_argument("--tohum", type=int, default=1337)
a = p.parse_args()

if a.cihaz == "otomatik":
    a.cihaz = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(a.tohum)

# --- veri: her harf bir token ---
metin = Path(a.veri).read_text(encoding="utf-8")
harfler = sorted(set(metin))
stoi = {c: i for i, c in enumerate(harfler)}
kodla = lambda s: [stoi[c] for c in s if c in stoi]
coz = lambda l: "".join(harfler[i] for i in l)
veri = torch.tensor(kodla(metin), dtype=torch.long)
n = int(0.9 * len(veri))
egitim, dogrulama = veri[:n], veri[n:]

def yigin_al(parca):
    ix = torch.randint(len(parca) - a.baglam - 1, (a.yigin,))
    x = torch.stack([parca[i:i + a.baglam] for i in ix])
    y = torch.stack([parca[i + 1:i + a.baglam + 1] for i in ix])
    return x.to(a.cihaz), y.to(a.cihaz)

# --- model (model.py) ---

def senkron():
    if a.cihaz == "cuda": torch.cuda.synchronize()
    elif a.cihaz == "mps": torch.mps.synchronize()

@torch.no_grad()
def kayip_olc(model, tur=20):
    model.eval()
    sonuc = {}
    for ad, parca in (("egitim", egitim), ("dogrulama", dogrulama)):
        sonuc[ad] = sum(model(*yigin_al(parca))[1].item() for _ in range(tur)) / tur
    model.train()
    return sonuc

def ornek_uret(model):
    model.eval()
    torch.manual_seed(a.tohum)
    bas = torch.tensor([kodla(a.baslangic)], device=a.cihaz)
    yazi = coz(model.uret(bas, 300)[0].tolist())
    model.train()
    return yazi

def cihaz_adi():
    if a.cihaz == "cuda":
        return torch.cuda.get_device_name(0)
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return platform.processor() or platform.machine()

model = MinikGPT(argparse.Namespace(katman=a.katman, kafa=a.kafa, boyut=a.boyut, baglam=a.baglam, harf=len(harfler))).to(a.cihaz)
parametre = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.oran)
ornek_adimlari = {int(x) for x in a.ornek_adimlari.split(",")}

out = Path(a.cikti); out.mkdir(parents=True, exist_ok=True)
kayit = open(out / "kayit.jsonl", "w", encoding="utf-8")
ornekler = open(out / "ornekler.md", "w", encoding="utf-8")
print(f"cihaz={a.cihaz} parametre={parametre:,} harf={len(harfler)} veri={len(veri):,} karakter")

senkron(); t0 = time.time(); egitim_suresi = 0.0
for adim in range(a.adim + 1):
    if adim % 100 == 0 or adim in ornek_adimlari:
        senkron(); egitim_suresi += time.time() - t0
        k = kayip_olc(model)
        kayit.write(json.dumps({"adim": adim, **k, "sure_sn": round(egitim_suresi, 2)}) + "\n"); kayit.flush()
        print(f"adim {adim:5d}  egitim {k['egitim']:.3f}  dogrulama {k['dogrulama']:.3f}  {egitim_suresi:7.1f} sn")
        if adim in ornek_adimlari:
            ornekler.write(f"## Adım {adim} (kayıp {k['dogrulama']:.3f})\n\n```\n{ornek_uret(model)}\n```\n\n"); ornekler.flush()
        senkron(); t0 = time.time()
    if adim == a.adim:
        break
    x, y = yigin_al(egitim)
    _, kayip = model(x, y)
    opt.zero_grad(set_to_none=True)
    kayip.backward()
    opt.step()

ozet = {
    "cihaz": a.cihaz,
    "gpu": cihaz_adi(),
    "torch": torch.__version__,
    "parametre": parametre,
    "adim": a.adim,
    "egitim_suresi_sn": round(egitim_suresi, 1),
    "adim_basi_ms": round(1000 * egitim_suresi / a.adim, 2),
    "son_dogrulama_kaybi": k["dogrulama"],
    "ayarlar": vars(a),
}
(out / "ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
torch.save({"model": model.state_dict(), "harfler": harfler, "ayarlar": vars(a)}, out / "model.pt")
print(json.dumps(ozet, ensure_ascii=False))
