# Minik dil modeli: bilgisayarımda sıfırdan bir "ChatGPT" eğittim

YouTube videosundaki deneyin kodu ve gerçek sonuçları. Harf düzeyinde, 3,2 milyon parametreli minik bir GPT. Ömer Seyfettin'in 56 hikâyesiyle eğitiliyor; Apple M4'te 5.000 adım 4 dk 57 sn sürüyor.

> Bu ChatGPT değil. Aynı fikrin (transformer, "sıradaki parçayı tahmin et") oyuncak bir sürümü.

## Kurulum

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

NVIDIA kartlarda PyTorch'u kartınıza uygun CUDA sürümüyle kurun (ör. eski Pascal kartlar için `--index-url https://download.pytorch.org/whl/cu126`).

## Çalıştırma

```bash
.venv/bin/python veri_indir.py      # Vikikaynak'tan Ömer Seyfettin hikâyeleri → veri/omer_seyfettin.txt
.venv/bin/python temizle.py         # dipnot ve kaynakçaları atar → veri/temiz.txt
.venv/bin/python egit.py --veri veri/temiz.txt --cikti sonuc/benim
```

`egit.py` cihazı kendisi seçer (CUDA → Apple MPS → işlemci). Çıktı klasörüne `kayit.jsonl` (loss), `ornekler.md` (0, 100, 500, 1.000, 2.000, 5.000. adımdaki metinler), `ozet.json` ve `model.pt` yazar.

Videodaki diğer ölçümler:

```bash
.venv/bin/python ek_deneyler.py     # sayma modeli, sıradaki harf olasılıkları, temperature, token sayımı, attention, kopyalama testi
.venv/bin/python cemsit_minik.py    # "Cemşit Ağa kapıyı açtı. Cemşit" testi
```

## Dosyalar

| Dosya | Ne yapar |
|---|---|
| `model.py` | Minik GPT: token ve konum gömmesi, 4 transformer bloğu (attention + MLP), çıkış katmanı |
| `egit.py` | Eğitim döngüsü, loss ölçümü, örnek üretimi |
| `veri_indir.py` | Vikikaynak API'sinden hikâyeleri indirir (istekler arasında bekler) |
| `temizle.py` | Metni temizler |
| `ek_deneyler.py` | Videodaki ek ölçümler |
| `cemsit_minik.py` | Kopyalama testi |
| `sonuc/` | Videodaki gerçek çıktılar |

## Sonuçlar (videodaki koşu)

| | Mac | Windows laptop |
|---|---|---|
| Donanım | Apple M4 (MPS) | GTX 1050 Ti (CUDA 12.6) |
| 5.000 adım | 297 sn | ölçüm geçersiz* |
| Son validation loss | 1,6952957 | 1,6952961 |

\* Laptopun ekran kartı eğitim sırasında başka bir işle paylaşıldığı için süre karşılaştırması adil olmadı. Loss değerleri ilk beş haneye kadar aynı.

- Validation loss 3.700. adımda en düşük (1,616), 5.000. adımda 1,695: model ezberlemeye başlıyor (overfitting).
- Cemşit testi: minik model 3 denemede de "Ağa"yı bulamadı (`sonuc/cemsit/minik.json`). Aynı cümle Claude Code'a (`claude -p`, "devamını yaz") 3 kez verildi, 3'ünde de "Ağa" ile devam etti (`sonuc/cemsit/claude.json`).

## Veri ve lisans

- Eğitim verisi: Ömer Seyfettin (1884–1920) hikâyeleri. Yazar 1920'de öldüğü için eserleri kamu malı. Metinler [Vikikaynak](https://tr.wikisource.org/wiki/Ki%C5%9Fi:%C3%96mer_Seyfettin)'tan `veri_indir.py` ile indirilir, repoda yer almaz. İndirilen hikâyelerin listesi: `veri/kaynaklar.tsv`.
- Kod: MIT lisansı (`LICENSE`).

## Kaynaklar

- Vaswani ve diğ., "Attention Is All You Need" (2017): https://arxiv.org/abs/1706.03762
- Olsson ve diğ. (Anthropic), "In-context Learning and Induction Heads" (2022): https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Ouyang ve diğ. (OpenAI), "Training language models to follow instructions with human feedback" (2022): https://arxiv.org/abs/2203.02155
- Meta, Llama 3.1 model kartı: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
- OpenAI tiktoken: https://github.com/openai/tiktoken
