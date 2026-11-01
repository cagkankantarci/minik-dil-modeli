"""Minik, harf düzeyinde GPT. c: katman, kafa, boyut, baglam, harf (sözlük boyutu)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Dikkat(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.qkv = nn.Linear(c.boyut, 3 * c.boyut)
        s.cikis = nn.Linear(c.boyut, c.boyut)
        s.son_harita = None  # görselleştirme için son dikkat ağırlıkları

    def forward(s, x, harita_tut=False):
        B, T, C = x.shape
        q, k, v = s.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, s.c.kafa, C // s.c.kafa).transpose(1, 2) for t in (q, k, v))
        agirlik = (q @ k.transpose(-2, -1)) / math.sqrt(C // s.c.kafa)
        maske = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        agirlik = F.softmax(agirlik.masked_fill(~maske, float("-inf")), dim=-1)
        if harita_tut:
            s.son_harita = agirlik.detach().cpu()
        y = (agirlik @ v).transpose(1, 2).contiguous().view(B, T, C)
        return s.cikis(y)

class Blok(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(c.boyut), nn.LayerNorm(c.boyut)
        s.dikkat = Dikkat(c)
        s.mlp = nn.Sequential(nn.Linear(c.boyut, 4 * c.boyut), nn.GELU(), nn.Linear(4 * c.boyut, c.boyut))

    def forward(s, x, harita_tut=False):
        x = x + s.dikkat(s.n1(x), harita_tut)
        return x + s.mlp(s.n2(x))

class MinikGPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.tok = nn.Embedding(c.harf, c.boyut)
        s.konum = nn.Embedding(c.baglam, c.boyut)
        s.bloklar = nn.ModuleList(Blok(c) for _ in range(c.katman))
        s.son_norm = nn.LayerNorm(c.boyut)
        s.kafa = nn.Linear(c.boyut, c.harf)

    def forward(s, idx, hedef=None, harita_tut=False):
        T = idx.shape[1]
        x = s.tok(idx) + s.konum(torch.arange(T, device=idx.device))
        for b in s.bloklar:
            x = b(x, harita_tut)
        logit = s.kafa(s.son_norm(x))
        kayip = None if hedef is None else F.cross_entropy(logit.view(-1, logit.size(-1)), hedef.view(-1))
        return logit, kayip

    @torch.no_grad()
    def uret(s, idx, adet, sicaklik=0.8):
        for _ in range(adet):
            logit, _ = s(idx[:, -s.c.baglam:])
            olasilik = F.softmax(logit[:, -1, :] / sicaklik, dim=-1)
            idx = torch.cat([idx, torch.multinomial(olasilik, 1)], dim=1)
        return idx
