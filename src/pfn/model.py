# -*- coding: utf-8 -*-
"""Architecture du BatteryPFN.

Forme de la tache
-----------------
Un exemple = un laboratoire dont une cellule est retiree.

    contexte  (n_ctx_total, 4)   les 5 autres cellules : (T, C, cycle, SOH)
    requete   (n_query, 3)       la cellule retiree : (T, C, cycle), SANS SOH
    cible     (n_query,)         son SOH latent a ces cycles

Le reseau doit donc apprendre, en un seul passage avant, ce qu'un ajustement par
moindres carres fait par optimisation : lire dans 5 trajectoires la loi qui les
engendre, puis l'appliquer a une condition (T, C) qu'il n'a jamais vue. C'est
la definition d'un PFN - l'inference bayesienne amortie dans les poids.

Choix d'architecture, et leurs raisons
--------------------------------------
1. AUCUN ENCODAGE POSITIONNEL. Les tokens de contexte sont un ENSEMBLE non
   ordonne : permuter deux points, ou deux cellules, ne doit rien changer. Le
   cycle et les conditions sont deja dans les variables du token ; un encodage
   positionnel ferait apprendre au reseau un ordre qui n'existe pas et le
   rendrait sensible a l'ordre de lecture des fichiers.

2. ATTENTION BIDIRECTIONNELLE COMPLETE sur [contexte ; requete]. Les tokens de
   requete ne portent AUCUNE information de cible - seulement (T, C, cycle) -
   donc il n'y a rien a masquer : aucune fuite n'est possible. Laisser les
   requetes s'attendre entre elles est voulu : on predit une TRAJECTOIRE, pas
   des points independants, et la coherence entre cycles voisins est une
   propriete qu'on veut que le reseau produise lui-meme.

3. TETE A QUANTILES, pas une moyenne. Le point de la branche est de
   MARGINALISER sur ce que les donnees ne determinent pas - la structure de
   b(T), et surtout l'exposant p. Un reseau entraine a l'erreur quadratique
   rendrait la moyenne a posteriori et perdrait toute l'information de
   dispersion, qui est precisement le produit du prior. Les quantiles sont
   contraints CROISSANTS par construction (cumul de softplus) : un croisement
   de quantiles rendrait la calibration ininterpretable.

4. PERTE PINBALL. C'est la perte propre pour des quantiles ; minimisee, elle
   rend les quantiles vrais de la loi a posteriori du prior.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantiles predits. q50 sert de prediction ponctuelle ; les autres portent
# l'incertitude et servent au controle de calibration F.1.
QUANTILES = (0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95)


class BlocTransformeur(nn.Module):
    """Pre-norm : plus stable que post-norm sans reglage de warmup fin."""

    def __init__(self, d, n_tetes, mult_ff=4, p_drop=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, n_tetes, dropout=p_drop,
                                         batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, mult_ff * d), nn.GELU(),
                                nn.Linear(mult_ff * d, d))
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.n1(x)
        a, _ = self.att(h, h, h, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class BatteryPFN(nn.Module):
    """4 couches, d_model 256 par defaut."""

    def __init__(self, d_model=256, n_couches=4, n_tetes=8, mult_ff=4,
                 p_drop=0.0, quantiles=QUANTILES):
        super().__init__()
        self.quantiles = tuple(quantiles)
        nq = len(self.quantiles)
        self.d_model = d_model

        # Deux encodeurs distincts : un token de contexte porte 4 variables
        # (SOH compris), un token de requete seulement 3. Les projeter avec la
        # meme couche en mettant un zero a la place du SOH ferait apprendre au
        # reseau que "SOH = 0 normalise" (soit 85 %) est une valeur plausible.
        self.enc_ctx = nn.Sequential(nn.Linear(4, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))
        self.enc_req = nn.Sequential(nn.Linear(3, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))
        # Marqueur appris contexte / requete : sans lui les deux familles de
        # tokens seraient indiscernables dans l'espace latent.
        self.marqueur = nn.Parameter(torch.zeros(2, d_model))

        self.blocs = nn.ModuleList([
            BlocTransformeur(d_model, n_tetes, mult_ff, p_drop)
            for _ in range(n_couches)])
        self.norme = nn.LayerNorm(d_model)

        # Tete : un niveau de base + des increments POSITIFS, ce qui garantit
        # q_1 <= q_2 <= ... quelles que soient les valeurs des poids.
        self.tete = nn.Linear(d_model, nq)
        nn.init.zeros_(self.tete.bias)
        nn.init.normal_(self.tete.weight, std=0.02)

    def forward(self, x_ctx, x_req):
        """x_ctx (B, Nc, 4), x_req (B, Nq, 3) -> quantiles (B, Nq, nq)."""
        hc = self.enc_ctx(x_ctx) + self.marqueur[0]
        hq = self.enc_req(x_req) + self.marqueur[1]
        h = torch.cat([hc, hq], dim=1)
        for b in self.blocs:
            h = b(h)
        h = self.norme(h[:, x_ctx.shape[1]:])          # on ne lit que la requete

        brut = self.tete(h)
        base = brut[..., :1]
        incr = F.softplus(brut[..., 1:])               # strictement positifs
        return torch.cat([base, base + torch.cumsum(incr, dim=-1)], dim=-1)

    @torch.no_grad()
    def predire(self, x_ctx, x_req, i_median=None):
        q = self(x_ctx, x_req)
        i = self.quantiles.index(0.50) if i_median is None else i_median
        return q[..., i], q


def perte_pinball(pred, cible, quantiles=QUANTILES, poids=None):
    """Perte pinball moyenne sur les quantiles et les points.

    `pred` (B, Nq, nq), `cible` (B, Nq). `poids` (B, Nq) optionnel, pour
    ponderer certains cycles - par exemple la zone notee SOH >= 70 %.
    """
    tau = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    e = cible.unsqueeze(-1) - pred
    l = torch.maximum(tau * e, (tau - 1.0) * e).mean(dim=-1)
    if poids is not None:
        return (l * poids).sum() / poids.sum().clamp_min(1.0)
    return l.mean()


def couverture_empirique(pred, cible, quantiles=QUANTILES):
    """Part des cibles sous chaque quantile predit. C'est le test F.1.

    Un q90 bien calibre doit couvrir 90 % des cibles. On la mesure separement
    sur le pool d'entrainement et sur celui de validation : si elle tient sur
    l'un et derive sur l'autre, le pool est epuise.
    """
    with torch.no_grad():
        sous = (cible.unsqueeze(-1) <= pred).float().mean(dim=(0, 1))
    return {q: float(v) for q, v in zip(quantiles, sous)}


def compte_parametres(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
