# -*- coding: utf-8 -*-
"""F.1 (calibration) et F.3 (gain en monde charniere) pour le BatteryPFN.

F.1 - CALIBRATION. Couverture empirique des quantiles predits, mesuree
SEPAREMENT sur le pool d'entrainement et sur le pool de validation (seeds
disjointes). Un q90 bien calibre couvre 90 % des cibles. Si la couverture tient
sur l'entrainement et derive sur la validation, le pool fige est epuise :
l'entrainement doit s'arreter et le fait doit etre rapporte, pas compense.

F.3 - GAIN EN MONDE CHARNIERE. Le critere de la branche :

    le PFN doit recuperer au moins UN TIERS de l'ecart d'APE sur n@70 entre
    monde Arrhenius et monde charniere, mesure contre la borne moindres carres.

Borne C.2a (L3 + Arrhenius, exposant fige a 0.8 - exactement le modele soumis),
mesuree sur 180 laboratoires :

    Arrhenius 13.53 %   charniere 15.81 %   ecart 2.28 pts
    seuil applicable : APE n@70 <= 15.05 % en monde charniere

L'evaluation se fait en leave-one-cell-out sur des laboratoires de VALIDATION,
donc jamais vus a l'entrainement, et la verite de reference est le SOH latent -
erreur de modele comprise - exactement comme pour la borne.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data import (PoolShard, liste_shards, normalise, denormalise_soh,
                   cible_depuis_observation, _sous_ech)

# Borne C.2a, figee. Toute comparaison de F.3 se fait contre ces valeurs.
BORNE_ARRHENIUS = 13.53
BORNE_CHARNIERE = 15.81
BORNE_ECART = BORNE_CHARNIERE - BORNE_ARRHENIUS
SEUIL_F3 = BORNE_CHARNIERE - BORNE_ECART / 3.0        # 15.05 %

GRILLE_EVAL = np.unique(np.round(np.geomspace(1.0, 12000.0, 96))).astype(float)


def _n_at(cycles, y, seuil=70.0):
    """Cycle ou y franchit `seuil`, NaN si jamais."""
    env = np.minimum.accumulate(np.asarray(y, float))
    sous = np.flatnonzero(env <= seuil)
    if not sous.size:
        return np.nan
    j = int(sous[0])
    if j == 0:
        return float(cycles[0])
    return float(np.interp(seuil, [env[j], env[j - 1]],
                           [cycles[j], cycles[j - 1]]))


@torch.no_grad()
def evalue_f3(modele, split="valid", n_labs=300, n_ctx=32, seed=0,
              pool_dir=None, device="cpu"):
    """APE sur n@70, par structure de b(T), en leave-one-cell-out."""
    modele.eval()
    shards = liste_shards(split, pool_dir)
    sh = PoolShard(*shards[0])
    rng = np.random.default_rng(seed)
    ids = rng.choice(sh.lab_ids, min(n_labs, len(sh.lab_ids)), replace=False)

    i_med = modele.quantiles.index(0.50)
    lignes = []
    for lid in ids:
        labs = sh.labs[lid]
        k = len(labs)
        cible = int(rng.integers(k))

        ctx = []
        for ci in range(k):
            if ci == cible:
                continue
            r = labs.iloc[ci]
            t = sh.traj[(lid, ci)]
            n, y = _sous_ech(t.cycle.to_numpy(np.float64),
                             t.soh.to_numpy(np.float64), n_ctx, rng)
            ctx.append(normalise(np.full(len(n), r["T"]),
                                 np.full(len(n), r["C"]), n, y))
        Xc = torch.tensor(np.concatenate(ctx)[None], dtype=torch.float32,
                          device=device)

        rc = labs.iloc[cible]
        Xq = torch.tensor(
            normalise(np.full(len(GRILLE_EVAL), rc["T"]),
                      np.full(len(GRILLE_EVAL), rc["C"]), GRILLE_EVAL)[None],
            dtype=torch.float32, device=device)

        q = modele(Xc, Xq)[0].cpu().numpy()
        traj_pred = denormalise_soh(q[:, i_med])

        tq = sh.traj[(lid, cible)]
        traj_vrai = cible_depuis_observation(
            tq.cycle.to_numpy(np.float64), tq.soh.to_numpy(np.float64),
            rc, GRILLE_EVAL)

        n70_v = _n_at(GRILLE_EVAL, traj_vrai)
        n70_p = _n_at(GRILLE_EVAL, traj_pred)
        note = traj_vrai >= 70.0
        rmse = (float(np.sqrt(np.mean((traj_pred[note] - traj_vrai[note]) ** 2)))
                if note.sum() >= 5 else np.nan)
        ape = (abs(n70_p - n70_v) / n70_v * 100.0
               if np.isfinite(n70_v) and np.isfinite(n70_p) else np.nan)
        lignes.append(dict(lab=int(lid), structure=rc["structure"],
                           p_vrai=float(rc["p"]), T=float(rc["T"]),
                           C=float(rc["C"]), n70_vrai=n70_v, n70_predit=n70_p,
                           ape_n70=ape, rmse_zone_notee=rmse))
    modele.train()
    return pd.DataFrame(lignes)


def resume_f3(d):
    """Verdict F.3 : le seuil est-il atteint en monde charniere ?"""
    o = d.dropna(subset=["ape_n70"])
    par_struct = o.groupby("structure").ape_n70.median()
    ch = float(par_struct.get("charniere", np.nan))
    ar = float(par_struct.get("arrhenius", np.nan))
    return dict(
        n=len(o),
        ape_globale=round(float(o.ape_n70.median()), 2),
        ape_arrhenius=round(ar, 2), ape_charniere=round(ch, 2),
        ecart=round(ch - ar, 2) if np.isfinite(ch - ar) else np.nan,
        rmse_zone_notee=round(float(o.rmse_zone_notee.median()), 2),
        seuil_F3=SEUIL_F3,
        borne_charniere=BORNE_CHARNIERE,
        gain_vs_borne=round(BORNE_CHARNIERE - ch, 2) if np.isfinite(ch) else np.nan,
        part_ecart_recuperee=(round((BORNE_CHARNIERE - ch) / BORNE_ECART, 3)
                              if np.isfinite(ch) else np.nan),
        F3_atteint=bool(np.isfinite(ch) and ch <= SEUIL_F3))


@torch.no_grad()
def evalue_calibration(modele, generateur, n_batches=8, device="cpu"):
    """F.1 : couverture empirique de chaque quantile, sur un pool donne."""
    modele.eval()
    it = iter(generateur)
    sous, tot, pertes = None, 0, []
    from .model import perte_pinball
    for _ in range(n_batches):
        Xc, Xq, Y, W = next(it)
        Xc = torch.tensor(Xc, device=device)
        Xq = torch.tensor(Xq, device=device)
        Y = torch.tensor(Y, device=device)
        q = modele(Xc, Xq)
        pertes.append(float(perte_pinball(q, Y, modele.quantiles,
                                          poids=torch.tensor(W, device=device))))
        s = (Y.unsqueeze(-1) <= q).float().sum(dim=(0, 1)).cpu().numpy()
        sous = s if sous is None else sous + s
        tot += Y.numel()
    modele.train()
    return dict(perte=float(np.mean(pertes)),
                couverture={q: round(float(s / tot), 4)
                            for q, s in zip(modele.quantiles, sous)})


def ecart_calibration(couv):
    """Ecart absolu moyen entre couverture visee et couverture obtenue."""
    return round(float(np.mean([abs(v - q) for q, v in couv.items()])), 4)
