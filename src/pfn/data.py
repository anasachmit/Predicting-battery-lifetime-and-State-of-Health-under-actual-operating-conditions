# -*- coding: utf-8 -*-
"""Chargeur du pool fige vers les batches du BatteryPFN.

La tache, telle que l'impose l'interface de soumission
-----------------------------------------------------
Au moment du test, le modele recoit UNIQUEMENT un couple (T, C-rate) :

    def predict_soh(self, temperature_degC, c_rate) -> SOH pour n = 1, 2, 3, ...

Aucun cycle mesure de la cellule a predire. L'entrainement doit donc reproduire
exactement cette asymetrie : le CONTEXTE est constitue des autres cellules du
laboratoire, entierement observees, et la REQUETE n'est qu'un couple (T, C)
accompagne des cycles auxquels on veut une valeur. C'est un leave-one-cell-out
tire a chaque exemple.

Un chargeur qui donnerait au reseau ne serait-ce que quelques points de la
cellule cible produirait un modele qui s'effondre a l'inference, ou il n'en a
aucun.

Ce que voit le reseau
---------------------
    contexte : (n_cellules - 1) x n_ctx points, chacun (T, C, cycle, SOH)
    requete  : n_query cycles, avec le (T, C) de la cellule retiree
    cible    : SOH LATENT de la cellule retiree a ces cycles

La cible est le SOH latent - loi L3 plus erreur de modele, sans bruit de mesure
et non tronque - et non les points observes. Deux raisons : le challenge note la
trajectoire complete jusqu'a 70 %, bien au-dela de l'arret de l'essai ; et le
bruit de mesure n'est pas predictible, l'exiger du reseau ne ferait qu'ajouter
une variance irreductible a la perte.

Sous-echantillonnage
--------------------
Le pool stocke 128 points par trajectoire ; le reseau en consomme `n_ctx`
(48 par defaut). C'est deliberé : l'attention est quadratique, et 6 x 128 = 768
tokens couteraient 6.5 fois plus par pas que 6 x 48 = 288 pour une information
quasi identique sur des trajectoires aussi lisses. On stocke fin parce que le
pool est fige et qu'on ne peut pas remonter la resolution apres coup ; on
consomme grossier parce que c'est le temps d'entrainement qui est rare.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA
from .generate import l3_trajectory, erreur_modele

POOL_DIR = DATA / "pfn_pool"

SOH_LO, SOH_HI = 0.5, 119.9

# Normalisation des entrees. Constantes FIGEES : elles doivent etre identiques
# a l'entrainement et a l'inference, donc elles ne se deduisent pas du batch.
T_MOY, T_ECH = 40.0, 15.0          # domaine note [25, 55]
LNC_MOY, LNC_ECH = -0.35, 0.35     # ln C sur [0.5, 1.0]
# ln n : centre MESURE sur les grilles geometriques reellement produites, pas
# devine. Une grille geometrique de 1 a ~5000 place la moitie de ses points sous
# le cycle 70, donc E[ln n] ~ 4.9 et non ln(3000) ~ 8.
LNN_MOY, LNN_ECH = 4.9, 2.1        # ln n sur les grilles de requete/contexte
SOH_MOY, SOH_ECH = 85.0, 15.0      # points de SOH


def normalise(T, C, n, soh=None):
    """(T, C, cycle, SOH) -> variables centrees reduites."""
    z = [(np.asarray(T, np.float32) - T_MOY) / T_ECH,
         (np.log(np.asarray(C, np.float32)) - LNC_MOY) / LNC_ECH,
         (np.log(np.maximum(np.asarray(n, np.float32), 1.0)) - LNN_MOY) / LNN_ECH]
    if soh is not None:
        z.append((np.asarray(soh, np.float32) - SOH_MOY) / SOH_ECH)
    return np.stack(z, axis=-1)


def denormalise_soh(z):
    return z * SOH_ECH + SOH_MOY


# --------------------------------------------------------------- lecture ---
class PoolShard:
    """Un shard charge en memoire, indexe par laboratoire."""

    def __init__(self, f_labs, f_traj):
        labs = pd.read_parquet(f_labs)
        traj = pd.read_parquet(f_traj)
        self.lab_ids = labs.lab_id.unique()
        # theta par (lab, cellule), dans l'ordre des cellules
        self.labs = {k: g.sort_values("cell_id").reset_index(drop=True)
                     for k, g in labs.groupby("lab_id")}
        self.traj = {k: g for k, g in traj.groupby(["lab_id", "cell_id"])}

    def __len__(self):
        return len(self.lab_ids)


def liste_shards(split, pool_dir=None):
    d = Path(pool_dir or POOL_DIR) / split
    labs = sorted(glob.glob(str(d / "labs_*.parquet")))
    return [(f, f.replace("labs_", "traj_")) for f in labs]


def lire_manifest(pool_dir=None):
    p = Path(pool_dir or POOL_DIR) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ------------------------------------------------------- cible latente -----
def soh_latent(theta_ligne, cycles):
    """SOH vrai de la cellule (L3 + erreur de modele), aux cycles demandes.

    Reconstruit depuis les theta stockes, sans avoir a garder la trajectoire
    latente dans le pool. L'erreur de modele est reconstruite avec les memes
    coefficients puisqu'ils derivent des theta - a une exception pres : le pool
    ne stocke PAS `c_modele` (3 coefficients par cellule), qui serait de la
    place perdue. On la reconstruit donc a partir de l'ecart entre la
    trajectoire observee et la L3 pure, ce qui revient au meme a l'echelle qui
    nous interesse. Voir `cible_depuis_observation`.
    """
    r = theta_ligne
    y = l3_trajectory(cycles, r["s0"], r["b"], r["soh_k"], r["g"],
                      r["tau_frac"], r["p"])
    return np.clip(y, SOH_LO, SOH_HI)


def cible_depuis_observation(n_obs, y_obs, theta_ligne, cycles):
    """Cible latente : L3 pure + ecart LISSE lu sur la trajectoire observee.

    Le pool ne stocke pas les coefficients de l'erreur de modele. On les
    retrouve en projetant l'ecart observe (y_obs - L3) sur la meme base de
    modes lisses que celle du generateur, puis on evalue cette projection aux
    cycles demandes. Le bruit de mesure, lui, ne se projette quasiment pas sur
    ces trois modes tres basse frequence : il est donc naturellement filtre,
    ce qui est exactement ce qu'on veut pour une cible.
    """
    n_fin = float(n_obs[-1])
    l3_obs = l3_trajectory(n_obs, theta_ligne["s0"], theta_ligne["b"],
                           theta_ligne["soh_k"], theta_ligne["g"],
                           theta_ligne["tau_frac"], theta_ligne["p"])
    ecart = np.asarray(y_obs, float) - l3_obs

    K = 3
    u = np.minimum(np.asarray(n_obs, float) / max(n_fin, 1.0), 1.0)
    k = np.arange(1, K + 1, dtype=float)
    B = np.sin((k[None, :] - 0.5) * np.pi * u[:, None]) / k[None, :]
    coef, *_ = np.linalg.lstsq(B, ecart, rcond=None)

    uq = np.minimum(np.asarray(cycles, float) / max(n_fin, 1.0), 1.0)
    Bq = np.sin((k[None, :] - 0.5) * np.pi * uq[:, None]) / k[None, :]
    l3_q = l3_trajectory(cycles, theta_ligne["s0"], theta_ligne["b"],
                         theta_ligne["soh_k"], theta_ligne["g"],
                         theta_ligne["tau_frac"], theta_ligne["p"])
    return np.clip(l3_q + Bq @ coef, SOH_LO, SOH_HI)


# ------------------------------------------------------------- batches -----
def _sous_ech(n, y, m, rng):
    """m points parmi les len(n) disponibles, premier et dernier conserves."""
    if len(n) <= m:
        idx = np.arange(len(n))
        idx = np.pad(idx, (0, m - len(n)), mode="edge")
        return n[idx], y[idx]
    idx = np.unique(np.round(np.linspace(0, len(n) - 1, m)).astype(int))
    if len(idx) < m:
        sup = rng.choice(len(n), m - len(idx), replace=False)
        idx = np.sort(np.unique(np.concatenate([idx, sup])))[:m]
    return n[idx], y[idx]


def cycles_requete(n_query, rng, n_max=12000, jitter=0.06):
    """Grille de cycles ou l'on interroge le reseau : 1 a `n_max`, LOG.

    CORRECTIF DU DEFAUT DU RUN v1. La version precedente bornait cette grille au
    franchissement de 70 % de SOH. Consequence mesuree : le reseau n'etait
    jamais interroge au-dela, le dernier point valait toujours ~70 %, et a
    l'evaluation - qui interroge jusqu'a 12 000 cycles - il maintenait une
    valeur proche de 70 % hors de sa plage d'entrainement. Le franchissement
    etait place 2.27 fois trop tard, dans 100 % des cas.

    L'interface de soumission exige `predict_soh` sur 1..12 000 : l'entrainement
    doit couvrir la MEME plage de cycles que l'inference. C'est non negociable.

    La grille est LOGARITHMIQUE : la courbure qui porte l'information sur p et
    sur le knee est concentree dans les premiers milliers de cycles ; une grille
    uniforme y mettrait une poignee de points et gaspillerait le reste sur la
    queue plate.

    Le jitter multiplicatif evite que le reseau memorise 48 valeurs de ln n
    exactes plutot que d'apprendre une fonction continue du cycle.
    """
    g = np.geomspace(1.0, float(n_max), n_query)
    g = g * np.exp(rng.normal(0.0, jitter, n_query))
    return np.clip(np.sort(g), 1.0, float(n_max))


# Ponderation de la perte. Le challenge ne note que SOH >= 70 % ; couvrir
# 1..12 000 est necessaire pour placer correctement le franchissement, mais
# depenser autant de capacite sur la queue sous 70 % - plate et non notee -
# serait du gaspillage. Poids 1.0 au-dessus du seuil, POIDS_SOUS_SEUIL en
# dessous. Sensibilite verifiee sur canari (voir rapport) : 0.1 / 0.2 / 0.3.
POIDS_SOUS_SEUIL = 0.2
SEUIL_NOTE = 70.0


def poids_requete(soh_cible, sous=POIDS_SOUS_SEUIL, seuil=SEUIL_NOTE):
    return np.where(np.asarray(soh_cible) >= seuil, 1.0, sous).astype(np.float32)


def fabrique_exemple(shard, lab_id, rng, n_ctx=48, n_query=48):
    """Un exemple : contexte = 5 cellules, requete = la 6e."""
    labs = shard.labs[lab_id]
    k = len(labs)
    cible = int(rng.integers(k))

    ctx = []
    for ci in range(k):
        if ci == cible:
            continue
        r = labs.iloc[ci]
        t = shard.traj[(lab_id, ci)]
        n, y = _sous_ech(t.cycle.to_numpy(np.float64),
                         t.soh.to_numpy(np.float64), n_ctx, rng)
        ctx.append(normalise(np.full(len(n), r["T"]), np.full(len(n), r["C"]),
                             n, y))
    X_ctx = np.concatenate(ctx, axis=0).astype(np.float32)

    rc = labs.iloc[cible]
    tq = shard.traj[(lab_id, cible)]
    nq = cycles_requete(n_query, rng)
    # Cible calculee sur TOUTE la plage depuis la loi latente, sans troncature.
    y_cible = cible_depuis_observation(
        tq.cycle.to_numpy(np.float64), tq.soh.to_numpy(np.float64), rc, nq)

    X_q = normalise(np.full(len(nq), rc["T"]), np.full(len(nq), rc["C"]),
                    nq).astype(np.float32)
    y_q = ((y_cible - SOH_MOY) / SOH_ECH).astype(np.float32)
    w_q = poids_requete(y_cible)
    return X_ctx, X_q, y_q, w_q


class GenerateurBatches:
    """Itere sur le pool en produisant des batches prets pour le reseau.

    Un shard est charge a la fois (196 Mo), et on en tire `reutilisation`
    batches avant de passer au suivant : recharger un parquet a chaque batch
    dominerait le temps de calcul.
    """

    def __init__(self, split, batch=64, n_ctx=48, n_query=48, seed=0,
                 pool_dir=None, reutilisation=200, boucle=True):
        self.shards = liste_shards(split, pool_dir)
        if not self.shards:
            raise FileNotFoundError(
                f"aucun shard pour le split '{split}' dans "
                f"{Path(pool_dir or POOL_DIR)}")
        self.batch, self.n_ctx, self.n_query = batch, n_ctx, n_query
        self.reutilisation, self.boucle = reutilisation, boucle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        ordre = list(range(len(self.shards)))
        while True:
            self.rng.shuffle(ordre)
            for si in ordre:
                sh = PoolShard(*self.shards[si])
                for _ in range(self.reutilisation):
                    ids = self.rng.choice(sh.lab_ids, self.batch, replace=False)
                    Xc, Xq, Yq, Wq = [], [], [], []
                    for lid in ids:
                        a, b, c, w = fabrique_exemple(sh, lid, self.rng,
                                                      self.n_ctx, self.n_query)
                        Xc.append(a); Xq.append(b); Yq.append(c); Wq.append(w)
                    yield (np.stack(Xc), np.stack(Xq), np.stack(Yq),
                           np.stack(Wq))
            if not self.boucle:
                return
