# -*- coding: utf-8 -*-
"""Generateur de laboratoires synthetiques pour le BatteryPFN.

`prior.draw_lab` tire les theta ; ce module les transforme en ce qu'un banc
d'essai aurait REELLEMENT enregistre : trajectoire L3 propre, puis derive lente
de capteur, puis bruit de mesure, puis arret de l'essai (par profondeur de SOH
ou par budget de cycles), puis trous d'enregistrement.

L'ordre compte et n'est pas arbitraire :

  1. trajectoire propre            theta -> SOH(n), n = 1..budget
  2. derive lente                  erreur de capteur, correlee dans le temps
  3. bruit de mesure               i.i.d., c'est ce qu'un filtre pourrait oter
  4. arret de l'essai              la decision d'arret se prend sur la valeur
                                   MESUREE, pas sur la valeur vraie
  5. trous d'enregistrement        blocs de cycles perdus (fichiers, maintenance)

Mettre l'arret apres le bruit est un detail qui a des consequences : c'est ce
qui fait qu'un essai arrete "a 70 %" s'arrete en realite quelque part autour de
70 %, et que le dernier point d'une trajectoire est legerement biaise vers le
bas. Un generateur qui tronque sur la valeur vraie apprendrait au reseau une
regularite qui n'existe pas.

Ce que le module renvoie, pour chaque cellule :
    n        cycles observes (croissants, avec trous)
    y        SOH observe, en points (convention SOH_nom du scoring)
    y_vrai   SOH sans bruit ni derive, aux memes cycles - pour l'etape C
    theta    parametres qui l'ont engendree - la verite terrain du PFN
"""
from __future__ import annotations

import numpy as np

from .prior import DEFAULT, FADE_P, draw_lab


# --------------------------------------------------------------- loi L3 ----
def nk_from_sohk(s0, b, soh_k, p=FADE_P):
    """Cycle ou la loi de base franchit SOH_k."""
    return float(np.power(max((s0 - soh_k) / max(b, 1e-12), 1e-9), 1.0 / p))


def l3_trajectory(n, s0, b, soh_k, g, tau_frac, p=FADE_P):
    """Trajectoire L3 propre, sans bruit ni borne.

    `p` par defaut a 0.8 (valeur du baseline officiel) pour que les appels
    existants restent valides ; le prior le tire desormais dans [0.40, 1.00].
    """
    n = np.asarray(n, float)
    soh = s0 - b * np.power(n, p)
    if g > 0.0:
        nk = nk_from_sohk(s0, b, soh_k, p)
        tau = max(tau_frac * nk, 1.0)
        z = np.clip((n - nk) / tau, -50.0, 50.0)
        soh = soh - np.where(n > nk, g * tau * np.expm1(z), 0.0)
    return soh


def erreur_modele(n, coeffs, sigma, n_fin, alpha=1.0):
    """Ce que L3 ne sait pas representer : un ecart LISSE et autocorrele.

    e(u) = sigma * sum_k (c_k / k) * sin((k - 1/2) * pi * u) / norme,   u = n/n_fin

    Nul en u = 0 - le debut de vie sert a mesurer s0, l'erreur ne doit pas s'y
    loger - et libre en fin de trajectoire. Le spectre en 1/k concentre
    l'energie dans le mode 1, ce qui donne une longueur de correlation de
    l'ordre de la duree de vie : c'est ce que mesure le residu reel
    (autocorrelation 0.78 a 0.99 encore au lag 50).

    La normalisation rend `sigma` egal a l'ecart-type de e sur u dans [0, 1],
    quel que soit le nombre de modes.
    """
    c = np.asarray(coeffs, float)
    if not c.size or sigma <= 0.0:
        return np.zeros(len(np.atleast_1d(n)), float)
    # u est BORNE a 1 : au-dela de la fin de l'essai l'ecart est PROLONGE a sa
    # derniere valeur plutot que de continuer a osciller. Une mauvaise
    # specification de la loi ne disparait pas et ne se met pas a battre :
    # elle persiste. Sans cette borne, l'extrapolation jusqu'au cycle 12000
    # engendrerait des oscillations sans support physique.
    u = np.minimum(np.asarray(n, float) / max(float(n_fin), 1.0), 1.0)
    k = np.arange(1, len(c) + 1, dtype=float)
    poids = c / k ** alpha
    base = np.sin((k[None, :] - 0.5) * np.pi * u[:, None])
    # var de sin((k-1/2)pi u) sur u uniforme dans [0,1] vaut 1/2
    norme = np.sqrt(0.5 * np.sum((1.0 / k ** alpha) ** 2))
    return sigma * (base @ poids) / norme


# ----------------------------------------------------------- une cellule ---
def cycle_at_soh(s0, b, soh_k, g, tau_frac, seuil, n_max, p=FADE_P):
    """Cycle ou la trajectoire PROPRE franchit `seuil`, ou `n_max` si jamais.

    Recherche sur une grille geometrique puis interpolation lineaire : la
    trajectoire est monotone decroissante, 400 points suffisent a la resoudre
    a mieux que 1 % en cycle sur tout l'horizon.
    """
    grille = np.unique(np.round(np.geomspace(1.0, n_max, 400)))
    y = l3_trajectory(grille, s0, b, soh_k, g, tau_frac, p)
    dessous = np.flatnonzero(y <= seuil)
    if not dessous.size:
        return float(n_max)
    j = int(dessous[0])
    if j == 0:
        return 1.0
    return float(np.interp(seuil, [y[j], y[j - 1]], [grille[j], grille[j - 1]]))


def simulate_cell(cell, rng, cfg=DEFAULT):
    """Une trajectoire observee, des theta a ce que le banc enregistre."""
    p = float(cell.get("p", FADE_P))
    n_vise = cycle_at_soh(cell["s0"], cell["b"], cell["soh_k"], cell["g"],
                          cell["tau_frac"], cell["stop_soh"], cfg.n_max_absolu,
                          p)
    budget = int(np.clip(round(n_vise * cell["marge_budget"]), 30,
                         cfg.n_max_absolu))
    n = np.arange(1, budget + 1, dtype=float)

    # 1. trajectoire L3 pure
    y_l3 = l3_trajectory(n, cell["s0"], cell["b"], cell["soh_k"],
                         cell["g"], cell["tau_frac"], p)

    # 1bis. ERREUR DE MODELE : ce que L3 ne represente pas. Elle fait partie de
    # la VERITE de la cellule, pas de son observation - une cellule reelle ne
    # suit pas L3, et c'est le fait central que le prior doit reproduire.
    # `y_vrai` est donc la trajectoire AVEC l'erreur de modele : c'est elle que
    # l'etape C compare a la prediction, et c'est elle qui decide de l'arret.
    y_vrai = y_l3 + erreur_modele(n, cell.get("c_modele", ()),
                                  cell.get("sigma_modele", 0.0), n[-1],
                                  cell.get("alpha_modele", 1.0))

    # 2. derive lente de capteur : erreur d'INSTRUMENT, monotone et faible
    y_obs = y_vrai.copy()
    if cell["derive"]:
        u = (n - n[0]) / max(n[-1] - n[0], 1.0)
        y_obs = y_obs + cell["derive_amp"] * np.power(u, cell["derive_exp"])

    # 3. bruit de mesure i.i.d.
    y_obs = y_obs + rng.normal(0.0, cell["sigma_meas"], budget)

    # 4. arret de l'essai, decide sur la valeur MESUREE
    env = np.minimum.accumulate(y_obs)
    dessous = np.flatnonzero(env <= cell["stop_soh"])
    stop = int(dessous[0]) + 1 if dessous.size else budget
    stop = max(stop, 30)
    n, y_obs, y_vrai, y_l3 = n[:stop], y_obs[:stop], y_vrai[:stop], y_l3[:stop]

    # 5. trous d'enregistrement : blocs CONTIGUS perdus (fichiers manquants,
    #    arrets de maintenance), jamais sur les tout premiers cycles - c'est
    #    la structure des trous reels, pas un masque aleatoire point par point.
    garde = np.ones(len(n), bool)
    manquant = len(n) - int(round(cell["couverture"] * len(n)))
    n_trous = int(cell["n_trous"])
    if n_trous > 0 and manquant > 0:
        # repartition aleatoire du deficit total entre les blocs
        coupes = np.sort(rng.uniform(0.0, 1.0, n_trous - 1))
        parts = np.diff(np.concatenate([[0.0], coupes, [1.0]]))
        for taille in np.maximum(1, np.round(parts * manquant).astype(int)):
            haut = len(n) - taille
            if haut <= 10:
                continue
            debut = int(rng.integers(10, haut))
            garde[debut:debut + taille] = False
    garde[0] = True
    if garde.sum() < 20:
        garde[:] = True

    return dict(n=n[garde], y=y_obs[garde], y_vrai=y_vrai[garde],
                y_l3=y_l3[garde],
                n_max=float(n[-1]), stop_soh_atteint=float(y_obs[-1]),
                theta=dict(s0=cell["s0"], b=cell["b"], p=p,
                           soh_k=cell["soh_k"], g=cell["g"],
                           tau_frac=cell["tau_frac"], T=cell["T"], C=cell["C"],
                           sigma_modele=cell.get("sigma_modele", 0.0),
                           alpha_modele=cell.get("alpha_modele", 1.0),
                           c_modele=tuple(cell.get("c_modele", ()))))


def simulate_lab(lab, rng, cfg=DEFAULT):
    """Les 6 trajectoires d'un laboratoire tire."""
    out = dict(lab)
    sigma = lab["theta"]["sigma_meas"]
    cells = []
    for c in lab["cellules"]:
        c = dict(c, sigma_meas=sigma)
        cells.append(simulate_cell(c, rng, cfg))
    out["trajectoires"] = cells
    return out


# ------------------------------------------------------------ generation ---
def generate(n_labs, seed=20260101, cfg=DEFAULT, progress=False):
    """`n_labs` laboratoires complets. Un seul RNG, donc reproductible."""
    rng = np.random.default_rng(seed)
    labs = []
    for i in range(int(n_labs)):
        labs.append(simulate_lab(draw_lab(rng, cfg), rng, cfg))
        if progress and (i + 1) % max(1, n_labs // 10) == 0:
            print(f"  {i + 1}/{n_labs} laboratoires")
    return labs


def to_frame(labs):
    """Aplatit une liste de laboratoires en un DataFrame long (diagnostic)."""
    import pandas as pd
    rows = []
    for li, lab in enumerate(labs):
        for ci, tr in enumerate(lab["trajectoires"]):
            th = {k: v for k, v in tr["theta"].items() if k != "c_modele"}
            rows.append(dict(lab=li, cell=ci, structure=lab["structure_nom"],
                             ea_25_35=lab["ea_25_35"], ea_35_45=lab["ea_35_45"],
                             ea_45_55=lab["ea_45_55"],
                             n_points=len(tr["n"]), n_max=tr["n_max"],
                             soh_final=float(tr["y"][-1]),
                             loss_1000=float(th["b"] * 1000.0 ** th["p"]),
                             **th))
    return pd.DataFrame(rows)


def soh_at(traj_n, traj_y, cycles):
    """SOH interpole aux cycles demandes ; NaN au-dela du dernier observe."""
    cycles = np.asarray(cycles, float)
    out = np.interp(cycles, traj_n, traj_y, left=np.nan, right=np.nan)
    return out


if __name__ == "__main__":
    labs = generate(200, progress=True)
    df = to_frame(labs)
    import pandas as pd
    pd.set_option("display.width", 200)
    print(df.groupby("structure")[["ea_25_35", "ea_35_45", "ea_45_55"]]
          .median().round(2).to_string())
    print()
    print(df[["n_points", "n_max", "soh_final", "b", "g", "tau_frac"]]
          .describe(percentiles=[0.05, 0.5, 0.95]).round(4).to_string())
    print()
    print("part des trajectoires arretees au-dessus de 70 % :",
          round(float((df.soh_final > 70).mean()), 3))
