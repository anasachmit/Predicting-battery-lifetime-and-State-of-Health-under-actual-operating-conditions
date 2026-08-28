# -*- coding: utf-8 -*-
"""Audit de plausibilite physique sur toute la grille 25-55 degC x 0.5-1.0 C.

Le LOCO ne teste pas le coin 55 degC / 0.5C : aucune cellule ne s'y trouve et il
tombe hors de l'enveloppe convexe du design. Un modele peut donc passer le LOCO
et produire une trajectoire aberrante la ou le scoring interrogera. Cet audit
est ELIMINATOIRE.
"""
import numpy as np
import pandas as pd

from .config import FIGURES

T_GRID = np.arange(25.0, 55.001, 2.5)
C_GRID = np.arange(0.5, 1.0001, 0.05)


def _n_at(traj, thr=70.0):
    below = np.where(traj < thr)[0]
    if not len(below) or below[0] == 0:
        return np.nan
    j = below[0]
    return float(np.interp(thr, [traj[j], traj[j - 1]], [j + 1, j]))


def sweep(predict_fn, t_grid=T_GRID, c_grid=C_GRID, n_knee_min=30):
    """predict_fn(T, C) -> trajectoire SOH% pour les cycles 1..N."""
    rows = []
    for T in t_grid:
        for C in c_grid:
            traj = np.asarray(predict_fn(T, C), float)
            d = np.diff(traj)
            # detection du knee : maximum de la derivee seconde (acceleration)
            dd = np.diff(d)
            n_knee = float(np.argmin(dd) + 2) if len(dd) else np.nan
            rows.append(dict(
                # arrondi : np.arange(0.5, 1.0, 0.05) produit 0.7500000000000001,
                # ce qui rend l'indexation par valeur impossible ensuite.
                T_amb=round(float(T), 3), C_rate=round(float(C), 3),
                n_at_70=_n_at(traj, 70.0), n_at_80=_n_at(traj, 80.0),
                soh_1=float(traj[0]), soh_min=float(traj.min()),
                finite=bool(np.all(np.isfinite(traj))),
                dans_bornes=bool(np.all((traj > 0) & (traj <= 120))),
                sup_103=bool(np.any(traj > 103.0)),
                taux_violation_monotonie=float(np.mean(d > 1e-9)),
                n_knee=n_knee))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Enveloppe BILATERALE sur le rapport de duree de vie entre C-rates.
#
# Derivation, entierement a partir des 6 cellules cible (results/
# step1_ratio_crate_mesure.csv) :
#
#  1. Les deux seules temperatures ou 0.5C et 1C coexistent sont 25 et 45 degC.
#     A ces temperatures, le rapport n@seuil(0.5C) / n@seuil(1C) mesure sur les
#     seuils ou les DEUX cellules ont une verite terrain (85 a 95 % de SOH)
#     vaut 0.935 a 1.029, mediane 0.960. Le C-rate est donc sans effet
#     detectable sur la duree de vie PAR CYCLE dans cette plage.
#  2. Ce rapport derive lentement avec la profondeur : a 45 degC il passe de
#     1.017 (seuil 95 %) a 0.943 (seuil 85 %), soit 0.73 % par point de SOH.
#  3. Le rapport a 70 % n'est PAS observable : aucune paire de cellules a T
#     egal n'atteint 70 % des deux cotes. Extrapoler la derive mesuree de 85 %
#     jusqu'a 70 % (15 points de plus) donne environ +/- 11 %, auxquels
#     s'ajoute la dispersion observee de +/- 5 %.
#  4. On retient une bande symetrique en echelle logarithmique autour de 1,
#     de demi-largeur 25 % : [0.80, 1.25]. Elle couvre largement la derive
#     extrapolee, et laisse une marge parce que le rapport a 70 % reste une
#     grandeur non mesuree.
#
# Bilaterale par construction : elle rejette aussi bien un exposant en ln(C)
# libre qui inverse l'ordre (rapport trop grand) qu'un terme calendaire qui
# s'emballe au coin vide (rapport trop petit).
RATIO_C_MIN, RATIO_C_MAX = 0.80, 1.25


def crate_ratio_envelope(sw, c_lo=0.5, c_hi=1.0):
    """Rapport n@70(c_lo) / n@70(c_hi) a chaque temperature de la grille."""
    piv = sw.pivot(index="T_amb", columns="C_rate", values="n_at_70")
    if c_lo not in piv.columns or c_hi not in piv.columns:
        return pd.DataFrame()
    r = piv[c_lo] / piv[c_hi]
    return pd.DataFrame({"T_amb": r.index, "ratio_n70_0p5C_sur_1C": r.values,
                         "dans_enveloppe": (r.values >= RATIO_C_MIN)
                                           & (r.values <= RATIO_C_MAX)})


def audit(sw, n_knee_min=30):
    """Verdict par critere. Tout False est eliminatoire."""
    piv = sw.pivot(index="T_amb", columns="C_rate", values="n_at_70")
    # Monotonie en T : STRICTE. La dependance a la temperature est massive et
    # sans ambiguite de signe dans les donnees.
    dec_T = bool(np.all(np.diff(piv.to_numpy(), axis=0) <= 1e-6))
    env = crate_ratio_envelope(sw)
    ratio_ok = bool(env["dans_enveloppe"].all()) if len(env) else True
    checks = {
        "trajectoires finies": bool(sw["finite"].all()),
        "SOH dans (0, 120]": bool(sw["dans_bornes"].all()),
        "SOH <= 103 %": bool(~sw["sup_103"].any()),
        "SOH monotone decroissante": bool((sw["taux_violation_monotonie"] == 0).all()),
        "n@70 decroit avec T (strict)": dec_T,
        f"rapport n@70 entre C-rates dans [{RATIO_C_MIN}, {RATIO_C_MAX}]": ratio_ok,
        "n@70 defini partout": bool(sw["n_at_70"].notna().all()),
        f"aucun knee avant {n_knee_min} cycles":
            bool((sw["n_knee"].fillna(1e9) >= n_knee_min).all()),
    }
    return checks, piv


def plot_sweep(sw, path=None, titre=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or (FIGURES / "audit_plausibilite.png")
    piv = sw.pivot(index="T_amb", columns="C_rate", values="n_at_70")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto", cmap="viridis",
                   extent=[piv.columns.min(), piv.columns.max(),
                           piv.index.min(), piv.index.max()])
    fig.colorbar(im, ax=ax, label="cycle d'atteinte de 70 %")
    ax.scatter([0.5, 1.0, 1.0, 0.5, 1.0, 1.0], [25, 25, 35, 45, 45, 55],
               marker="o", s=90, fc="none", ec="w", lw=2, label="cellules")
    ax.scatter([0.5], [55], marker="X", s=170, c="crimson", label="coin sans donnee")
    ax.set_xlabel("C-rate"); ax.set_ylabel("T ambiante (degC)")
    ax.set_title(f"n@70 sur la grille notee\n{titre}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for C in [0.5, 0.75, 1.0]:
        s = sw[np.isclose(sw["C_rate"], C)]
        ax.plot(s["T_amb"], s["n_at_70"], "o-", ms=4, label=f"{C}C")
    ax.set_xlabel("T ambiante (degC)"); ax.set_ylabel("n@70")
    ax.set_yscale("log"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title("Monotonie en T (doit decroitre)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
