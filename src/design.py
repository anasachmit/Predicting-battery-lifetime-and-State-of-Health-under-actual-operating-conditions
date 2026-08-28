"""Qualification du design experimental des 6 cellules cible.

Le challenge est note sur 25-55 degC x 0.5-1.0 C. On veut savoir, AVANT de
construire une validation leave-one-cell-out, si chaque fold LOCO ressemble
au test reel (interpolation) ou s'il est plus dur que le test (extrapolation,
notamment retrait d'un coin du domaine).

Geometrie retenue : (1000/T_K, ln C). C'est l'espace ou la loi d'Arrhenius +
loi puissance en C-rate est *lineaire*, donc l'espace ou "interpoler" a un
sens pour le modele qu'on va ajuster.
"""
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay

from .config import FIGURES

# Domaine de notation officiel.
T_DOMAIN = (25.0, 55.0)
C_DOMAIN = (0.5, 1.0)


def feature_space(T_degC, c_rate):
    """(T, C) -> coordonnees ou le modele Arrhenius x puissance est lineaire."""
    return np.column_stack([1000.0 / (np.asarray(T_degC, float) + 273.15),
                            np.log(np.asarray(c_rate, float))])


def domain_corners():
    T = np.array([T_DOMAIN[0], T_DOMAIN[0], T_DOMAIN[1], T_DOMAIN[1]])
    C = np.array([C_DOMAIN[0], C_DOMAIN[1], C_DOMAIN[0], C_DOMAIN[1]])
    return T, C


def _inside(hull_pts, pt):
    """pt est-il dans l'enveloppe convexe de hull_pts ?"""
    if len(hull_pts) < 3:
        return False
    try:
        return bool(Delaunay(hull_pts).find_simplex(pt[None, :])[0] >= 0)
    except Exception:
        return False


def _dist_to_hull(hull_pts, pt):
    """Distance signee (positive = dehors) du point a l'enveloppe convexe."""
    if len(hull_pts) < 3:
        return np.nan
    try:
        h = ConvexHull(hull_pts)
    except Exception:
        return np.nan
    # eq: [normale | offset], normale sortante, <= 0 a l'interieur
    d = h.equations[:, :-1] @ pt + h.equations[:, -1]
    return float(d.max())


def loco_qualification(summ_target):
    """Une ligne par fold LOCO : nature du fold et identifiabilite restante."""
    d = summ_target.sort_values(["T_amb", "C_rate_dchg"]).reset_index(drop=True)
    T = d["T_amb"].to_numpy(float)
    C = d["C_rate_dchg"].to_numpy(float)
    X = feature_space(T, C)
    Tc, Cc = domain_corners()
    corners = feature_space(Tc, Cc)

    rows = []
    for i in range(len(d)):
        keep = np.ones(len(d), bool)
        keep[i] = False
        Xk, pt = X[keep], X[i]

        inside = _inside(Xk, pt)
        dist = _dist_to_hull(Xk, pt)

        # Identifiabilite du modele Arrhenius x puissance sur le fold restant.
        A = np.column_stack([np.ones(keep.sum()), Xk[:, 0], Xk[:, 1]])
        cond = float(np.linalg.cond(A))

        # Niveaux restants : peut-on encore separer T de C ?
        n_T = len(np.unique(T[keep]))
        n_C = len(np.unique(C[keep]))
        # combien de cellules restent sur la meme ligne de C-rate ?
        same_C = int((C[keep] == C[i]).sum())

        # Le point retire est-il un coin du domaine de notation ?
        is_corner = bool(np.min(np.linalg.norm(corners - pt, axis=1)) < 1e-9)
        # Le domaine restant couvre-t-il encore les 4 coins ?
        corners_lost = int(sum(not _inside(Xk, c) for c in corners))

        rows.append(dict(
            fold=d.loc[i, "cell_id"], T_amb=T[i], C_rate=C[i],
            nature="interpolation" if inside else "extrapolation",
            dist_hors_hull=round(dist, 4),
            coin_du_domaine=is_corner,
            coins_domaine_non_couverts=corners_lost,
            n_T_restants=n_T, n_C_restants=n_C,
            cellules_meme_C_rate=same_C,
            cond_design=round(cond, 1),
            SOH_min=round(float(d.loc[i, "SOH_min"]), 3),
            n_at_70_observe=(d.loc[i, "n_at_70"] > 0),
        ))
    return pd.DataFrame(rows)


def full_design_note(summ_target):
    """Ce que le design complet (6 cellules) couvre deja ou non."""
    T = summ_target["T_amb"].to_numpy(float)
    C = summ_target["C_rate_dchg"].to_numpy(float)
    X = feature_space(T, C)
    Tc, Cc = domain_corners()
    notes = []
    for tc, cc in zip(Tc, Cc):
        pt = feature_space([tc], [cc])[0]
        obs = bool(((T == tc) & (C == cc)).any())
        notes.append(dict(coin=f"{tc:.0f}degC / {cc}C", cellule_observee=obs,
                          dans_enveloppe=_inside(X, pt),
                          dist_hors_hull=round(_dist_to_hull(X, pt), 4)))
    return pd.DataFrame(notes)


def plot_design(summ_target, loco, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = path or (FIGURES / "step0_design_TC.png")
    T = summ_target["T_amb"].to_numpy(float)
    C = summ_target["C_rate_dchg"].to_numpy(float)
    reach = summ_target["n_at_70"].to_numpy(float) > 0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    ax = axes[0]
    ax.add_patch(plt.Rectangle((T_DOMAIN[0], C_DOMAIN[0]),
                               T_DOMAIN[1] - T_DOMAIN[0],
                               C_DOMAIN[1] - C_DOMAIN[0],
                               fc="0.93", ec="0.6", ls="--", label="domaine note"))
    ax.scatter(T[reach], C[reach], s=170, marker="o", c="#1f77b4",
               ec="k", zorder=3, label="cellule, 70 % observe")
    ax.scatter(T[~reach], C[~reach], s=170, marker="s", c="white",
               ec="crimson", lw=2, zorder=3, label="cellule, 70 % NON observe")
    Tc, Cc = domain_corners()
    for tc, cc in zip(Tc, Cc):
        if not ((T == tc) & (C == cc)).any():
            ax.scatter([tc], [cc], s=260, marker="X", c="crimson", zorder=4)
            ax.annotate("coin sans donnee", (tc, cc), textcoords="offset points",
                        xytext=(-92, 10), color="crimson", fontsize=9)
    for t, c, cid in zip(T, C, summ_target["cell_id"]):
        ax.annotate(cid.replace("102Ah_", "").replace("_cell", " c"),
                    (t, c), textcoords="offset points", xytext=(7, -12), fontsize=8)
    ax.set_xlabel("T ambiante (degC)"); ax.set_ylabel("C-rate")
    ax.set_xlim(20, 60); ax.set_ylim(0.35, 1.15)
    ax.set_title("Design des 6 cellules cible dans le domaine note")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.25)

    ax = axes[1]
    X = feature_space(T, C)
    Xc = feature_space(Tc, Cc)
    ax.scatter(Xc[:, 0], Xc[:, 1], marker="X", s=140, c="crimson",
               label="coins du domaine", zorder=4)
    for i, (nat, xi) in enumerate(zip(loco["nature"], X)):
        col = "#1f77b4" if nat == "interpolation" else "crimson"
        ax.scatter(*xi, s=150, c=col, ec="k", zorder=3)
        ax.annotate(f"{T[i]:.0f}C/{C[i]}", xi, textcoords="offset points",
                    xytext=(7, 5), fontsize=8)
    try:
        h = ConvexHull(X)
        for s in h.simplices:
            ax.plot(X[s, 0], X[s, 1], "0.5", lw=1)
    except Exception:
        pass
    ax.set_xlabel("1000 / T (K$^{-1}$)"); ax.set_ylabel("ln C-rate")
    ax.set_title("Espace ou Arrhenius x puissance est lineaire\n"
                 "(bleu = fold LOCO interpolant, rouge = extrapolant)",
                 fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
