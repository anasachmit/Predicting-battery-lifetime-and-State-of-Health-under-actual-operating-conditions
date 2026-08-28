# -*- coding: utf-8 -*-
"""Etape B - test de couverture du prior. BLOQUANT.

Question posee : les six trajectoires reelles du challenge sont-elles des
tirages PLAUSIBLES de notre prior ? Un PFN entraine sur un prior qui ne les
engendre jamais n'a aucune raison de savoir les traiter - il aura appris a
inferer dans un monde ou elles n'existent pas.

Protocole
---------
1. On tire `n_labs` laboratoires avec le generateur reel (`generate.generate`),
   donc avec exactement le processus d'observation qui servira a l'entrainement.
2. Pour chaque cellule synthetique on retient la trajectoire LATENTE (sans
   bruit, non tronquee) evaluee sur une grille de cycles fixe. C'est la
   trajectoire latente qui est comparee, et non l'observee : la troncature est
   un processus d'observation, pas une propriete de la cellule, et un essai
   arrete a 2000 cycles n'a rien a dire sur ce qu'aurait donne le cycle 4000.
   Le bruit, lui, est negligeable devant la largeur de l'enveloppe (0.4 point
   au plus, contre plusieurs dizaines de points).
3. Pour chaque cellule REELLE, on ne compare qu'aux cellules synthetiques a
   CONDITIONS COMPARABLES (meme temperature a `tol_T` pres, meme C-rate a
   `tol_C` pres en facteur). Comparer une cellule a 25 degC a l'enveloppe
   globale, toutes temperatures confondues, rendrait le test trivialement vert.
4. On rapporte, a plusieurs cycles, le RANG QUANTILE de la cellule reelle dans
   la distribution synthetique conditionnelle.

Critere
-------
* ECHEC DUR   : rang a 0 % ou 100 % - la cellule reelle sort de l'enveloppe.
* ECHEC       : rang < 5 % ou > 95 % - le prior est mal centre a cet endroit.
* SUCCES      : tous les rangs dans [5 %, 95 %].

Wheeler et SNL sont testes en SECONDAIRE : autre chimie (18650 1.1 Ah), autre
format, autre capacite de reference. Une couverture moins bonne y est
legitime et informative, pas bloquante. Ces deux sources sont comparees sur la
PERTE (s0 - SOH) et non sur le SOH brut, pour neutraliser le decalage de
formation : leurs SOH_nom demarrent a 91-97 %, contre 102.5 % pour la cible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import RESULTS
from .generate import generate, l3_trajectory, erreur_modele
from .prior import DEFAULT, draw_structure
from .real_theta import load_real_cells

FIG_PATH = RESULTS / "pfn" / "coverage_test.png"
RANK_CSV = RESULTS / "pfn" / "coverage_ranks.csv"
CROSS_CSV = RESULTS / "pfn" / "coverage_franchissements.csv"
COND_CSV = RESULTS / "pfn" / "coverage_conditionnelle.csv"

SOH_LO, SOH_HI = 0.5, 119.9      # bornes du modele soumis (framework/io.py)

# Grille de cycles sur laquelle l'enveloppe est calculee. Geometrique : la
# trajectoire bouge vite au debut et lentement ensuite.
GRILLE = np.unique(np.round(np.geomspace(20.0, 12000.0, 160)))

# Cycles auxquels le rang quantile est rapporte.
CYCLES_RAPPORT = (250, 500, 1000, 1500, 2000, 3000, 4000, 5000)

TOL_T = 3.0        # degC
TOL_C = 1.35       # facteur sur le C-rate


# --------------------------------------------------------- echantillonnage --
def envelope_bank(n_labs=1000, seed=20260101, cfg=DEFAULT, progress=True):
    """Trajectoires latentes de toutes les cellules synthetiques, sur GRILLE.

    Renvoie un dict : T, C, S0 de longueur N ; Y et vivant de forme
    (N, len(GRILLE)).
    """
    T, C, S0, Y, vivant = [], [], [], [], []
    labs = generate(n_labs, seed=seed, cfg=cfg, progress=progress)
    for lab in labs:
        for tr in lab["trajectoires"]:
            th = tr["theta"]
            T.append(th["T"])
            C.append(th["C"])
            S0.append(th["s0"])
            # BORNE PHYSIQUE, indispensable : au-dela du knee le terme
            # g*tau*(exp((n-n_k)/tau) - 1) diverge, et sans borne les quantiles
            # bas de l'enveloppe valent -10^7 - ce qui rendrait le test
            # trivialement vert. C'est exactement la borne que le modele soumis
            # applique dans `predict_soh` (framework/io.py l'exige).
            # La trajectoire de reference inclut l'ERREUR DE MODELE : une
            # cellule reelle ne suit pas L3, et c'est justement ce qu'on teste.
            y = l3_trajectory(GRILLE, th["s0"], th["b"], th["soh_k"],
                              th["g"], th["tau_frac"], th["p"])
            y = y + erreur_modele(GRILLE, th.get("c_modele", ()),
                                  th.get("sigma_modele", 0.0), tr["n_max"],
                                  th.get("alpha_modele", 1.0))
            Y.append(np.clip(y, SOH_LO, SOH_HI))
            vivant.append(GRILLE <= tr["n_max"])
    return dict(T=np.asarray(T), C=np.asarray(C), S0=np.asarray(S0),
                Y=np.asarray(Y), vivant=np.asarray(vivant), labs=labs)


def _match(T_bank, C_bank, T, C, tol_T=TOL_T, tol_C=TOL_C):
    """Masque des cellules synthetiques a conditions comparables."""
    m = np.abs(T_bank - T) <= tol_T
    if C is not None and np.isfinite(C):
        m &= np.abs(np.log(C_bank / C)) <= np.log(tol_C)
    return m


def _rank(valeurs, x):
    """Rang quantile de x dans `valeurs`, en pourcent."""
    v = valeurs[np.isfinite(valeurs)]
    if not v.size:
        return np.nan
    return 100.0 * float(np.mean(v < x) + 0.5 * np.mean(v == x))


# -------------------------------------------------------------- test -------
def coverage_table(bank, cells, mode="soh", tol_T=TOL_T, tol_C=TOL_C,
                   cycles=CYCLES_RAPPORT):
    """Rang quantile de chaque cellule reelle, a chaque cycle rapporte.

    `mode="soh"`   : compare le SOH brut (cible, meme convention).
    `mode="perte"` : compare s0 - SOH (Wheeler, SNL - autre convention).
    """
    T_bank, C_bank, Y_bank, S0_bank = bank["T"], bank["C"], bank["Y"], bank["S0"]
    rows = []
    for cl in cells:
        m = _match(T_bank, C_bank, cl["T"], cl["C"], tol_T, tol_C)
        if m.sum() < 30:
            rows.append(dict(cell_id=cl["cell_id"], source=cl["source"],
                             T=cl["T"], C=cl["C"], cycle=np.nan,
                             n_synth=int(m.sum()), rang=np.nan,
                             soh_reel=np.nan, statut="pas assez de synthetiques"))
            continue
        # en mode "perte" chaque cellule synthetique est ramenee a son PROPRE
        # s0, ce qui neutralise le decalage de formation des deux cotes.
        Ym = (S0_bank[m][:, None] - Y_bank[m]) if mode == "perte" else Y_bank[m]
        for c in cycles:
            if c > cl["n"].max():
                continue
            y_reel = float(np.interp(c, cl["n"], cl["y"]))
            x = (cl["s0"] - y_reel) if mode == "perte" else y_reel
            vals = np.asarray([np.interp(c, GRILLE, Ym[i])
                               for i in range(Ym.shape[0])])
            r = _rank(vals, x)
            rows.append(dict(cell_id=cl["cell_id"], source=cl["source"],
                             T=cl["T"], C=cl["C"], cycle=int(c),
                             n_synth=int(m.sum()), rang=round(r, 1),
                             soh_reel=round(y_reel, 2),
                             q05=round(float(np.percentile(vals, 5)), 2),
                             q50=round(float(np.percentile(vals, 50)), 2),
                             q95=round(float(np.percentile(vals, 95)), 2),
                             statut=_statut(r)))
    return pd.DataFrame(rows)


SEUILS_FRANCHISSEMENT = (95.0, 90.0, 85.0, 80.0, 75.0, 70.0)


def _n_at(grille, y, seuil):
    """Cycle ou y franchit `seuil`, +inf si jamais sur la grille."""
    dessous = np.flatnonzero(y <= seuil)
    if not dessous.size:
        return np.inf
    j = int(dessous[0])
    if j == 0:
        return float(grille[0])
    return float(np.interp(seuil, [y[j], y[j - 1]], [grille[j], grille[j - 1]]))


def crossing_table(bank, cells, tol_T=TOL_T, tol_C=TOL_C,
                   seuils=SEUILS_FRANCHISSEMENT):
    """Rang quantile du CYCLE DE FRANCHISSEMENT de chaque seuil de SOH.

    Test bien plus exigeant que le precedent, et bien plus pertinent : c'est
    la grandeur que note le challenge (n@70), et sa dispersion synthetique est
    bornee, alors que celle du SOH a cycle fixe s'ouvre a toute l'echelle des
    que le knee est passe. Un prior peut couvrir toutes les trajectoires en SOH
    et se tromper d'un facteur trois sur le cycle de fin de vie ; c'est ici que
    ca se verrait.

    Les cellules synthetiques qui ne franchissent JAMAIS le seuil dans
    l'horizon comptent comme +inf : elles sont au-dessus de la cellule reelle,
    pas exclues. Les exclure biaiserait le rang vers le haut.
    """
    T_bank, C_bank, Y_bank = bank["T"], bank["C"], bank["Y"]
    rows = []
    for cl in cells:
        m = _match(T_bank, C_bank, cl["T"], cl["C"], tol_T, tol_C)
        if m.sum() < 30:
            continue
        Ym = Y_bank[m]
        env_reel = np.minimum.accumulate(cl["y"])
        for s in seuils:
            if env_reel.min() > s:
                continue                     # seuil non atteint par le reel
            n_reel = _n_at(cl["n"], env_reel, s)
            n_synth = np.asarray([_n_at(GRILLE, Ym[i], s)
                                  for i in range(Ym.shape[0])])
            r = _rank(np.where(np.isinf(n_synth), 1e9, n_synth), n_reel)
            fini = n_synth[np.isfinite(n_synth)]
            rows.append(dict(
                cell_id=cl["cell_id"], source=cl["source"], T=cl["T"], C=cl["C"],
                seuil=s, n_reel=round(n_reel, 1), n_synth=int(m.sum()),
                part_jamais_atteint=round(float(np.mean(np.isinf(n_synth))), 3),
                q05=round(float(np.percentile(fini, 5)), 0) if fini.size else np.nan,
                q50=round(float(np.percentile(fini, 50)), 0) if fini.size else np.nan,
                q95=round(float(np.percentile(fini, 95)), 0) if fini.size else np.nan,
                rang=round(r, 1), statut=_statut(r)))
    return pd.DataFrame(rows)


SEUIL_ANCRAGE = 90.0
TOL_ANCRAGE = 0.25       # +/- 25 % sur le cycle d'ancrage


def conditional_crossing_table(bank, cells, tol_T=TOL_T, tol_C=TOL_C,
                               ancrage=SEUIL_ANCRAGE, tol_ancrage=TOL_ANCRAGE,
                               seuils=(85.0, 80.0, 75.0, 70.0)):
    """Couverture CONDITIONNELLE : la forme du knee, a rythme de base donne.

    Motivation. Les tests B.1 et B.2 sont domines par la dispersion de `b` :
    l'enveloppe y est si large que deplacer SOH_k de 70 a 90 ne les fait pas
    bouger d'un pouce (verifie par falsification). Ils ne disent donc RIEN du
    knee - or le knee est la moitie du modele.

    On conditionne donc sur le rythme de debut de vie : on ne garde que les
    cellules synthetiques qui franchissent `ancrage` (90 %) au meme cycle que
    la cellule reelle, a `tol_ancrage` pres. A rythme de base egal, ce qui
    reste de dispersion vient du knee et de lui seul. On rapporte alors le rang
    des franchissements suivants.

    C'est le test qui peut faire echouer un prior dont le knee est mal place.
    """
    T_bank, C_bank, Y_bank = bank["T"], bank["C"], bank["Y"]
    n_anc_bank = np.asarray([_n_at(GRILLE, Y_bank[i], ancrage)
                             for i in range(Y_bank.shape[0])])
    rows = []
    for cl in cells:
        env_reel = np.minimum.accumulate(cl["y"])
        if env_reel.min() > ancrage:
            continue
        n_anc_reel = _n_at(cl["n"], env_reel, ancrage)
        m = _match(T_bank, C_bank, cl["T"], cl["C"], tol_T, tol_C)
        tol = tol_ancrage
        for _ in range(3):       # elargit la fenetre si trop peu de voisins
            mc = m & np.isfinite(n_anc_bank) & (
                np.abs(n_anc_bank / n_anc_reel - 1.0) <= tol)
            if mc.sum() >= 30:
                break
            tol *= 1.6
        if mc.sum() < 30:
            continue
        Ym = Y_bank[mc]
        for s in seuils:
            if env_reel.min() > s:
                continue
            n_reel = _n_at(cl["n"], env_reel, s)
            n_synth = np.asarray([_n_at(GRILLE, Ym[i], s)
                                  for i in range(Ym.shape[0])])
            r = _rank(np.where(np.isinf(n_synth), 1e9, n_synth), n_reel)
            fini = n_synth[np.isfinite(n_synth)]
            rows.append(dict(
                cell_id=cl["cell_id"], source=cl["source"], T=cl["T"], C=cl["C"],
                seuil=s, n_ancrage=round(n_anc_reel, 0),
                tol_ancrage=round(tol, 2), n_synth=int(mc.sum()),
                n_reel=round(n_reel, 1),
                q05=round(float(np.percentile(fini, 5)), 0) if fini.size else np.nan,
                q50=round(float(np.percentile(fini, 50)), 0) if fini.size else np.nan,
                q95=round(float(np.percentile(fini, 95)), 0) if fini.size else np.nan,
                rang=round(r, 1), statut=_statut(r)))
    return pd.DataFrame(rows)


def _statut(r):
    if not np.isfinite(r):
        return "indefini"
    if r <= 0.0 or r >= 100.0:
        return "HORS ENVELOPPE"
    if r < 5.0 or r > 95.0:
        return "bordure"
    return "ok"


def verdict(table):
    """Resume bloquant / non bloquant du test."""
    t = table.dropna(subset=["rang"])
    par_cellule = (t.groupby("cell_id")["statut"]
                   .apply(lambda s: "HORS ENVELOPPE" if (s == "HORS ENVELOPPE").any()
                          else ("bordure" if (s == "bordure").any() else "ok")))
    return dict(cellules=par_cellule.to_dict(),
                n_hors=int((t.statut == "HORS ENVELOPPE").sum()),
                n_bordure=int((t.statut == "bordure").sum()),
                n_ok=int((t.statut == "ok").sum()),
                vert=bool((t.statut == "ok").all()))


# ------------------------------------------------------------- figure ------
def _b_reels(csv=None):
    """b mesure par temperature sur les 6 cellules cible, normalise a 40 degC.

    Moyenne geometrique par niveau de temperature, puis interpolation log a
    40 degC. C'est la courbe que les trois structures du prior doivent
    encadrer - et c'est elle qui n'est pas une droite d'Arrhenius.
    """
    csv = csv or (RESULTS / "pfn" / "theta_real.csv")
    if not csv.exists():
        return None
    t = pd.read_csv(csv)
    t = t[t.source == "target"]
    if t.empty:
        return None
    grp = t.groupby("T")["b"].apply(lambda s: float(np.exp(np.mean(np.log(s)))))
    Ts = grp.index.to_numpy(float)
    bs = grp.to_numpy(float)
    b40 = float(np.exp(np.interp(40.0, Ts, np.log(bs))))
    return Ts, bs / b40



def plot_coverage(bank, cible, wheeler, snl, tbl_cible, cr_cible, cd_cible,
                  path=FIG_PATH):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T_bank, C_bank, Y_bank, S0_bank = bank["T"], bank["C"], bank["Y"], bank["S0"]
    fig, axes = plt.subplots(4, 3, figsize=(16.5, 18.0))

    # --- 6 cellules cible -------------------------------------------------
    for k, cl in enumerate(cible):
        ax = axes[k // 3, k % 3]
        m = _match(T_bank, C_bank, cl["T"], cl["C"])
        Ym = Y_bank[m]
        q = np.percentile(Ym, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(GRILLE, q[0], q[4], color="#4c72b0", alpha=0.18,
                        label="synthetique, 5-95 %")
        ax.fill_between(GRILLE, q[1], q[3], color="#4c72b0", alpha=0.33,
                        label="synthetique, 25-75 %")
        ax.plot(GRILLE, q[2], color="#4c72b0", lw=1.4, ls="--",
                label="mediane synthetique")
        ax.plot(cl["n"], cl["y"], color="k", lw=1.8, label="cellule reelle")
        ax.axhline(70, color="crimson", lw=0.9, ls=":")
        sub = tbl_cible[tbl_cible.cell_id == cl["cell_id"]].dropna(subset=["rang"])
        pire = sub.loc[(sub.rang - 50).abs().idxmax()] if len(sub) else None
        titre = (f"{cl['cell_id'].replace('102Ah_', '')}\n"
                 f"{int(m.sum())} cellules synthetiques comparables")
        if pire is not None:
            titre += f" - rang extreme {pire.rang:.0f} % (cycle {int(pire.cycle)})"
        ax.set_title(titre, fontsize=9)
        ax.set_xlim(0, max(cl["n"].max() * 1.1, 500))
        ax.set_ylim(35, 106)
        ax.set_xlabel("cycle")
        ax.set_ylabel("SOH (%)")
        ax.grid(alpha=0.25)
        if k == 0:
            ax.legend(fontsize=7, loc="lower left")

    # --- B.1 : rangs quantiles du SOH a cycle fixe ------------------------
    ax = axes[2, 0]
    for cl in cible:
        sub = tbl_cible[(tbl_cible.cell_id == cl["cell_id"])].dropna(subset=["rang"])
        if len(sub):
            ax.plot(sub.cycle, sub.rang, marker="o", ms=4, lw=1.2,
                    label=cl["cell_id"].replace("102Ah_", "").replace("_cell", " c"))
    ax.axhspan(5, 95, color="#55a868", alpha=0.13)
    for v in (5, 95):
        ax.axhline(v, color="#55a868", lw=1.0, ls="--")
    ax.axhline(50, color="0.5", lw=0.8)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("cycle")
    ax.set_ylabel("rang quantile (%)")
    ax.set_title("B.1 - rang du SOH a cycle fixe\ncritere : rester dans la "
                 "bande verte [5 %, 95 %]", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.25)

    # --- B.2 : rangs quantiles du CYCLE DE FRANCHISSEMENT -----------------
    ax = axes[2, 1]
    for cl in cible:
        sub = cr_cible[cr_cible.cell_id == cl["cell_id"]].dropna(subset=["rang"])
        if len(sub):
            ax.plot(sub.seuil, sub.rang, marker="s", ms=4.5, lw=1.2,
                    label=cl["cell_id"].replace("102Ah_", "").replace("_cell", " c"))
    ax.axhspan(5, 95, color="#55a868", alpha=0.13)
    for v in (5, 95):
        ax.axhline(v, color="#55a868", lw=1.0, ls="--")
    ax.axhline(50, color="0.5", lw=0.8)
    ax.invert_xaxis()
    ax.set_ylim(-2, 102)
    ax.set_xlabel("seuil de SOH franchi (%)")
    ax.set_ylabel("rang quantile du cycle de franchissement (%)")
    ax.set_title("B.2 - rang du CYCLE de franchissement\n(test exigeant : c'est "
                 "n@70 que note le challenge)", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.25)

    # --- B.3 : couverture CONDITIONNELLE, la forme du knee ----------------
    ax = axes[2, 2]
    for cl in cible:
        sub = cd_cible[cd_cible.cell_id == cl["cell_id"]].dropna(subset=["rang"])
        if len(sub):
            ax.plot(sub.seuil, sub.rang, marker="^", ms=5, lw=1.2,
                    label=cl["cell_id"].replace("102Ah_", "").replace("_cell", " c"))
    ax.axhspan(5, 95, color="#55a868", alpha=0.13)
    for v in (5, 95):
        ax.axhline(v, color="#55a868", lw=1.0, ls="--")
    ax.axhline(50, color="0.5", lw=0.8)
    ax.invert_xaxis()
    ax.set_ylim(-2, 102)
    ax.set_xlabel("seuil de SOH franchi (%)")
    ax.set_ylabel("rang quantile, a rythme de base egal (%)")
    ax.set_title("B.3 - couverture CONDITIONNELLE (forme du knee)\n"
                 "seules les synthetiques passant 90 % au meme cycle",
                 fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.25)

    # --- Wheeler et SNL, en PERTE ----------------------------------------
    for ax, cells, nom in ((axes[3, 0], wheeler, "Wheeler (18650, 50 degC)"),
                           (axes[3, 1], snl, "SNL (18650 LFP)")):
        if nom.startswith("Wheeler"):
            m = np.abs(T_bank - 50.0) <= TOL_T
        else:
            m = (T_bank >= 25.0 - TOL_T) & (T_bank <= 35.0 + TOL_T)
        P = S0_bank[m][:, None] - Y_bank[m]
        q = np.percentile(P, [5, 50, 95], axis=0)
        ax.fill_between(GRILLE, q[0], q[2], color="#c44e52", alpha=0.18,
                        label="synthetique, 5-95 %")
        ax.plot(GRILLE, q[1], color="#c44e52", lw=1.4, ls="--",
                label="mediane synthetique")
        for cl in cells:
            ax.plot(cl["n"], cl["s0"] - cl["y"], color="k", lw=0.8, alpha=0.55)
        ax.set_xlim(0, 7000)
        ax.set_ylim(0, 70)
        ax.set_xlabel("cycle")
        ax.set_ylabel("perte s0 - SOH (points)")
        ax.set_title(f"SECONDAIRE - {nom}\n{int(m.sum())} synthetiques a T comparable",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)

    # --- le coeur du prior : les trois structures de b(T) -----------------
    ax = axes[3, 2]
    Tg = np.linspace(25.0, 55.0, 120)
    couleurs = dict(arrhenius="#4c72b0", charniere="#c44e52",
                    quadratique="#55a868")
    vus = set()
    rng = np.random.default_rng(7)
    for _ in range(180):
        s = draw_structure(rng)
        lab = s.nom if s.nom not in vus else None
        vus.add(s.nom)
        ax.plot(Tg, np.exp(s.log_ratio(Tg)), color=couleurs[s.nom],
                lw=0.8, alpha=0.30, label=lab)
    reel = _b_reels()
    if reel is not None:
        ax.plot(reel[0], reel[1], "ko--", ms=7, lw=1.6,
                label="b mesure, 6 cellules cible (normalise a 40 degC)")
    ax.set_xlabel("T (degC)")
    ax.set_ylabel("b(T) / b(40 degC)")
    ax.set_yscale("log")
    ax.set_title("A.1 - le melange de structures de b(T)\n"
                 "c'est CE melange qui est le mecanisme du gain", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Etape B - couverture du prior BatteryPFN : les trajectoires "
                 "reelles sont-elles des tirages plausibles ?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------- runner ----
def run(n_labs=1000, seed=20260101, cfg=DEFAULT, verbose=True):
    reels = load_real_cells()
    cible = [c for c in reels if c["source"] == "target"]
    wheeler = [c for c in reels if c["source"] == "wheeler"]
    snl = [c for c in reels if c["source"] == "snl"]

    bank = envelope_bank(n_labs, seed=seed, cfg=cfg, progress=verbose)
    tbl_cible = coverage_table(bank, cible, mode="soh")
    tbl_w = coverage_table(bank, wheeler, mode="perte", tol_C=np.inf)
    tbl_s = coverage_table(bank, snl, mode="perte", tol_C=np.inf)

    cr_cible = crossing_table(bank, cible)
    cr_w = crossing_table(bank, wheeler, tol_C=np.inf)
    cr_s = crossing_table(bank, snl, tol_C=np.inf)

    cd_cible = conditional_crossing_table(bank, cible)

    tbl = pd.concat([tbl_cible, tbl_w, tbl_s], ignore_index=True)
    cross = pd.concat([cr_cible, cr_w, cr_s], ignore_index=True)
    RANK_CSV.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(RANK_CSV, index=False)
    cross.to_csv(CROSS_CSV, index=False)
    cd_cible.to_csv(COND_CSV, index=False)
    fig = plot_coverage(bank, cible, wheeler, snl, tbl_cible, cr_cible, cd_cible)
    return dict(table=tbl, table_cible=tbl_cible, cross=cross,
                cross_cible=cr_cible, cond_cible=cd_cible,
                verdict=verdict(tbl_cible), verdict_cross=verdict(cr_cible),
                verdict_cond=verdict(cd_cible),
                verdict_wheeler=verdict(tbl_w), verdict_snl=verdict(tbl_s),
                verdict_cross_wheeler=verdict(cr_w),
                verdict_cross_snl=verdict(cr_s),
                figure=fig, bank=bank)


def _resume(v):
    return {k: v[k] for k in ("n_hors", "n_bordure", "n_ok", "vert")}


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 240)
    print()
    print("=== B.1  SOH a cycle fixe, 6 cellules cible ===")
    print(out["table_cible"].to_string(index=False))
    print()
    print("=== B.2  cycle de franchissement, 6 cellules cible (test exigeant) ===")
    print(out["cross_cible"].to_string(index=False))
    print()
    print("=== B.3  couverture conditionnelle : la forme du knee ===")
    print(out["cond_cible"].to_string(index=False))
    print()
    print("VERDICT CIBLE - SOH a cycle fixe :", _resume(out["verdict"]))
    print("  par cellule :", out["verdict"]["cellules"])
    print("VERDICT CIBLE - franchissements  :", _resume(out["verdict_cross"]))
    print("  par cellule :", out["verdict_cross"]["cellules"])
    print("VERDICT CIBLE - conditionnelle   :", _resume(out["verdict_cond"]))
    print("  par cellule :", out["verdict_cond"]["cellules"])
    print()
    print("secondaire Wheeler : SOH", _resume(out["verdict_wheeler"]),
          "| franchissements", _resume(out["verdict_cross_wheeler"]))
    print("secondaire SNL     : SOH", _resume(out["verdict_snl"]),
          "| franchissements", _resume(out["verdict_cross_snl"]))
    print()
    print("figure :", out["figure"])
