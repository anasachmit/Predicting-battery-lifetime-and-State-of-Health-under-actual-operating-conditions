# -*- coding: utf-8 -*-
"""Visualisation des trajectoires predites (section 7bis du notebook).

Raison d'etre : RMSE et APE sur n@70 sont des scalaires. Deux modeles de meme
RMSE peuvent echouer tres differemment - derive uniforme contre erreur
concentree sur le knee - et le second est bien plus grave. Aucun chiffre
agrege ne les separe ; il faut voir les courbes.

Toutes les figures partagent les memes echelles entre panneaux comparables,
sans quoi la comparaison visuelle est trompeuse.
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .config import FIGURES  # noqa: E402
from .metrics import cycle_at, knee_onset, ZONE_BOUNDS  # noqa: E402

DPI = 150
SEUIL_EOL = 70.0

COUL = {"verite": "#111111", "ancrage": "#1f77b4", "baseline": "#d62728",
        "vu": "#1f77b4", "predit": "#1f77b4"}
ZONE_COUL = {"quasi_lineaire": "#f2f6fa", "pre_knee": "#fff4e0",
             "post_knee": "#fde9e9"}


def _short(cell_id):
    return cell_id.replace("102Ah_", "").replace("degC", " degC").replace("_cell", " c")


def _fig_path(name):
    return FIGURES / name


# ==========================================================================
#  Primitive reutilisable
# ==========================================================================
def plot_trajectory(cell_id, models, truncation=None, ax=None, truth=None,
                    seuil=SEUIL_EOL, show_knee=True, ylim=None, xlim=None,
                    annotate=True, legend=False):
    """Trace la trajectoire vraie d'une cellule et une ou plusieurs predictions.

    cell_id    : identifiant, sert au titre
    models     : dict {label: trajectoire SOH% indexee sur les cycles 1..N}
                 ou {label: (trajectoire, dict de style)}
    truth      : (cycles, soh_pct) ; obligatoire
    truncation : niveau de SOH au-dela duquel la prediction est extrapolee.
                 Si fourni, la partie ajustee est en trait plein et la partie
                 extrapolee en pointille.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    n_true, y_true = truth
    n_true = np.asarray(n_true, float)
    y_true = np.asarray(y_true, float)

    # zones de SOH en fond : elles rendent lisible OU l'erreur se produit
    if show_knee:
        for name, (lo, hi) in ZONE_BOUNDS.items():
            ax.axhspan(max(lo, 0), min(hi, 130), color=ZONE_COUL[name],
                       zorder=0, lw=0)

    ax.plot(n_true, y_true, lw=1.0, color=COUL["verite"], alpha=0.85,
            label="verite terrain", zorder=3)

    n_cut = np.nan
    if truncation is not None:
        env = np.minimum.accumulate(y_true)
        below = np.where(env < truncation)[0]
        n_cut = float(n_true[below[0]]) if len(below) else np.nan

    for label, item in models.items():
        traj, style = (item if isinstance(item, tuple) else (item, {}))
        traj = np.asarray(traj, float)
        n = np.arange(1, len(traj) + 1, dtype=float)
        col = style.get("color", COUL.get(label, "#666666"))
        if truncation is not None and np.isfinite(n_cut):
            m_vu = n <= n_cut
            ax.plot(n[m_vu], traj[m_vu], lw=2.0, color=col, zorder=4,
                    label=f"{label} (ajuste)")
            ax.plot(n[~m_vu], traj[~m_vu], lw=1.6, ls="--", color=col,
                    zorder=4, label=f"{label} (extrapole)")
        else:
            ax.plot(n, traj, lw=1.7, color=col, zorder=4,
                    ls=style.get("ls", "-"), label=label)

        n_pred = cycle_at(traj, seuil)
        if np.isfinite(n_pred):
            ax.axvline(n_pred, color=col, lw=1.0, ls=":", alpha=0.8, zorder=2)

    # verite sur le seuil
    dense = np.interp(np.arange(1, int(n_true.max()) + 1), n_true,
                      np.minimum.accumulate(y_true))
    n_vrai = cycle_at(dense, seuil)
    ax.axhline(seuil, color="crimson", lw=1.2, ls="--", zorder=2)
    if np.isfinite(n_vrai):
        ax.axvline(n_vrai, color=COUL["verite"], lw=1.2, ls=":", zorder=2)

    if truncation is not None and np.isfinite(n_cut):
        ax.axvline(n_cut, color="0.35", lw=1.4, zorder=2)

    if annotate and np.isfinite(n_vrai):
        prem = next(iter(models.values()))
        traj = np.asarray(prem[0] if isinstance(prem, tuple) else prem, float)
        n_pred = cycle_at(traj, seuil)
        if np.isfinite(n_pred):
            ax.annotate(f"$\\Delta n@{seuil:.0f}$ = {n_pred - n_vrai:+.0f}",
                        xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8,
                        bbox=dict(fc="white", ec="0.7", alpha=0.9, pad=2))

    ax.set_xlim(xlim or (0, float(n_true.max()) * 1.35))
    ax.set_ylim(ylim or (55, 105))
    ax.grid(alpha=0.2, zorder=1)
    if legend:
        ax.legend(fontsize=7, loc="lower left")
    return ax


# ==========================================================================
#  7bis.1 - grille LOCO
# ==========================================================================
def fig_loco_grid(trajs, res, loco_qual, path=None):
    path = path or _fig_path("step1_trajectoires_loco.png")
    ids = list(trajs)
    nat = dict(zip(loco_qual["fold"], loco_qual["nature"]))
    ape = dict(zip(res["cell_id"], res["eol_ape_ancrage"]))

    nmax = max(float(np.max(t["n"])) for t in trajs.values())
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), sharex=True, sharey=True)
    for ax, cid in zip(axes.ravel(), ids):
        t = trajs[cid]
        plot_trajectory(cid, {"ancrage": (t["ancrage"], {"color": COUL["ancrage"]}),
                              "baseline refit": (t["baseline"], {"color": COUL["baseline"]})},
                        truth=(t["n"], t["y"]), ax=ax,
                        xlim=(0, nmax * 1.2), ylim=(55, 106))
        a = ape.get(cid, np.nan)
        titre = (f"{_short(cid)}  |  {nat.get(cid, '?')}"
                 + (f"  |  APE n@70 = {a:.1f} %" if np.isfinite(a) else
                    "  |  n@70 non observe"))
        ax.set_title(titre, fontsize=9.5)
    for ax in axes[-1]:
        ax.set_xlabel("cycle")
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH (%, convention nominale)")

    handles = [Line2D([], [], color=COUL["verite"], lw=1.2, label="verite terrain"),
               Line2D([], [], color=COUL["ancrage"], lw=1.8, label="ancrage (LOCO)"),
               Line2D([], [], color=COUL["baseline"], lw=1.8, label="baseline officiel refite"),
               Line2D([], [], color="crimson", lw=1.2, ls="--", label="seuil 70 %"),
               Line2D([], [], color="0.4", lw=1.0, ls=":", label="n@70 (vrai / predit)")]
    fig.legend(handles=handles, ncol=5, loc="lower center", fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("7bis.1 - Trajectoires en leave-one-cell-out : chaque cellule "
                 "predite sans avoir ete vue a l'entrainement\n"
                 "fond bleu = zone quasi-lineaire (>90 %), orange = pre-knee "
                 "(80-90 %), rouge = post-knee (<80 %)", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================================
#  7bis.2 - residus
# ==========================================================================
def fig_residuals_grid(trajs, path=None):
    path = path or _fig_path("step1_residus_par_cycle.png")
    ids = list(trajs)
    nmax = max(float(np.max(t["n"])) for t in trajs.values())
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.0), sharex=True, sharey=True)
    for ax, cid in zip(axes.ravel(), ids):
        t = trajs[cid]
        n = np.asarray(t["n"], float)
        y = np.asarray(t["y"], float)
        for lab, key in (("ancrage", "ancrage"), ("baseline refit", "baseline")):
            traj = np.asarray(t[key], float)
            e = traj[np.clip(n.astype(int) - 1, 0, len(traj) - 1)] - y
            ax.plot(n, e, lw=1.0, color=COUL[key if key in COUL else "ancrage"],
                    label=lab, alpha=0.9)
        # frontieres de zones, reportees sur l'axe des cycles
        env = np.minimum.accumulate(y)
        for lo, col in ((90.0, "0.55"), (80.0, "crimson")):
            b = np.where(env < lo)[0]
            if len(b):
                ax.axvline(n[b[0]], color=col, lw=1.1, ls="--", alpha=0.8)
        ax.axhline(0, color="0.2", lw=1.2)
        ax.set_title(f"{_short(cid)}", fontsize=9.5)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("cycle")
    for ax in axes[:, 0]:
        ax.set_ylabel("errSOH = predit - vrai  (points)")
    axes[0, 0].set_xlim(0, nmax * 1.05)
    axes[0, 0].set_ylim(-22, 22)
    handles = [Line2D([], [], color=COUL["ancrage"], lw=1.4, label="ancrage"),
               Line2D([], [], color=COUL["baseline"], lw=1.4, label="baseline refite"),
               Line2D([], [], color="0.55", lw=1.1, ls="--", label="entree en pre-knee (90 %)"),
               Line2D([], [], color="crimson", lw=1.1, ls="--", label="entree en post-knee (80 %)")]
    fig.legend(handles=handles, ncol=4, loc="lower center", fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("7bis.2 - Structure de l'erreur : errSOH en fonction du cycle\n"
                 "un biais constant, une derive lineaire et une explosion au "
                 "knee ont la meme signature dans un RMSE, pas ici", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================================
#  7bis.3 - extrapolation par niveau de troncature
# ==========================================================================
def fig_extrapolation(cells, fit_predict, troncatures=(95, 90, 85, 80),
                      path=None, cell_ids=None):
    """fit_predict(n_vu, y_vu, c_rate, n_max) -> trajectoire 1..n_max."""
    path = path or _fig_path("step1_extrapolation_troncature.png")
    cells = [c for c in cells if cell_ids is None or c["cell_id"] in cell_ids]
    nmax_glob = max(float(c["n"].max()) for c in cells)

    fig, axes = plt.subplots(len(troncatures), len(cells),
                             figsize=(3.5 * len(cells), 2.9 * len(troncatures)),
                             sharex=True, sharey=True, squeeze=False)
    tableau = []
    for i, tr in enumerate(troncatures):
        for j, cl in enumerate(cells):
            ax = axes[i][j]
            n, y = np.asarray(cl["n"], float), np.asarray(cl["y"], float)
            env = np.minimum.accumulate(y)
            below = np.where(env < tr)[0]
            if not len(below) or below[0] < 30:
                ax.text(0.5, 0.5, f"jamais sous {tr} %", ha="center",
                        va="center", transform=ax.transAxes, fontsize=9,
                        color="0.4")
                ax.set_facecolor("#fafafa")
                if i == 0:
                    ax.set_title(_short(cl["cell_id"]), fontsize=9.5)
                continue
            stop = int(below[0])
            # horizon de prediction genereux : sinon n@70 peut etre "non
            # atteint" par simple manque de longueur, ce qui se confondrait
            # avec un echec de la loi.
            traj = fit_predict(n[:stop], y[:stop], cl["C"], 12000)
            plot_trajectory(cl["cell_id"], {"prediction": (traj, {"color": COUL["ancrage"]})},
                            truth=(n, y), truncation=tr, ax=ax,
                            xlim=(0, nmax_glob * 1.15), ylim=(55, 106),
                            annotate=False, show_knee=False)
            n_vrai = cycle_at(np.interp(np.arange(1, int(n.max()) + 1), n, env),
                              SEUIL_EOL)
            n_pred = cycle_at(traj, SEUIL_EOL)
            err = (n_pred - n_vrai if np.isfinite(n_vrai) and np.isfinite(n_pred)
                   else np.nan)
            sens = ("optimiste" if err > 0 else "pessimiste") if np.isfinite(err) else "-"
            if np.isfinite(err):
                ax.annotate(f"{err:+.0f} cy ({sens})", xy=(0.03, 0.06),
                            xycoords="axes fraction", fontsize=8,
                            color=("#b8860b" if err > 0 else "#1f6f3f"),
                            bbox=dict(fc="white", ec="0.75", alpha=0.9, pad=2))
            tableau.append(dict(troncature_pct=tr, cell_id=cl["cell_id"],
                                cycles_vus=int(n[stop - 1]),
                                n70_vrai=n_vrai, n70_pred=n_pred,
                                err_cycles=err, sens=sens,
                                ape_pct=(abs(err) / n_vrai * 100
                                         if np.isfinite(err) and n_vrai else np.nan)))
            if i == 0:
                ax.set_title(_short(cl["cell_id"]), fontsize=9.5)
        axes[i][0].set_ylabel(f"tronque a {tr} %\nSOH (%)", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("cycle")
    handles = [Line2D([], [], color=COUL["verite"], lw=1.2, label="verite terrain"),
               Line2D([], [], color=COUL["ancrage"], lw=2.0, label="ajuste sur la partie vue"),
               Line2D([], [], color=COUL["ancrage"], lw=1.6, ls="--", label="extrapole"),
               Line2D([], [], color="0.35", lw=1.4, label="point de troncature"),
               Line2D([], [], color="crimson", lw=1.2, ls="--", label="seuil 70 %")]
    fig.legend(handles=handles, ncol=5, loc="lower center", fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("7bis.3 - Jusqu'ou la loi tient-elle en extrapolation ?\n"
                 "chaque ligne n'ajuste que sur ce qui aurait ete observe si "
                 "l'essai s'etait arrete a ce niveau de SOH", fontsize=11)
    fig.tight_layout(rect=[0, 0.025, 1, 0.94])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path, pd.DataFrame(tableau)


# ==========================================================================
#  7bis.4 - les 5 lois sur une meme cellule
# ==========================================================================
def fig_laws_comparison(cell, law_trajs, troncature=85, path=None, n_cut=None):
    path = path or _fig_path("step1_comparaison_lois_troncature85.png")
    n, y = np.asarray(cell["n"], float), np.asarray(cell["y"], float)
    nmax = float(n.max())
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    models = {lab: (tr, {"color": cmap(i)}) for i, (lab, tr) in enumerate(law_trajs.items())}
    plot_trajectory(cell["cell_id"], models, truth=(n, y), truncation=troncature,
                    ax=ax, xlim=(0, nmax * 1.25), ylim=(55, 106), annotate=False)
    ax.set_xlabel("cycle"); ax.set_ylabel("SOH (%)")
    ax.set_title(f"{_short(cell['cell_id'])} - 5 lois, ajustees sur les seuls "
                 f"cycles au-dessus de {troncature} %", fontsize=10)
    h = [Line2D([], [], color=COUL["verite"], lw=1.2, label="verite")]
    h += [Line2D([], [], color=cmap(i), lw=1.8, label=lab)
          for i, lab in enumerate(law_trajs)]
    ax.legend(handles=h, fontsize=8, loc="lower left")

    ax = axes[1]
    env = np.minimum.accumulate(y)
    n_vrai = cycle_at(np.interp(np.arange(1, int(nmax) + 1), n, env), SEUIL_EOL)
    labs, errs = [], []
    for lab, tr in law_trajs.items():
        np_ = cycle_at(tr, SEUIL_EOL)
        labs.append(lab)
        errs.append(np_ - n_vrai if np.isfinite(np_) and np.isfinite(n_vrai) else np.nan)
    col = ["#1f6f3f" if (np.isfinite(e) and e < 0) else "#b8860b" for e in errs]
    ax.barh(labs, errs, color=col, ec="0.3")
    ax.axvline(0, color="0.2", lw=1.2)
    ax.set_xlabel("erreur sur n@70 (cycles) - negatif = pessimiste")
    ax.set_title("Erreur sur le cycle d'atteinte de 70 %", fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    for i, e in enumerate(errs):
        if np.isfinite(e):
            ax.annotate(f"{e:+.0f}", (e, i), fontsize=8,
                        va="center", ha="left" if e >= 0 else "right",
                        xytext=(4 if e >= 0 else -4, 0), textcoords="offset points")
    fig.suptitle("7bis.4 - Ce que les tableaux de l'etape 1 disent en chiffres, "
                 "vu d'un coup d'oeil", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================================
#  7bis.5 - faisceau sur la grille notee
# ==========================================================================
def fig_beam(predict_fn, cells, path=None, t_grid=None, c_grid=(0.5, 0.75, 1.0)):
    path = path or _fig_path("step1_faisceau_grille_notee.png")
    t_grid = t_grid if t_grid is not None else np.arange(25.0, 55.01, 5.0)
    cmap = plt.get_cmap("plasma")
    norm = matplotlib.colors.Normalize(vmin=25, vmax=55)
    ls_map = {0.5: "-", 0.75: "--", 1.0: ":"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for T in t_grid:
        for C in c_grid:
            traj = np.asarray(predict_fn(T, C), float)
            nshow = min(len(traj), 9000)
            ax.plot(np.arange(1, nshow + 1), traj[:nshow], lw=1.4,
                    color=cmap(norm(T)), ls=ls_map.get(C, "-"), alpha=0.9)
    for cl in cells:
        ax.plot(cl["n"], cl["y"], lw=0.0, marker=".", ms=1.2, color="0.15",
                alpha=0.5)
    # le coin vide, mis en evidence
    traj = np.asarray(predict_fn(55.0, 0.5), float)
    ax.plot(np.arange(1, min(len(traj), 9000) + 1), traj[:9000], lw=3.0,
            color="crimson", alpha=0.95, zorder=5)
    ax.axhline(70, color="crimson", lw=1.2, ls="--")
    ax.set_xlim(0, 9000); ax.set_ylim(55, 106)
    ax.set_xlabel("cycle"); ax.set_ylabel("SOH (%)")
    ax.set_title("Faisceau sur 25-55 degC x 0.5-1.0 C\n"
                 "couleur = temperature, style = C-rate, points noirs = "
                 "cellules reelles, rouge epais = coin vide 55 degC / 0.5C",
                 fontsize=10)
    ax.grid(alpha=0.2)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="T ambiante (degC)")

    ax = axes[1]
    Tf = np.arange(25.0, 55.01, 1.0)
    for C in c_grid:
        n70 = [cycle_at(np.asarray(predict_fn(T, C), float), 70.0) for T in Tf]
        ax.plot(Tf, n70, ls=ls_map.get(C, "-"), lw=1.8, label=f"{C}C")
    ax.scatter([55], [cycle_at(np.asarray(predict_fn(55.0, 0.5), float), 70.0)],
               s=140, marker="X", c="crimson", zorder=5, label="coin vide")
    ax.set_xlabel("T ambiante (degC)"); ax.set_ylabel("n@70 predit")
    ax.set_yscale("log"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.set_title("n@70 le long du domaine\n(doit decroitre regulierement en T)",
                 fontsize=10)
    fig.suptitle("7bis.5 - Le coin sans donnee s'insere-t-il dans le faisceau ?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================================
#  Distribution du niveau de SOH au knee, toutes sources
# ==========================================================================
def fig_soh_knee_distribution(cred, soh_k_cible=80.4, path=None):
    """Le critere de decision sur un prior externe : le knee de la source
    se produit-il au meme NIVEAU DE SOH que le notre ?"""
    path = path or _fig_path("knee_soh_distribution.png")
    ordre = ["matr", "snl", "target", "wheeler"]
    lib = {"matr": "MATR / Severson\n169 cellules, 30 degC, charge rapide 3.6-6C",
           "snl": "SNL\n18 cellules LFP, 15-35 degC, 0.5-3C",
           "target": "Cibles du challenge\n6 cellules, 25-55 degC, 0.5-1C",
           "wheeler": "Wheeler\n20 cellules, 50 degC"}
    coul = {"matr": "#8c8c8c", "snl": "#b8860b", "target": "#1f77b4",
            "wheeler": "#a0393c"}

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2),
                             gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    bins = np.arange(72, 100.1, 1.5)
    for s in ordre:
        g = cred[cred["source"] == s]
        if not len(g):
            continue
        ax.hist(g["soh_au_knee"], bins=bins, alpha=0.62, color=coul[s],
                label=f"{s} (n={len(g)})", edgecolor="white", lw=0.6)
    ax.axvline(soh_k_cible, color="crimson", lw=2.0, ls="--", zorder=5)
    ax.annotate(f"SOH_k cible = {soh_k_cible} %", xy=(soh_k_cible, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", color="crimson",
                fontsize=9.5, fontweight="bold")
    ax.set_xlabel("SOH au point de rupture (%)")
    ax.set_ylabel("nombre de cellules")
    ax.set_title("Ou se declenche le knee, par source\n"
                 "(Bacon-Watts, detecteur valide sur b2c12 : 366.1 +/- 1.4 "
                 "contre 365-391 documentes)", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.22)

    ax = axes[1]
    data, labs, cols = [], [], []
    for s in ordre:
        g = cred[cred["source"] == s]
        if len(g):
            data.append(g["soh_au_knee"].to_numpy())
            labs.append(f"{s}\nn={len(g)}")
            cols.append(coul[s])
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55,
                    tick_labels=labs)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.62)
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_linewidth(1.6)
    ax.axvline(soh_k_cible, color="crimson", lw=2.0, ls="--", zorder=5)
    ax.set_xlabel("SOH au point de rupture (%)")
    ax.set_title("Recouvrement avec le regime cible", fontsize=10)
    ax.grid(alpha=0.22, axis="x")

    fig.suptitle("Le critere de decision n'est pas la PRESENCE d'un knee, "
                 "mais son NIVEAU DE SOH", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ==========================================================================
#  Tableau de bord de performance : notre modele vs verite vs baseline
# ==========================================================================
def fig_performance(trajs, res, path=None, titre="modele v3"):
    """Barres par cellule : RMSE et erreur sur n@70, notre modele vs baseline.

    Complete la grille de trajectoires : celle-ci montre COMMENT le modele se
    trompe, celle-la COMBIEN, cellule par cellule - jamais en mediane, puisque
    quatre cellules seulement ont un n@70 observe.
    """
    path = path or _fig_path("perf_vs_baseline.png")
    d = res.copy()
    d["court"] = [_short(c) for c in d["cellule"]]
    x = np.arange(len(d))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0))

    ax = axes[0]
    ax.bar(x - w / 2, d["rmse_v3"], w, label=titre, color=COUL["ancrage"],
           ec="0.25", lw=0.6)
    ax.bar(x + w / 2, d["rmse_base"], w, label="baseline officiel refite",
           color=COUL["baseline"], ec="0.25", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(d["court"], rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel("RMSE pire cas (points de SOH)")
    ax.set_title("Erreur de trajectoire, perimetre note SOH >= 70 %\n"
                 "(max des deux grilles de cycles)", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
    for xi, (a, b) in enumerate(zip(d["rmse_v3"], d["rmse_base"])):
        gagne = a < b
        ax.annotate("v" if gagne else "x", (xi, max(a, b)),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9, color=("#1f6f3f" if gagne else "#a0393c"),
                    fontweight="bold")

    ax = axes[1]
    o = d.dropna(subset=["ape_v3"]).reset_index(drop=True)
    xo = np.arange(len(o))
    ax.bar(xo - w / 2, o["ape_v3"], w, label=titre, color=COUL["ancrage"],
           ec="0.25", lw=0.6)
    ax.bar(xo + w / 2, o["ape_base"], w, label="baseline officiel refite",
           color=COUL["baseline"], ec="0.25", lw=0.6)
    ax.set_xticks(xo)
    ax.set_xticklabels([_short(c) for c in o["cellule"]], rotation=28,
                       ha="right", fontsize=8.5)
    ax.set_ylabel("erreur sur le cycle d'atteinte de 70 % (%)")
    ax.set_title("Prediction de fin de vie\n"
                 "(seules les 4 cellules dont n@70 est observe)", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, axis="y")
    for xi, (a, b) in enumerate(zip(o["ape_v3"], o["ape_base"])):
        for dx, v in ((-w / 2, a), (w / 2, b)):
            ax.annotate(f"{v:.0f}", (xi + dx, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle("Performance en leave-one-cell-out : chaque cellule predite "
                 "sans avoir ete vue a l'entrainement", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
