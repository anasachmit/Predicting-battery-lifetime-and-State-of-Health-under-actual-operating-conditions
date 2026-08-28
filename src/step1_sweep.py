# -*- coding: utf-8 -*-
"""Etape 1 - balayage complet : 6 lois x 4 troncatures x 6 cellules.

Reference de selection : metriques restreintes au perimetre de notation
(SOH vrai >= 70 %). La version non restreinte est calculee en parallele et
rapportee en secondaire.

Critere primaire  : erreur sur le cycle d'atteinte de 70 %, en refit tronque.
Critere secondaire: RMSE dans la zone 70-80 % de SOH.
Selection         : maximum des deux grilles de cycles, jamais leur moyenne.

Diagnostics d'identifiabilite joints a chaque ajustement : correlation maximale
entre parametres, conditionnement du jacobien, et nombre de points disponibles
SOUS le niveau de declenchement du knee (c'est cette derniere quantite qui
determine si les parametres de knee sont contraints par les donnees ou par les
bornes de l'optimiseur).
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import laws as L
from .metrics import cycle_at, evaluate_cell_scoped, SOH_FLOOR
from .runlog import log_run

TRONCATURES = (95, 90, 85, 80)
SOH_K_REF = 80.4          # niveau de declenchement estime conjointement
N_MAX_PRED = 12000


def _identifiability(law, params, n, y, c_rate=None):
    """Correlation max entre parametres et conditionnement, au point ajuste."""
    params = np.asarray(params, float)

    def resid(p):
        if law is L.L5_trajectory:
            t = L.L5_trajectory(int(n.max()), *p, c_rate)
            return t[np.clip(n.astype(int) - 1, 0, len(t) - 1)] - y
        return law(n, *p) - y

    eps = np.maximum(np.abs(params) * 1e-5, 1e-9)
    J = np.empty((len(n), len(params)))
    r0 = resid(params)
    for k in range(len(params)):
        q = params.copy()
        q[k] += eps[k]
        J[:, k] = (resid(q) - r0) / eps[k]
    try:
        JTJ = J.T @ J
        cond = float(np.linalg.cond(JTJ))
        cov = np.linalg.pinv(JTJ) * float(r0 @ r0) / max(len(n) - len(params), 1)
        sd = np.sqrt(np.abs(np.diag(cov)))
        with np.errstate(all="ignore"):
            corr = cov / np.outer(sd, sd)
        off = corr[~np.eye(len(params), dtype=bool)]
        cmax = float(np.nanmax(np.abs(off))) if off.size else np.nan
    except Exception:
        cond, cmax = np.inf, np.nan
    return cond, cmax


def run_sweep(cells, troncatures=TRONCATURES, seed=0, verbose=True):
    rows = []
    for cl in cells:
        n_all = np.asarray(cl["n"], float)
        y_all = np.asarray(cl["y"], float)
        env = np.minimum.accumulate(y_all)
        dense = np.interp(np.arange(1, int(n_all.max()) + 1), n_all, env)
        n70_vrai = cycle_at(dense, 70.0)

        for tr in troncatures:
            n_vu, y_vu, n_cut = L.truncate_at(n_all, y_all, tr)
            if len(n_vu) < 40:
                continue
            # combien de points sous le niveau de declenchement du knee ?
            n_sous_sohk = int((np.minimum.accumulate(y_vu) < SOH_K_REF).sum())

            fits = L.fit_all_laws(n_vu, y_vu, cl["C"], N_MAX_PRED, seed=seed)
            for lab, (traj, info) in fits.items():
                law = L.CANDIDATES[lab]
                params = info.get("params")
                cond, cmax = (_identifiability(law, params, n_vu, y_vu, cl["C"])
                              if params is not None else (np.nan, np.nan))

                m_note = evaluate_cell_scoped(traj, n_all, y_all,
                                              soh_floor=SOH_FLOOR)
                m_tout = evaluate_cell_scoped(traj, n_all, y_all,
                                              soh_floor=None)
                n70_pred = cycle_at(traj, 70.0)
                err = (n70_pred - n70_vrai
                       if np.isfinite(n70_pred) and np.isfinite(n70_vrai)
                       else np.nan)

                rows.append(dict(
                    cellule=cl["cell_id"], T=cl["T"], C=cl["C"],
                    troncature=tr, loi=lab,
                    cycles_vus=int(n_vu[-1]), n_points_vus=len(n_vu),
                    n_points_sous_SOHk=n_sous_sohk,
                    rmse_ajustement=info.get("rmse", np.nan),
                    # --- primaire : perimetre de notation
                    n70_vrai=n70_vrai, n70_pred=n70_pred,
                    err_n70=err,
                    ape_n70=(abs(err) / n70_vrai * 100
                             if np.isfinite(err) and n70_vrai else np.nan),
                    sens=("optimiste" if np.isfinite(err) and err > 0
                          else "pessimiste" if np.isfinite(err) else "-"),
                    rmse_pire_note=max(m_note.get("rmse_uniforme_cycle", np.nan),
                                       m_note.get("rmse_uniforme_soh", np.nan)),
                    rmse_zone_basse_note=m_note.get("rmse_post_knee_note", np.nan),
                    biais_zone_basse_note=m_note.get("biais_median_post_knee_note",
                                                     np.nan),
                    ecart_grilles_note=abs(
                        m_note.get("rmse_uniforme_cycle", np.nan)
                        - m_note.get("rmse_uniforme_soh", np.nan)),
                    # --- secondaire : hors perimetre inclus
                    rmse_pire_tout=max(m_tout.get("rmse_uniforme_cycle", np.nan),
                                       m_tout.get("rmse_uniforme_soh", np.nan)),
                    rmse_zone_basse_tout=m_tout.get("rmse_post_knee", np.nan),
                    # --- identifiabilite
                    cond_jacobien=cond, corr_max_parametres=cmax,
                    n_parametres=(len(params) if params is not None else np.nan),
                    violation_monotonie=m_note.get("taux_violation_monotonie",
                                                   np.nan),
                ))
            if verbose:
                print(f"  {cl['cell_id']:26s} troncature {tr:3d} % : "
                      f"{len(n_vu):5d} pts vus, {n_sous_sohk:5d} sous SOH_k, "
                      f"{len(fits)} lois")
    return pd.DataFrame(rows)


def rank_laws(sw, only_observed=True):
    """Classement des lois. Agregation par MEDIANE des APE sur les cellules a
    n@70 observe, mais les valeurs individuelles restent rapportees a part."""
    d = sw.dropna(subset=["ape_n70"]) if only_observed else sw
    g = d.groupby(["loi", "troncature"])
    out = g.agg(ape_n70_median=("ape_n70", "median"),
                ape_n70_max=("ape_n70", "max"),
                n_cellules=("ape_n70", "size"),
                rmse_zone_basse=("rmse_zone_basse_note", "median"),
                rmse_pire=("rmse_pire_note", "median"),
                corr_max=("corr_max_parametres", "median"),
                cond=("cond_jacobien", "median")).reset_index()
    return out.sort_values(["troncature", "ape_n70_median"])


def guard_threshold(sw, law_prefixes=("L3", "L4", "L5")):
    """Determine empiriquement N : nombre minimal de points sous SOH_k
    au-dela duquel l'ajustement libre du knee devient fiable."""
    d = sw[sw["loi"].str.startswith(tuple(law_prefixes))].dropna(subset=["ape_n70"])
    bins = [(0, 0), (1, 100), (101, 500), (501, 1500), (1501, 10 ** 9)]
    rows = []
    for lo, hi in bins:
        m = d[(d["n_points_sous_SOHk"] >= lo) & (d["n_points_sous_SOHk"] <= hi)]
        if not len(m):
            continue
        rows.append(dict(points_sous_SOHk=f"{lo}-{hi if hi < 10**8 else '+'}",
                         n_cas=len(m),
                         ape_n70_median=float(m["ape_n70"].median()),
                         ape_n70_p90=float(m["ape_n70"].quantile(0.9)),
                         part_optimiste=float((m["sens"] == "optimiste").mean()),
                         corr_max_median=float(m["corr_max_parametres"].median())))
    return pd.DataFrame(rows)
