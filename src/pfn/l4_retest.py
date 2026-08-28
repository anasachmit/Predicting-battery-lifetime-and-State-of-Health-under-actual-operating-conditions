# -*- coding: utf-8 -*-
"""Retest L3 (p fige a 0.8) contre L4 (p libre) DANS LE REGIME DE SOUMISSION.

Pourquoi ce retest
------------------
L4 avait ete ecarte a l'etape 1 sur 16.6 % contre 9.4 % d'erreur mediane, mais
ce verdict a ete rendu dans un regime qui n'est PAS celui de la soumission :

  * en REFIT TRONQUE (donnees coupees a 85 % de SOH), alors que la soumission
    ajuste sur les six trajectoires COMPLETES ;
  * avec des bornes d'optimiseur qui saturaient - l'exposant tapait 0.35 et
    SOH_k tapait 98, ce qui rendait les estimations ininterpretables.

Le travail sur le prior PFN a montre deux choses qui remettent ce verdict en
cause. D'abord, qu'un estimateur qui INFERE l'exposant localise SOH_k a 1.33
point la ou un estimateur qui le FIGE a 0.8 le rate a 3.42 - autrement dit que
figer p n'est pas neutre, c'est une mauvaise specification. Ensuite, qu'une
fois les bornes elargies, les six cellules cible donnent des estimations
propres, aucune en butee : p de 0.454 a 0.588, SOH_k de 84 a 92.

D'ou ce retest, dans le regime exact de la soumission.

Protocole - copie fidele de `my_model/model_template.py`
--------------------------------------------------------
* leave-one-cell-out sur les 6 cellules cible, trajectoires COMPLETES ;
* `s0` mesure par cellule sur les cycles <= 10 ;
* (SOH_k, g, tau_frac) communs aux 5 cellules d'entrainement, un `b` par
  cellule, plus `p` commun pour L4 ;
* regle de garde sur la profondeur (GARDE_PTS, GARDE_MIN_CELLS) ;
* carte des conditions ln b = w0 + w1 * 1000/T_K, sans terme en ln(C) ;
* metriques restreintes a SOH vrai >= 70 %, selection sur le MAXIMUM des deux
  grilles de cycles - jamais leur moyenne ;
* audit de plausibilite complet, coin vide 55 degC / 0.5C compris.

Critere de decision, pose AVANT de voir les resultats
------------------------------------------------------
L4 remplace L3 en soumission si et seulement si :
  1. il ne degrade l'APE sur n@70 sur AUCUN des quatre folds ou n@70 est
     observe (pas de compensation entre folds, pas de mediane) ;
  2. il passe l'audit de plausibilite INTEGRALEMENT.
Sinon L3 reste, et le resultat est note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..config import RESULTS
from ..metrics import evaluate_cell_scoped
from ..plausibility import sweep, audit
from .real_theta import load_real_cells

OUT_DIR = RESULTS / "pfn"
SOH_LO, SOH_HI = 0.5, 119.9
N_MAX_PREDICT = 12000
GARDE_PTS = 10.0
GARDE_MIN_CELLS = 2
SOH_K_INIT = 80.4

# Bornes ELARGIES, celles qui ont donne des estimations propres sur les 6
# cellules cible (aucune butee). Les bornes d'origine (p dans [0.2, 3.0],
# SOH_k dans [60, 98] avec p libre) saturaient.
P_BOUNDS = (0.35, 1.60)
SOHK_BOUNDS = (60.0, 98.0)


def _traj(n, s0, b, soh_k, g, tau_frac, p):
    n = np.asarray(n, float)
    soh = s0 - b * np.power(n, p)
    if g > 0.0:
        nk = float(np.power(max((s0 - soh_k) / max(b, 1e-12), 1e-9), 1.0 / p))
        tau = max(tau_frac * nk, 1.0)
        z = np.clip((n - nk) / tau, -50.0, 50.0)
        soh = soh - np.where(n > nk, g * tau * np.expm1(z), 0.0)
    return np.clip(soh, SOH_LO, SOH_HI)


GARDE_P_SOH = 86.0     # profondeur exigee pour oser liberer l'exposant
GARDE_P_MIN_CELLS = 2


def fit_joint(cells, p_free, seed=0, n_restarts=8, p_guard=False):
    """(SOH_k, g, tau_frac[, p]) communs + un b par cellule."""
    m = len(cells)
    avec_knee = sum(c["soh_min"] - SOH_K_INIT <= GARDE_PTS
                    for c in cells) >= GARDE_MIN_CELLS
    # REGLE DE GARDE SUR L'EXPOSANT, calquee sur celle du knee. Liberer p
    # exige des donnees PROFONDES : c'est la courbure sous 86 % qui separe un
    # exposant de 0.4 d'un exposant de 0.8. Au-dessus, les deux ajustent aussi
    # bien les donnees vues et divergent completement en extrapolation.
    profond_p = sum(c["soh_min"] <= GARDE_P_SOH
                    for c in cells) >= GARDE_P_MIN_CELLS
    if p_guard and not profond_p:
        p_free = False
    b0 = [max(c["s0"] - c["y"].min(), 1.0) / c["n"].max() ** 0.8 for c in cells]

    tete = [SOH_K_INIT, 0.005, 0.25] + ([0.8] if p_free else [])
    lo_t = [SOHK_BOUNDS[0], 0.0, 0.02] + ([P_BOUNDS[0]] if p_free else [])
    hi_t = [SOHK_BOUNDS[1], 1.0, 5.0] + ([P_BOUNDS[1]] if p_free else [])
    ns = len(tete)
    p0 = np.array(tete + b0, float)
    lo = np.array(lo_t + [1e-9] * m, float)
    hi = np.array(hi_t + [5.0] * m, float)

    def unpack(q):
        pp = float(q[3]) if p_free else 0.8
        gg = float(q[1]) if avec_knee else 0.0
        return float(q[0]), gg, float(q[2]), pp

    def resid(q):
        sk, gg, tf, pp = unpack(q)
        out = []
        for i, c in enumerate(cells):
            pred = _traj(c["n"], c["s0"], q[ns + i], sk, gg, tf, pp)
            out.append((pred - c["y"]) / np.sqrt(len(c["n"])))
        return np.concatenate(out)

    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        st = p0 if k == 0 else np.clip(p0 * rng.uniform(0.5, 1.8, len(p0)), lo, hi)
        try:
            r = least_squares(resid, st, bounds=(lo, hi), max_nfev=30000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("ajustement conjoint echoue")
    q = best.x
    sk, gg, tf, pp = unpack(q)
    b = np.asarray(q[ns:], float)
    # butees : une estimation en butee n'est pas une estimation
    butees = []
    if p_free and (pp <= P_BOUNDS[0] + 1e-3 or pp >= P_BOUNDS[1] - 1e-3):
        butees.append("p")
    if sk <= SOHK_BOUNDS[0] + 1e-3 or sk >= SOHK_BOUNDS[1] - 1e-3:
        butees.append("SOH_k")
    return dict(soh_k=sk, g=gg, tau_frac=tf, p=pp, b=b,
                s0=float(np.mean([c["s0"] for c in cells])),
                regime=("avec_knee" if avec_knee else "sans_knee")
                       + ("" if p_free else "|p_fige"),
                butees=",".join(butees),
                rmse=float(np.sqrt(np.mean(best.fun ** 2))))


def carte_conditions(cells, b):
    """ln b = w0 + w1 * 1000/T_K. Pas de terme en ln(C) - choix 5 de v3."""
    Tk = np.array([c["T"] for c in cells], float) + 273.15
    X = np.column_stack([np.ones(len(Tk)), 1000.0 / Tk])
    w, *_ = np.linalg.lstsq(X, np.log(np.asarray(b, float)), rcond=None)
    return float(w[0]), float(w[1])


def predicteur(fit, w):
    def f(T, C):
        b = float(np.exp(w[0] + w[1] * 1000.0 / (float(T) + 273.15)))
        n = np.arange(1, N_MAX_PREDICT + 1, dtype=float)
        return _traj(n, fit["s0"], b, fit["soh_k"], fit["g"], fit["tau_frac"],
                     fit["p"])
    return f


def run_loco(p_free, cells=None, seed=0, p_guard=False):
    cells = cells or [c for c in load_real_cells() if c["source"] == "target"]
    rows, params = [], []
    for i, held in enumerate(cells):
        train = [c for j, c in enumerate(cells) if j != i]
        fit = fit_joint(train, p_free=p_free, seed=seed * 100 + i,
                        p_guard=p_guard)
        w = carte_conditions(train, fit["b"])
        pred = predicteur(fit, w)(held["T"], held["C"])
        ev = evaluate_cell_scoped(pred, held["n"], held["y"], soh_floor=70.0)
        pire = max(ev.get("rmse_uniforme_cycle", np.nan),
                   ev.get("rmse_uniforme_soh", np.nan))
        rows.append(dict(
            fold=held["cell_id"], T=held["T"], C=held["C"],
            rmse_pire=pire, eol_ape=ev.get("eol_ape", np.nan),
            n70_vrai=ev.get("n70_true", np.nan),
            n70_predit=ev.get("n70_pred", np.nan),
            eol_observe=ev.get("eol_observe", False)))
        params.append(dict(fold=held["cell_id"], soh_k=fit["soh_k"],
                           g=fit["g"], tau_frac=fit["tau_frac"], p=fit["p"],
                           w0=w[0], w1=w[1], regime=fit["regime"],
                           butees=fit["butees"], rmse_ajust=fit["rmse"]))
    return pd.DataFrame(rows), pd.DataFrame(params)


def audit_complet(p_free, cells=None, seed=0):
    """Ajustement sur les 6 cellules, puis balayage de toute la grille notee."""
    cells = cells or [c for c in load_real_cells() if c["source"] == "target"]
    fit = fit_joint(cells, p_free=p_free, seed=seed)
    w = carte_conditions(cells, fit["b"])
    sw = sweep(predicteur(fit, w))
    checks, piv = audit(sw)
    coin = sw[(sw.T_amb == 55.0) & (sw.C_rate == 0.5)]
    return checks, piv, fit, w, sw, coin


def run(seed=0):
    cells = [c for c in load_real_cells() if c["source"] == "target"]
    out = {}
    for nom, p_free in (("L3 (p = 0.8)", False), ("L4 (p libre)", True)):
        loco, par = run_loco(p_free, cells, seed)
        checks, piv, fit, w, sw, coin = audit_complet(p_free, cells, seed)
        out[nom] = dict(loco=loco, params=par, checks=checks, fit=fit,
                        sweep=sw, coin=coin, piv=piv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for nom, d in out.items():
        tag = "L3" if nom.startswith("L3") else "L4"
        d["loco"].to_csv(OUT_DIR / f"l4retest_loco_{tag}.csv", index=False)
        d["params"].to_csv(OUT_DIR / f"l4retest_params_{tag}.csv", index=False)
    return out


def verdict(out):
    """Applique le critere pose avant de voir les resultats."""
    a = out["L3 (p = 0.8)"]["loco"].set_index("fold")
    b = out["L4 (p libre)"]["loco"].set_index("fold")
    obs = a.index[a.eol_observe.astype(bool)]
    degrade = [f for f in obs if b.loc[f, "eol_ape"] > a.loc[f, "eol_ape"] + 1e-9]
    audit_ok = all(out["L4 (p libre)"]["checks"].values())
    return dict(
        folds_avec_n70=list(obs),
        folds_degrades=degrade,
        aucun_fold_degrade=len(degrade) == 0,
        audit_L4_complet=audit_ok,
        audit_L3_complet=all(out["L3 (p = 0.8)"]["checks"].values()),
        L4_remplace_L3=(len(degrade) == 0 and audit_ok))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    out = run()
    for nom, d in out.items():
        print(f"========== {nom} ==========")
        print(d["loco"].round(3).to_string(index=False))
        print()
        print(d["params"].round(4).to_string(index=False))
        print()
        print("audit de plausibilite :")
        for k, v in d["checks"].items():
            print(f"   {'OK   ' if v else 'ECHEC'}  {k}")
        print(f"   coin vide 55 degC/0.5C : n@70 = "
              f"{float(d['coin'].n_at_70.iloc[0]):.0f}")
        print()
    v = verdict(out)
    print("========== VERDICT ==========")
    for k, val in v.items():
        print(f"  {k} : {val}")
