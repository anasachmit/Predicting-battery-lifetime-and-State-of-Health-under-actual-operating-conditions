# -*- coding: utf-8 -*-
"""Etape C - sanity check inferentiel sur donnees synthetiques.

Deux questions, sur des laboratoires ou la verite est CONNUE :

C.1  RECUPERATION. En ajustant L3 par moindres carres sur les 6 trajectoires
     generees, retrouve-t-on les theta tires ? Ce test valide le generateur
     dans les deux sens :
       - si un parametre qu'on sait identifiable n'est PAS retrouve, le
         generateur ou l'ajustement a un bug ;
       - si un parametre qu'on sait NON identifiable (g et tau_frac pris
         separement) est retrouve parfaitement, c'est que le generateur est
         trop simple : il ne reproduit pas la degenerescence que les vraies
         donnees imposent, et le PFN entraine dessus apprendrait une confiance
         qu'il n'aura pas sur le reel.

C.2  BORNE DE PERFORMANCE. Le meme ajustement par moindres carres, en
     leave-one-cell-out, donne le score qu'un estimateur classique atteint
     dans un monde ou la loi generative est EXACTEMENT L3 et ou l'on connait
     la forme de b(T). C'est le plancher que le PFN devra battre. S'il ne le
     bat pas dans ce monde-la, il ne le battra pas sur le reel.

Protocole d'ajustement - identique a celui du modele soumis, et c'est
deliberé : s0 mesure par cellule sur les cycles <= 10, (SOH_k, g, tau_frac)
communs aux 6 cellules, un b par cellule, puis carte ln b = w0 + w1 * 1000/T_K.
Un protocole plus riche donnerait une borne plus haute et flatterait le PFN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..config import RESULTS
from .generate import generate, l3_trajectory, erreur_modele
from .prior import DEFAULT, FADE_P
from .real_theta import nk_from_sohk

N_REF = 1000.0

RECOVERY_CSV = RESULTS / "pfn" / "recovery_table.csv"
BOUND_CSV = RESULTS / "pfn" / "recovery_borne_loco.csv"

SOH_LO, SOH_HI = 0.5, 119.9
N_MAX_PREDICT = 12000
N_POINTS_FIT = 300     # sous-echantillonnage pour le cout ; sigma <= 0.4 pt


# ------------------------------------------------------------ ajustement ---
def _prepare(traj):
    """Une trajectoire observee -> ce que voit l'ajustement."""
    n, y = traj["n"], traj["y"]
    if len(n) > N_POINTS_FIT:
        idx = np.unique(np.linspace(0, len(n) - 1, N_POINTS_FIT).astype(int))
        n, y = n[idx], y[idx]
    tot = np.flatnonzero(traj["n"] <= 10)
    s0 = (float(np.median(traj["y"][tot])) if tot.size
          else float(np.median(traj["y"][:5])))
    return dict(n=n, y=y, s0=s0, T=traj["theta"]["T"], C=traj["theta"]["C"])


def fit_joint_l3(cells, seed=0, n_restarts=4, p_free=False):
    """(SOH_k, g, tau_frac) communs + un b par cellule, moindres carres.

    `p_free=False` fige l'exposant a 0.8 : c'est EXACTEMENT ce que fait le
    modele soumis, donc la borne honnete a battre. `p_free=True` donne la
    borne d'un estimateur classique plus fort, qui sait que l'exposant varie.
    Les deux sont rapportees : ne montrer que la premiere flatterait le PFN.
    """
    m = len(cells)
    b0 = [max(c["s0"] - c["y"].min(), 1.0) / c["n"].max() ** FADE_P
          for c in cells]
    n_sh = 4 if p_free else 3
    p0 = np.array([82.0, 0.005, 0.5] + ([0.8] if p_free else []) + b0, float)
    lo = np.array([60.0, 1e-7, 0.02] + ([0.30] if p_free else []) + [1e-9] * m, float)
    hi = np.array([98.0, 1.0, 40.0] + ([1.30] if p_free else []) + [5.0] * m, float)

    def _p_of(q):
        return float(q[3]) if p_free else FADE_P

    def resid(q):
        pp = _p_of(q)
        out = []
        for i, c in enumerate(cells):
            pred = l3_trajectory(c["n"], c["s0"], q[n_sh + i], q[0], q[1], q[2], pp)
            out.append((np.clip(pred, SOH_LO, SOH_HI) - c["y"])
                       / np.sqrt(len(c["n"])))
        return np.concatenate(out)

    rng = np.random.default_rng(seed)
    best = None
    for k in range(n_restarts):
        st = p0 if k == 0 else np.clip(p0 * rng.uniform(0.5, 2.0, len(p0)),
                                       lo, hi)
        try:
            r = least_squares(resid, st, bounds=(lo, hi), max_nfev=8000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    q = best.x
    pp = _p_of(q)
    # RMSE NON PONDERE, cellule par cellule. Les residus de `resid` sont
    # divises par sqrt(N_i) pour que chaque cellule pese autant ; les utiliser
    # tels quels donnerait un RMSE de sigma/sqrt(N), sans rapport avec le
    # RMSE des ajustements reels auquel on veut le comparer.
    rmses = []
    for i, c in enumerate(cells):
        pred = np.clip(l3_trajectory(c["n"], c["s0"], q[n_sh + i], q[0], q[1],
                                     q[2], pp), SOH_LO, SOH_HI)
        rmses.append(float(np.sqrt(np.mean((pred - c["y"]) ** 2))))
    b = np.asarray(q[n_sh:], float)
    return dict(soh_k=float(q[0]), g=float(q[1]), tau_frac=float(q[2]),
                p=pp, b=b, loss=b * N_REF ** pp,
                s0=np.array([c["s0"] for c in cells]),
                rmse=float(np.sqrt(np.mean(np.square(rmses)))),
                rmse_par_cellule=rmses)


def fit_arrhenius(T, b):
    """ln b = w0 + w1 * 1000 / T_K, comme le modele soumis."""
    Tk = np.asarray(T, float) + 273.15
    X = np.column_stack([np.ones(len(Tk)), 1000.0 / Tk])
    w, *_ = np.linalg.lstsq(X, np.log(np.asarray(b, float)), rcond=None)
    return float(w[0]), float(w[1])


# ----------------------------------------------------- C.1 recuperation ---
def _derivees(s0, b, soh_k, g, tau_frac, p=FADE_P):
    """Grandeurs derivees, celles dont on ATTEND qu'elles soient identifiables.

    `c = g * tau` est l'amplitude du knee dans l'ancienne parametrisation ;
    `n_k` est le cycle de declenchement ; `perte_15` est la perte
    supplementaire due au knee a 1.5 * n_k, c'est-a-dire l'effet OBSERVABLE du
    couple (g, tau_frac) plutot que chacun de ses membres.
    """
    nk = nk_from_sohk(s0, b, soh_k, p)
    tau = max(tau_frac * nk, 1.0)
    z = np.clip(0.5 * nk / tau, -50.0, 50.0)
    return dict(n_k=nk, c_knee=g * tau,
                perte_15=float(min(g * tau * np.expm1(z), 200.0)))


def _autocorr_resid(cells, fit, lag=1):
    ac = []
    for i, c in enumerate(cells):
        pred = np.clip(l3_trajectory(c["n"], c["s0"], fit["b"][i], fit["soh_k"],
                                     fit["g"], fit["tau_frac"], fit["p"]),
                       SOH_LO, SOH_HI)
        r = pred - c["y"]
        if len(r) > 3 * lag and r.std() > 1e-12:
            ac.append(float(np.corrcoef(r[:-lag], r[lag:])[0, 1]))
    return float(np.median(ac)) if ac else np.nan


def recovery_table(n_labs=120, seed=4242, cfg=DEFAULT, verbose=True,
                   p_free=False):
    """Compare theta tires et theta re-estimes, laboratoire par laboratoire."""
    labs = generate(n_labs, seed=seed, cfg=cfg, progress=verbose)
    rows = []
    for li, lab in enumerate(labs):
        cells = [_prepare(t) for t in lab["trajectoires"]]
        fit = fit_joint_l3(cells, seed=li, p_free=p_free)
        if fit is None:
            continue
        vrai = lab["theta"]
        b_vrai = np.array([t["theta"]["b"] for t in lab["trajectoires"]])
        g_vrai = np.array([t["theta"]["g"] for t in lab["trajectoires"]])
        p_vrai = float(vrai["p"])
        # L(1000) = b * 1000^p : echelle INDEPENDANTE de p, donc la seule
        # comparable entre un tirage a p = 0.45 et un ajustement a p = 0.8.
        # Comparer les `b` bruts n'aurait aucun sens des lors que p varie.
        L_vrai = b_vrai * N_REF ** p_vrai
        T = np.array([c["T"] for c in cells])

        w0_v, w1_v = fit_arrhenius(T, L_vrai)
        w0_f, w1_f = fit_arrhenius(T, fit["loss"])

        d_v = _derivees(np.mean(fit["s0"]), float(np.exp(np.mean(np.log(b_vrai)))),
                        vrai["soh_k"], float(np.exp(np.mean(np.log(g_vrai)))),
                        vrai["tau_frac"], p_vrai)
        d_f = _derivees(np.mean(fit["s0"]), float(np.exp(np.mean(np.log(fit["b"])))),
                        fit["soh_k"], fit["g"], fit["tau_frac"], fit["p"])

        rows.append(dict(
            lab=li, structure=lab["structure_nom"],
            sigma_meas=vrai["sigma_meas"],
            profondeur_min=float(min(t["y"].min() for t in lab["trajectoires"])),
            n_cellules_profondes=int(sum(t["y"].min() <= vrai["soh_k"]
                                         for t in lab["trajectoires"])),
            rmse_ajust=fit["rmse"],
            # autocorrelation du residu au lag 1 : mesure si l'erreur restante est
            # BLANCHE (bruit de mesure) ou STRUCTUREE (erreur de modele).
            autocorr_resid=_autocorr_resid(cells, fit),
            # --- parametres directs
            soh_k_vrai=vrai["soh_k"], soh_k_fit=fit["soh_k"],
            g_vrai=float(np.exp(np.mean(np.log(g_vrai)))), g_fit=fit["g"],
            tau_frac_vrai=vrai["tau_frac"], tau_frac_fit=fit["tau_frac"],
            L_geo_vrai=float(np.exp(np.mean(np.log(L_vrai)))),
            L_geo_fit=float(np.exp(np.mean(np.log(fit["loss"])))),
            L_err_log_med=float(np.median(np.abs(np.log(fit["loss"] / L_vrai)))),
            p_vrai=p_vrai, p_fit=fit["p"],
            sigma_modele=vrai["sigma_modele"],
            # --- carte des conditions
            w1_vrai=w1_v, w1_fit=w1_f,
            ea_vrai=-w1_v * 8.314e-3 * 1000.0, ea_fit=-w1_f * 8.314e-3 * 1000.0,
            # --- grandeurs derivees
            n_k_vrai=d_v["n_k"], n_k_fit=d_f["n_k"],
            c_knee_vrai=d_v["c_knee"], c_knee_fit=d_f["c_knee"],
            perte_15_vrai=d_v["perte_15"], perte_15_fit=d_f["perte_15"],
        ))
    return pd.DataFrame(rows)


def summarize_recovery(t):
    """Une ligne par parametre : erreur de recuperation et verdict."""
    def bloc(nom, col_v, col_f, echelle, unite, attendu):
        v = t[col_v].to_numpy(float)
        f = t[col_f].to_numpy(float)
        ok = np.isfinite(v) & np.isfinite(f)
        v, f = v[ok], f[ok]
        if echelle == "log":
            v_, f_ = np.log(np.clip(v, 1e-12, None)), np.log(np.clip(f, 1e-12, None))
            err = np.abs(f_ - v_)
            err_med = float(np.median(err))
            err_txt = f"x{np.exp(err_med):.2f}"
            biais = f"x{np.exp(float(np.median(f_ - v_))):.2f}"
        else:
            err = np.abs(f - v)
            err_med = float(np.median(err))
            err_txt = f"{err_med:.2f} {unite}"
            biais = f"{float(np.median(f - v)):+.2f} {unite}"
        r = float(np.corrcoef(v, f)[0, 1]) if len(v) > 2 and v.std() > 0 else np.nan
        # Un parametre est dit RECUPERE si la correlation vrai/estime depasse
        # 0.7 : en dessous, l'ajustement rend surtout du bruit.
        return dict(parametre=nom, echelle=echelle, n=len(v),
                    erreur_mediane=err_txt, biais_median=biais,
                    correlation=round(r, 3),
                    verdict=("recupere" if r >= 0.7 else
                             "partiel" if r >= 0.4 else "NON identifiable"),
                    attendu=attendu)

    # Colonne `attendu` : ce que le banc a DEJA etabli sur donnees reelles,
    # avant tout PFN. Voir src/knee_param.py : dans la parametrisation (c, tau)
    # la correlation entre les deux parametres de forme du knee vaut 0.9998 -
    # ils sont litteralement un seul parametre. La reparametrisation en
    # (g, tau) la ramene a 0.919, et c'est `g` = pente initiale du knee qui
    # porte le signal, `tau` qui reste mal determine. L'attendu ci-dessous
    # suit donc cette mesure : g identifiable, tau_frac non.
    return pd.DataFrame([
        bloc("L(1000) (echelle de perte)", "L_geo_vrai", "L_geo_fit", "log", "",
             "recupere"),
        bloc("Ea apparente de L(T)", "ea_vrai", "ea_fit", "lin", "kJ/mol",
             "recupere"),
        bloc("SOH_k", "soh_k_vrai", "soh_k_fit", "lin", "pts", "partiel"),
        bloc("g (pente initiale du knee)", "g_vrai", "g_fit", "log", "",
             "recupere"),
        bloc("tau_frac (seul)", "tau_frac_vrai", "tau_frac_fit", "log", "",
             "NON identifiable"),
        bloc("c = g * tau (combine)", "c_knee_vrai", "c_knee_fit", "log", "",
             "partiel"),
        bloc("n_k (cycle du knee)", "n_k_vrai", "n_k_fit", "log", "", "recupere"),
        bloc("perte du knee a 1.5 n_k", "perte_15_vrai", "perte_15_fit", "log",
             "", "recupere"),
    ])


def recovery_par_profondeur(t):
    """Recuperation stratifiee par nombre de cellules descendues sous SOH_k.

    C'est le test qui dit si le generateur reproduit la DEGENERESCENCE reelle.
    Sur les 6 cellules cible, 4 seulement franchissent 80 % et deux d'entre
    elles a peine. Si la recuperation du knee est aussi bonne avec 1 cellule
    profonde qu'avec 6, le generateur ne reproduit pas cette difficulte et le
    PFN entraine dessus sera trop confiant sur le reel.
    """
    out = []
    for k, g in t.groupby(t.n_cellules_profondes.clip(upper=4)):
        lab = f"{int(k)} cellule(s) sous SOH_k" + (" ou plus" if k == 4 else "")
        def cor(a, b, log=True):
            x, y = g[a].to_numpy(float), g[b].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 4:
                return np.nan
            x, y = (np.log(np.clip(x[m], 1e-12, None)),
                    np.log(np.clip(y[m], 1e-12, None))) if log else (x[m], y[m])
            return round(float(np.corrcoef(x, y)[0, 1]), 3) if x.std() > 0 else np.nan
        out.append(dict(strate=lab, n_labs=len(g),
                        corr_L=cor("L_geo_vrai", "L_geo_fit"),
                        corr_SOH_k=cor("soh_k_vrai", "soh_k_fit", log=False),
                        corr_g=cor("g_vrai", "g_fit"),
                        corr_tau_frac=cor("tau_frac_vrai", "tau_frac_fit"),
                        corr_perte_15=cor("perte_15_vrai", "perte_15_fit"),
                        err_SOH_k_med=round(float(np.median(np.abs(
                            g.soh_k_fit - g.soh_k_vrai))), 2)))
    return pd.DataFrame(out)


# ------------------------------------------------- C.2 borne LOCO ---------
def _n_at(traj, seuil):
    below = np.flatnonzero(np.asarray(traj, float) < seuil)
    if not below.size or below[0] == 0:
        return np.nan
    j = int(below[0])
    return float(np.interp(seuil, [traj[j], traj[j - 1]], [j + 1, j]))


def verite_latente(held, n):
    """Trajectoire VRAIE de la cellule : L3 + erreur de modele, puis bornee.

    L'erreur de modele fait partie de la verite, pas de l'observation : une
    cellule reelle ne suit pas L3. Evaluer la borne contre la L3 pure
    reviendrait a noter l'estimateur sur un monde ou il est bien specifie -
    exactement l'illusion que ce generateur corrige.
    """
    th = held["theta"]
    y = l3_trajectory(n, th["s0"], th["b"], th["soh_k"], th["g"],
                      th["tau_frac"], th["p"])
    y = y + erreur_modele(n, th.get("c_modele", ()),
                          th.get("sigma_modele", 0.0), held["n_max"],
                          th.get("alpha_modele", 1.0))
    return np.clip(y, SOH_LO, SOH_HI)


def loco_bound(n_labs=60, seed=99, cfg=DEFAULT, verbose=True, p_free=False):
    """Borne de performance : L3 + Arrhenius, en leave-one-cell-out.

    La cellule retiree est predite a partir des 5 autres, exactement comme le
    modele soumis le fait sur le reel. La reference est la trajectoire latente
    de la cellule retiree, ERREUR DE MODELE COMPRISE, sur le perimetre note
    SOH >= 70 %.

    La carte des conditions est ajustee sur L(1000) = b * 1000^p plutot que sur
    `b` : c'est la grandeur qui ne depend pas de l'exposant, donc la seule dont
    la regression en 1/T ait un sens quand p peut varier.
    """
    labs = generate(n_labs, seed=seed, cfg=cfg, progress=verbose)
    rows = []
    for li, lab in enumerate(labs):
        cells = [_prepare(t) for t in lab["trajectoires"]]
        for i, held in enumerate(lab["trajectoires"]):
            train = [c for j, c in enumerate(cells) if j != i]
            fit = fit_joint_l3(train, seed=li * 10 + i, p_free=p_free)
            if fit is None:
                continue
            w0, w1 = fit_arrhenius([c["T"] for c in train], fit["loss"])
            L_pred = float(np.exp(w0 + w1 * 1000.0 / (cells[i]["T"] + 273.15)))
            b_pred = L_pred / N_REF ** fit["p"]
            s0_pred = float(np.mean(fit["s0"]))
            n = np.arange(1, N_MAX_PREDICT + 1, dtype=float)
            pred = np.clip(l3_trajectory(n, s0_pred, b_pred, fit["soh_k"],
                                         fit["g"], fit["tau_frac"], fit["p"]),
                           SOH_LO, SOH_HI)
            vrai = verite_latente(held, n)
            note = vrai >= 70.0
            if note.sum() < 50:
                continue
            rmse = float(np.sqrt(np.mean((pred[note] - vrai[note]) ** 2)))
            n70_v, n70_p = _n_at(vrai, 70.0), _n_at(pred, 70.0)
            ape = (abs(n70_p - n70_v) / n70_v * 100.0
                   if np.isfinite(n70_v) and np.isfinite(n70_p) else np.nan)
            rows.append(dict(lab=li, cell=i, structure=lab["structure_nom"],
                             T=cells[i]["T"], C=cells[i]["C"],
                             p_vrai=held["theta"]["p"], p_fit=fit["p"],
                             profondeur=float(held["y"].min()),
                             rmse_zone_notee=rmse, n70_vrai=n70_v,
                             n70_predit=n70_p, ape_n70=ape))
    return pd.DataFrame(rows)


def summarize_bound(t):
    def q(s):
        s = s.dropna()
        return dict(n=len(s), mediane=round(float(s.median()), 2),
                    q25=round(float(s.quantile(0.25)), 2),
                    q75=round(float(s.quantile(0.75)), 2))
    out = [dict(metrique="RMSE zone notee (pts de SOH)", **q(t.rmse_zone_notee)),
           dict(metrique="APE sur n@70 (%)", **q(t.ape_n70))]
    for s, g in t.groupby("structure"):
        out.append(dict(metrique=f"  APE n@70, structure = {s}", **q(g.ape_n70)))
    prof = t[t.profondeur > 70.0]
    if len(prof):
        out.append(dict(metrique="  APE n@70, cellules arretees > 70 %",
                        **q(prof.ape_n70)))
    prof = t[t.profondeur <= 70.0]
    if len(prof):
        out.append(dict(metrique="  APE n@70, cellules descendues sous 70 %",
                        **q(prof.ape_n70)))
    return pd.DataFrame(out)


# --------------------------------------------------------------- runner ----
def run(n_labs_recovery=120, n_labs_bound=60, verbose=True):
    t = recovery_table(n_labs_recovery, verbose=verbose)
    resume = summarize_recovery(t)
    strates = recovery_par_profondeur(t)
    b = loco_bound(n_labs_bound, verbose=verbose, p_free=False)
    resume_b = summarize_bound(b)
    b_pf = loco_bound(n_labs_bound, verbose=verbose, p_free=True)
    resume_b_pf = summarize_bound(b_pf)

    RECOVERY_CSV.parent.mkdir(parents=True, exist_ok=True)
    resume.to_csv(RECOVERY_CSV, index=False)
    t.to_csv(RECOVERY_CSV.with_name("recovery_brut.csv"), index=False)
    strates.to_csv(RECOVERY_CSV.with_name("recovery_par_profondeur.csv"),
                   index=False)
    b.to_csv(BOUND_CSV, index=False)
    resume_b.to_csv(BOUND_CSV.with_name("recovery_borne_resume.csv"), index=False)
    b_pf.to_csv(BOUND_CSV.with_name("recovery_borne_loco_pfree.csv"), index=False)
    resume_b_pf.to_csv(BOUND_CSV.with_name("recovery_borne_resume_pfree.csv"),
                       index=False)
    return dict(brut=t, resume=resume, strates=strates, borne=b,
                resume_borne=resume_b, borne_pfree=b_pf,
                resume_borne_pfree=resume_b_pf)


if __name__ == "__main__":
    out = run()
    pd.set_option("display.width", 200)
    print()
    print("=== C.1  recuperation des theta (donnees synthetiques) ===")
    print(out["resume"].to_string(index=False))
    print()
    print("=== C.1bis  recuperation stratifiee par profondeur des essais ===")
    print(out["strates"].to_string(index=False))
    print()
    print("=== C.2a borne : L3 + Arrhenius, p FIGE a 0.8 (le modele soumis) ===")
    print(out["resume_borne"].to_string(index=False))
    print()
    print("=== C.2b borne : idem mais p LIBRE (estimateur classique renforce) ===")
    print(out["resume_borne_pfree"].to_string(index=False))
    print()
    print("tables ecrites :", RECOVERY_CSV, "|", BOUND_CSV)
