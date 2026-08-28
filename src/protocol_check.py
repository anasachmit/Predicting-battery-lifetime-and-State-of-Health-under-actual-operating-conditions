"""Deux verifications qui conditionnent la forme de L5.

(1) La duree de cycle derive avec le vieillissement (-5 a -28 %). Est-elle
    proportionnelle au SOH ? Si oui, t(n) n'est PAS lineaire en n et doit etre
    integre comme t(n) = a/C * somme(SOH) + b*n -- calculable a la prediction.
(2) s0 (SOH initial en convention nominale) depend-il de T ? Si non, le
    parametrer en T ajoute un degre de liberte pour rien.
"""
import numpy as np
import pandas as pd


def duration_vs_soh(dur, unified):
    """Regression duree ~ alpha/C * SOH + beta, par cellule."""
    rows = []
    for cid, g in dur.groupby("cell_id"):
        u = unified[unified["cell_id"] == cid][["cycle_n", "SOH"]]
        g = g.dropna(subset=["dur_h"])
        med = g["dur_h"].median()
        g = g[(g["dur_h"] > 0) & (g["dur_h"] < 3 * med)]
        m = g.merge(u, on="cycle_n", how="inner").dropna(subset=["SOH", "dur_h"])
        if len(m) < 50:
            continue
        s = m["SOH"].to_numpy(float)
        d = m["dur_h"].to_numpy(float)
        A = np.column_stack([s, np.ones_like(s)])
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        pred = A @ coef
        r2 = 1 - ((d - pred) ** 2).sum() / ((d - d.mean()) ** 2).sum()
        # comparaison : duree constante (hypothese t(n) lineaire en n)
        r2_const = 0.0
        rows.append(dict(cell_id=cid, T_amb=m["T_amb"].iloc[0],
                         C_rate=m["C_rate"].iloc[0], n=len(m),
                         pente_SOH_h=coef[0], ordonnee_h=coef[1],
                         r2_dur_vs_SOH=r2,
                         part_variable=coef[0] * (s.max() - s.min()) / d.mean(),
                         corr=float(np.corrcoef(s, d)[0, 1])))
    return pd.DataFrame(rows).sort_values(["C_rate", "T_amb"])


def s0_vs_T(unified_target, n_max=10):
    """SOH initial (convention nominale) par cellule, contre T."""
    rows = []
    for cid, g in unified_target.groupby("cell_id"):
        e = g[g["cycle_n"] <= n_max]
        rows.append(dict(cell_id=cid, T_amb=g["T_amb"].iloc[0],
                         C_rate=g["C_rate_dchg"].iloc[0],
                         s0_nom_pct=100 * float(e["SOH_nom"].median()),
                         Q_ref_Ah=float(g["Q_ref"].iloc[0])))
    d = pd.DataFrame(rows).sort_values("T_amb")
    T = d["T_amb"].to_numpy(float)
    y = d["s0_nom_pct"].to_numpy(float)
    A = np.column_stack([np.ones_like(T), T])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    # test de pente nul (t de Student, 4 ddl)
    se = np.sqrt(ss_res / (len(T) - 2) * np.linalg.inv(A.T @ A)[1, 1])
    from scipy import stats
    tstat = coef[1] / se
    p = 2 * (1 - stats.t.cdf(abs(tstat), len(T) - 2))
    stat = dict(pente_pct_par_degC=float(coef[1]), se=float(se),
                t=float(tstat), p_value=float(p),
                r2=float(1 - ss_res / ss_tot),
                etendue_s0_pct=float(y.max() - y.min()))
    return d, stat
