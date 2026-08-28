"""Sonde: le critere R2>0.99 discrimine-t-il vraiment les lois candidates ?"""
import sys, pathlib
sys.path.insert(0, r"c:\Users\anasa\Desktop\batterylife_health")
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

df = pd.read_parquet(r"c:\Users\anasa\Desktop\batterylife_health\data\unified.parquet")

def f_pow(n, s0, a, p):            return s0 - a * n**p
def f_user(n, s0, a, b, c, nk, ta):
    return s0 - a*np.sqrt(n) - b*n - c*(np.expm1(np.clip((n-nk)/ta, -50, 50)))*(n > nk)

def fit(fun, n, y, p0, lo, hi):
    try:
        pp, _ = curve_fit(fun, n, y, p0=p0, bounds=(lo, hi), maxfev=60000)
        r = y - fun(n, *pp)
        r2 = 1 - (r**2).sum() / ((y - y.mean())**2).sum()
        return r2, np.sqrt((r**2).mean()), pp
    except Exception as e:
        return np.nan, np.nan, None

rows = []
for cid, g in df.groupby("cell_id"):
    g = g.sort_values("cycle_n")
    n = g.cycle_n.to_numpy(float); y = g.SOH.to_numpy(float) * 100
    m = np.isfinite(y) & (n > 0)
    n, y = n[m], y[m]
    if len(n) < 15: continue
    nm = n.max()
    r2p, rmp, _ = fit(f_pow, n, y, [100, 1e-3, 1.0], [50, 0, 0.2], [120, 50, 3.0])
    r2u, rmu, pu = fit(f_user, n, y, [100, 0.05, 1e-3, 1e-3, 0.7*nm, 0.2*nm],
                       [50, 0, 0, 0, 0.05*nm, 1.0], [120, 10, 1, 50, 2.0*nm, 10*nm])
    # erreur dans la zone qui compte
    sub = y < 80
    rm_low_p = np.sqrt(np.mean((y[sub] - f_pow(n[sub], *fit(f_pow,n,y,[100,1e-3,1.0],[50,0,0.2],[120,50,3.0])[2]))**2)) if sub.sum() > 3 else np.nan
    rows.append(dict(cell_id=cid, source=g.source.iloc[0], npts=len(n),
                     span=y.max()-y.min(), r2_pow=r2p, rmse_pow=rmp,
                     r2_user=r2u, rmse_user=rmu, rmse_pow_below80=rm_low_p))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.groupby("source")[["span","r2_pow","r2_user","rmse_pow","rmse_user"]]
        .agg(["median", lambda x: np.nanpercentile(x, 10)]).round(4).to_string())
print()
print("R2 loi puissance simple  -> mediane %.5f  p10 %.5f  | passe le seuil (0.99/0.97) ? %s"
      % (r.r2_pow.median(), np.nanpercentile(r.r2_pow,10),
         r.r2_pow.median()>0.99 and np.nanpercentile(r.r2_pow,10)>0.97))
print("RMSE (points de SOH)     -> puissance %.3f   utilisateur %.3f"
      % (r.rmse_pow.median(), r.rmse_user.median()))
print("\nRMSE sous 80%% SOH (loi puissance), par source:")
print(r.groupby("source").rmse_pow_below80.median().round(3).to_string())
