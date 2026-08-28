# -*- coding: utf-8 -*-
"""Detection de knee, applicable a n'importe quelle source.

Critere principal : ecart maximal a la corde (kneedle), sans derivee, donc
robuste au bruit de mesure. Voir `src/metrics.knee_onset` pour les deux
definitions ecartees et pourquoi.

Un point important qui n'apparait qu'en comparant plusieurs jeux : l'ecart a la
corde depend de la PROFONDEUR parcourue. Une cellule arretee a 80 % de SOH n'a
pas eu le temps de montrer un coude, meme si son mecanisme en produirait un.
Comparer des jeux qui ne descendent pas au meme niveau exige donc de tronquer
tout le monde a la meme profondeur avant de mesurer.
"""
import numpy as np
import pandas as pd


def chord_deviation(cycles, soh_pct, soh_stop=None):
    """Ecart maximal a la corde, sur la trajectoire tronquee a `soh_stop`.

    Renvoie (ecart, cycle_du_knee, profondeur_parcourue).
    Positif = la courbe reste au-dessus de sa corde = elle plonge tard = knee.
    Proche de zero = trajectoire quasi rectiligne.
    """
    cycles = np.asarray(cycles, float)
    soh = np.asarray(soh_pct, float)
    ok = np.isfinite(cycles) & np.isfinite(soh)
    cycles, soh = cycles[ok], soh[ok]
    if len(cycles) < 15:
        return np.nan, np.nan, np.nan
    o = np.argsort(cycles)
    cycles, soh = cycles[o], soh[o]
    env = np.minimum.accumulate(soh)

    if soh_stop is not None:
        below = np.where(env < soh_stop)[0]
        if not len(below):
            return np.nan, np.nan, float(env[0] - env[-1])
        stop = below[0] + 1
        cycles, env = cycles[:stop], env[:stop]
    if len(cycles) < 15:
        return np.nan, np.nan, np.nan

    span = env[0] - env[-1]
    if span <= 1e-9:
        return np.nan, np.nan, float(span)
    x = (cycles - cycles[0]) / max(cycles[-1] - cycles[0], 1e-9)
    y = (env - env[-1]) / span
    d = y - (1.0 - x)
    i = int(np.argmax(d))
    return float(d[i]), float(cycles[i]), float(span)


def curvature_ratio(cycles, soh_pct, soh_stop=None):
    """Rapport entre la vitesse de perte sur le dernier quart et sur le premier.

    Complement independant de l'ecart a la corde, et plus directement
    interpretable : > 1 signifie que la degradation ACCELERE.
    """
    cycles = np.asarray(cycles, float)
    soh = np.asarray(soh_pct, float)
    ok = np.isfinite(cycles) & np.isfinite(soh)
    cycles, soh = cycles[ok], soh[ok]
    if len(cycles) < 20:
        return np.nan
    o = np.argsort(cycles)
    cycles, env = cycles[o], np.minimum.accumulate(soh[o])
    if soh_stop is not None:
        below = np.where(env < soh_stop)[0]
        if not len(below):
            return np.nan
        stop = below[0] + 1
        cycles, env = cycles[:stop], env[:stop]
    if len(cycles) < 20:
        return np.nan
    q = len(cycles) // 4
    d1 = (env[0] - env[q]) / max(cycles[q] - cycles[0], 1e-9)
    d4 = (env[-q - 1] - env[-1]) / max(cycles[-1] - cycles[-q - 1], 1e-9)
    if d1 <= 1e-12:
        return np.nan
    return float(d4 / d1)


def scan(trajectories, soh_stop=None, label=""):
    """trajectoires : dict cell_id -> (cycles, soh_pct)."""
    rows = []
    for cid, (n, y) in trajectories.items():
        dev, n_knee, span = chord_deviation(n, y, soh_stop)
        rows.append(dict(source=label, cell_id=cid,
                         soh_depart=float(np.nanmax(y)),
                         soh_min=float(np.nanmin(y)),
                         profondeur=span, ecart_corde=dev,
                         cycle_knee=n_knee,
                         acceleration=curvature_ratio(n, y, soh_stop),
                         n_points=int(np.isfinite(y).sum())))
    return pd.DataFrame(rows)


def verdict(df, seuil_knee=0.05):
    """Synthese par source : proportion de cellules presentant un coude."""
    g = df.dropna(subset=["ecart_corde"]).groupby("source")
    return pd.DataFrame({
        "n_cellules": g.size(),
        "profondeur_mediane": g["profondeur"].median().round(1),
        "ecart_corde_median": g["ecart_corde"].median().round(4),
        "ecart_corde_p90": g["ecart_corde"].quantile(0.9).round(4),
        "ecart_corde_max": g["ecart_corde"].max().round(4),
        "part_avec_knee": g["ecart_corde"].apply(
            lambda s: float((s > seuil_knee).mean())).round(3),
        "acceleration_mediane": g["acceleration"].median().round(2),
    }).reset_index()


# ==========================================================================
#  Bacon-Watts : modele a deux segments avec transition lisse.
#
#  Pourquoi ajouter cette methode alors que l'ecart a la corde existe deja :
#  sur la cellule de controle b2c12 du jeu Severson, dont le knee est
#  documente entre les cycles 365 et 391 par cinq methodes concordantes,
#  l'ecart a la corde renvoie 327. Il n'est pas faux dans son principe, il est
#  MAL CONDITIONNE ici : sa courbe de deviation vaut 0.336 au cycle 327 et
#  0.297 au cycle 391, soit 12 % d'ecart sur 64 cycles. L'argmax d'un plateau
#  ne localise rien.
#
#  Bacon-Watts ajuste explicitement deux droites reliees par une transition en
#  tangente hyperbolique, et le point de rupture x1 est un PARAMETRE du modele,
#  donc assorti d'un intervalle de confiance :
#
#      y(x) = a0 + a1*(x - x1) + a2*(x - x1)*tanh( (x - x1) / gamma )
#
#  a1 est la pente moyenne des deux segments, a2 la moitie de leur difference,
#  gamma la largeur de transition. C'est la formulation utilisee dans la
#  litterature sur les knees de batteries.
# ==========================================================================

def bacon_watts(cycles, soh_pct, soh_stop=None, gamma0=None, seed=0):
    """Point de rupture Bacon-Watts. Renvoie un dict avec x1 et son ecart-type."""
    from scipy.optimize import curve_fit

    cycles = np.asarray(cycles, float)
    soh = np.asarray(soh_pct, float)
    ok = np.isfinite(cycles) & np.isfinite(soh)
    cycles, soh = cycles[ok], soh[ok]
    if len(cycles) < 30:
        return None
    o = np.argsort(cycles)
    cycles, env = cycles[o], np.minimum.accumulate(soh[o])
    if soh_stop is not None:
        below = np.where(env < soh_stop)[0]
        if not len(below):
            return None
        cycles, env = cycles[:below[0] + 1], env[:below[0] + 1]
    if len(cycles) < 30 or (env[0] - env[-1]) < 3.0:
        return None

    span = cycles[-1] - cycles[0]
    g0 = gamma0 if gamma0 else max(span * 0.02, 1.0)

    def model(x, a0, a1, a2, x1):
        d = x - x1
        return a0 + a1 * d + a2 * d * np.tanh(d / g0)

    pente = (env[-1] - env[0]) / max(span, 1e-9)
    p0 = [float(np.median(env)), pente, pente * 0.5, float(cycles[0] + 0.7 * span)]
    lo = [env.min() - 20, -5.0, -5.0, float(cycles[0] + 0.05 * span)]
    hi = [env.max() + 20, 5.0, 5.0, float(cycles[0] + 0.98 * span)]
    try:
        p, cov = curve_fit(model, cycles, env, p0=p0, bounds=(lo, hi), maxfev=60000)
    except Exception:
        return None
    resid = env - model(cycles, *p)
    sd = float(np.sqrt(np.abs(cov[3, 3]))) if np.all(np.isfinite(cov)) else np.nan
    # pentes des deux segments : a1 -/+ a2
    pente_avant, pente_apres = float(p[1] - p[2]), float(p[1] + p[2])
    return dict(x1=float(p[3]), x1_sd=sd,
                pente_avant=pente_avant, pente_apres=pente_apres,
                rapport_pentes=(pente_apres / pente_avant
                                if abs(pente_avant) > 1e-12 else np.nan),
                soh_au_knee=float(np.interp(p[3], cycles, env)),
                rmse=float(np.sqrt(np.mean(resid ** 2))),
                gamma=g0, n_points=len(cycles))
