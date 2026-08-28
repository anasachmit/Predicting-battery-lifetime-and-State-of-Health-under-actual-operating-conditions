"""Metriques d'evaluation.

ATTENTION - la metrique officielle du challenge n'est PAS publiee. Ce que le
code fourni permet d'etablir avec certitude :
  * l'atome de l'erreur est errSOH = SOH_model - SOH_true, en POINTS de SOH,
    dans la convention nominale (Qdis / 102 Ah), cf. framework/io.py ;
  * les lignes evaluees sont des triplets (Cycle, T, C-rate) ;
  * le score est composite et rapporte au baseline `model_example.py` REFITE
    dans les memes conditions (baseline = 1.0, plus bas = meilleur).
Tout le reste (ponderation, grille de cycles, agregation) est inconnu. Les
fonctions ci-dessous en donnent un PROXY explicite, jamais presente comme la
metrique officielle. On rapporte systematiquement deux grilles de cycles,
parce que le choix de grille change le classement :
  * `uniforme_cycle` : un point par cycle -> pondere la longue zone plate ;
  * `uniforme_soh`   : points equirepartis en SOH -> pondere la fin de vie.
"""
import numpy as np


def _interp_traj(traj, cycles):
    """traj = SOH% pour cycles 1..len(traj) ; renvoie les valeurs demandees."""
    traj = np.asarray(traj, float)
    idx = np.clip(np.asarray(cycles, int) - 1, 0, len(traj) - 1)
    return traj[idx]


def err_soh(traj_pred, cycles_true, soh_true_pct):
    """errSOH en points de SOH, definition de framework/io.py."""
    return _interp_traj(traj_pred, cycles_true) - np.asarray(soh_true_pct, float)


def cycle_at(traj, thr_pct=70.0):
    """Premier cycle ou la trajectoire passe sous le seuil (interpole)."""
    traj = np.asarray(traj, float)
    below = np.where(traj < thr_pct)[0]
    if not len(below) or below[0] == 0:
        return np.nan
    j = below[0]
    return float(np.interp(thr_pct, [traj[j], traj[j - 1]], [j + 1, j]))


def soh_uniform_grid(cycles, soh_pct, n=60):
    """Sous-echantillonnage equireparti en SOH (pondere la fin de vie)."""
    cycles = np.asarray(cycles, float)
    soh = np.asarray(soh_pct, float)
    ok = np.isfinite(soh)
    cycles, soh = cycles[ok], soh[ok]
    if len(soh) < n:
        return cycles.astype(int), soh
    targets = np.linspace(soh.max(), soh.min(), n)
    env = np.minimum.accumulate(soh)
    idx = np.unique([int(np.argmin(np.abs(env - t))) for t in targets])
    return cycles[idx].astype(int), soh[idx]


def evaluate_cell(traj_pred, cycles_true, soh_true_pct, thr=70.0):
    """Toutes les metriques pour une cellule tenue a l'ecart."""
    cycles_true = np.asarray(cycles_true)
    soh_true_pct = np.asarray(soh_true_pct, float)
    ok = np.isfinite(soh_true_pct)
    cyc, soh = cycles_true[ok], soh_true_pct[ok]

    e_all = err_soh(traj_pred, cyc, soh)
    cyc_s, soh_s = soh_uniform_grid(cyc, soh)
    e_soh = err_soh(traj_pred, cyc_s, soh_s)

    low = soh < 80.0
    e_low = e_all[low] if low.sum() else np.array([np.nan])

    n70_true = cycle_at(np.interp(np.arange(1, int(cyc.max()) + 1), cyc, soh), thr)
    n70_pred = cycle_at(traj_pred, thr)

    traj = np.asarray(traj_pred, float)[:int(cyc.max())]
    viol = float(np.mean(np.diff(traj) > 1e-9)) if len(traj) > 1 else np.nan

    return dict(
        rmse_uniforme_cycle=float(np.sqrt(np.mean(e_all ** 2))),
        mae_uniforme_cycle=float(np.mean(np.abs(e_all))),
        rmse_uniforme_soh=float(np.sqrt(np.mean(e_soh ** 2))),
        rmse_sous_80=float(np.sqrt(np.mean(e_low ** 2))),
        n_points_sous_80=int(low.sum()),
        n70_true=n70_true, n70_pred=n70_pred,
        eol_ape=(abs(n70_pred - n70_true) / n70_true * 100
                 if np.isfinite(n70_true) and np.isfinite(n70_pred) else np.nan),
        eol_observe=bool(np.isfinite(n70_true)),
        taux_violation_monotonie=viol,
    )


def relative_to_baseline(m_model, m_base, keys=("rmse_uniforme_cycle",
                                                "rmse_uniforme_soh",
                                                "rmse_sous_80", "eol_ape")):
    """Score relatif au baseline officiel refite : baseline = 1.0."""
    out = {}
    for k in keys:
        a, b = m_model.get(k), m_base.get(k)
        out[f"rel_{k}"] = (float(a / b) if (a is not None and b not in (None, 0)
                                            and np.isfinite(a) and np.isfinite(b))
                           else np.nan)
    return out


# ==========================================================================
#  Metriques par zone et par seuil.
#
#  Motivation : le RMSE global et l'APE sur n@70 sont des scalaires. Deux
#  modeles de meme RMSE peuvent echouer tres differemment - derive uniforme
#  contre erreur concentree sur le knee - et le second est bien plus grave.
#  On decoupe donc l'erreur par zone de SOH et on regarde le franchissement a
#  plusieurs profondeurs, pas seulement 70 %.
# ==========================================================================

# Frontieres de zones, en % de SOH sur la trajectoire VRAIE. La borne basse de
# la zone pre-knee est le niveau de declenchement du knee estime conjointement
# sur les 6 cellules cible (SOH_k = 80.4 %), arrondi.
ZONE_BOUNDS = {"quasi_lineaire": (90.0, 200.0),
               "pre_knee": (80.0, 90.0),
               "post_knee": (-1.0, 80.0)}

SEUILS = (90.0, 85.0, 80.0, 75.0, 70.0)


def knee_onset(cycles, soh_pct, min_span=12.0, min_prominence=0.03):
    """Cycle de declenchement du knee sur une courbe observee.

    Methode : ecart maximal a la corde (kneedle). Apres normalisation des deux
    axes dans [0, 1], on trace la corde joignant le premier et le dernier point
    et on retient l'abscisse ou la courbe s'en ecarte le plus vers le haut.
    C'est la ou la trajectoire cesse d'etre plate et bascule.

    Deux definitions ont ete ecartees avant celle-ci, et pour des raisons
    mesurees, pas esthetiques :

    * minimum de la derivee seconde -> renvoie systematiquement le PREMIER
      cycle, parce qu'une loi puissance s0 - a*n^p (p < 1) a sa courbure
      maximale en n -> 0 ;
    * argmax de dSOH/dn (retournement de la vitesse de perte) -> correct sur
      courbe propre, mais avec 0.3 point de bruit de mesure il place le knee a
      1877 cycles au lieu de 3500 sur un cas synthetique.

    Le critere geometrique n'utilise aucune derivee et reste stable jusqu'a
    0.6 point de bruit. Renvoie NaN si la courbe descend de moins de
    `min_span` points de SOH, ou si l'ecart a la corde reste sous
    `min_prominence` (courbe sans knee : une loi puissance pure y est
    quasi-confondue avec sa corde).
    """
    cycles = np.asarray(cycles, float)
    soh = np.asarray(soh_pct, float)
    ok = np.isfinite(soh) & np.isfinite(cycles)
    cycles, soh = cycles[ok], soh[ok]
    if len(cycles) < 30 or (soh.max() - soh.min()) < min_span:
        return np.nan
    order = np.argsort(cycles)
    cycles, soh = cycles[order], soh[order]
    env = np.minimum.accumulate(soh)

    x = (cycles - cycles[0]) / max(cycles[-1] - cycles[0], 1e-9)
    span = env[0] - env[-1]
    if span <= 0:
        return np.nan
    y = (env - env[-1]) / span                    # 1 au depart, 0 a la fin
    corde = 1.0 - x
    d = y - corde
    i = int(np.argmax(d))
    if d[i] < min_prominence or i <= 2 or i >= len(d) - 3:
        return np.nan
    return float(cycles[i])


def zone_errors(traj_pred, cycles_true, soh_true_pct, bounds=None):
    """RMSE et erreur signee mediane, par zone de SOH vrai."""
    bounds = bounds or ZONE_BOUNDS
    cycles_true = np.asarray(cycles_true)
    soh = np.asarray(soh_true_pct, float)
    ok = np.isfinite(soh)
    cyc, soh = cycles_true[ok], soh[ok]
    e = err_soh(traj_pred, cyc, soh)
    out = {}
    for name, (lo, hi) in bounds.items():
        m = (soh > lo) & (soh <= hi)
        if m.sum() < 3:
            out[f"rmse_{name}"] = np.nan
            out[f"biais_median_{name}"] = np.nan
            out[f"n_pts_{name}"] = int(m.sum())
            continue
        out[f"rmse_{name}"] = float(np.sqrt(np.mean(e[m] ** 2)))
        out[f"biais_median_{name}"] = float(np.median(e[m]))
        out[f"n_pts_{name}"] = int(m.sum())
    return out


def threshold_errors(traj_pred, cycles_true, soh_true_pct, seuils=SEUILS):
    """Erreur absolue sur le cycle de franchissement, a plusieurs profondeurs.

    Donne une COURBE d'erreur en fonction de la profondeur de prediction au
    lieu d'un point unique a 70 %.
    """
    cycles_true = np.asarray(cycles_true, float)
    soh = np.asarray(soh_true_pct, float)
    ok = np.isfinite(soh)
    cyc, soh = cycles_true[ok], soh[ok]
    env = np.minimum.accumulate(soh)
    dense = np.interp(np.arange(1, int(cyc.max()) + 1), cyc, env)
    out = {}
    for s in seuils:
        n_true = cycle_at(dense, s)
        n_pred = cycle_at(traj_pred, s)
        out[f"n_at_{int(s)}_vrai"] = n_true
        out[f"n_at_{int(s)}_pred"] = n_pred
        out[f"err_abs_n_at_{int(s)}"] = (abs(n_pred - n_true)
                                         if np.isfinite(n_true) and np.isfinite(n_pred)
                                         else np.nan)
        out[f"ape_n_at_{int(s)}"] = (abs(n_pred - n_true) / n_true * 100
                                     if np.isfinite(n_true) and np.isfinite(n_pred)
                                     and n_true > 0 else np.nan)
    return out


def knee_errors(traj_pred, cycles_true, soh_true_pct, soh_k=80.4):
    """Erreur sur le declenchement du knee, sous DEUX definitions.

    Le "cycle de declenchement du knee" n'est pas une grandeur bien posee : sur
    une meme trajectoire simulee a knee connu, les trois criteres testes
    donnent des reponses distantes de plus de 1000 cycles (minimum de la
    derivee seconde -> premier cycle ; retournement de dSOH/dn -> biais precoce
    et instable au bruit ; ecart maximal a la corde -> stable mais biais tardif
    d'environ 1200 cycles, et aveugle si le knee arrive pres de la fin du
    releve). On rapporte donc les deux grandeurs reproductibles, nommees pour
    ce qu'elles sont, plutot qu'un chiffre unique faussement autoritaire.

    * `n_knee_geom` : point de knee geometrique (ecart maximal a la corde).
      Sans modele, robuste au bruit, mais tardif par construction : il marque
      le creux du coude, pas son debut.
    * `n_knee_soh`  : cycle de franchissement du niveau `soh_k`, c'est-a-dire
      le declenchement du knee AU SENS DU MODELE, qui ancre son knee sur un
      niveau de SOH (80.4 % estime conjointement sur les 6 cellules cible).
      Directement comparable entre verite et prediction.
    """
    cycles_true = np.asarray(cycles_true, float)
    soh = np.asarray(soh_true_pct, float)
    ok = np.isfinite(soh)
    cyc, s = cycles_true[ok], soh[ok]
    nmax = int(np.nanmax(cyc))
    traj = np.asarray(traj_pred, float)

    g_true = knee_onset(cyc, s)
    g_pred = knee_onset(np.arange(1, nmax + 1), traj[:nmax])

    dense = np.interp(np.arange(1, nmax + 1), cyc, np.minimum.accumulate(s))
    k_true = cycle_at(dense, soh_k)
    k_pred = cycle_at(traj, soh_k)

    def _err(a, b):
        return (abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan)

    def _ape(a, b):
        return (abs(a - b) / b * 100 if np.isfinite(a) and np.isfinite(b)
                and b > 0 else np.nan)

    return dict(n_knee_geom_vrai=g_true, n_knee_geom_pred=g_pred,
                err_abs_n_knee_geom=_err(g_pred, g_true),
                ape_n_knee_geom=_ape(g_pred, g_true),
                n_knee_soh_vrai=k_true, n_knee_soh_pred=k_pred,
                err_abs_n_knee_soh=_err(k_pred, k_true),
                ape_n_knee_soh=_ape(k_pred, k_true))


def evaluate_cell_full(traj_pred, cycles_true, soh_true_pct, thr=70.0):
    """evaluate_cell + zones + seuils multiples + knee."""
    out = evaluate_cell(traj_pred, cycles_true, soh_true_pct, thr)
    out.update(zone_errors(traj_pred, cycles_true, soh_true_pct))
    out.update(threshold_errors(traj_pred, cycles_true, soh_true_pct))
    out.update(knee_errors(traj_pred, cycles_true, soh_true_pct))
    return out


# ==========================================================================
#  Restriction au perimetre de notation : SOH >= 70 %.
#
#  Le challenge s'arrete a 70 % de SOH, mais certaines cellules cible
#  descendent bien plus bas (45 degC/0.5C va jusqu'a 42.8 %). Une part
#  importante du RMSE post-knee porte donc sur des points HORS perimetre.
#  Toutes les metriques sont recalculables avec ce plancher ; les deux
#  versions sont rapportees cote a cote.
# ==========================================================================

SOH_FLOOR = 70.0

ZONE_BOUNDS_NOTE = {"quasi_lineaire": (90.0, 200.0),
                    "pre_knee": (80.0, 90.0),
                    "post_knee_note": (70.0, 80.0)}


def _apply_floor(cycles_true, soh_true_pct, soh_floor):
    """Ne garde que les points dont le SOH VRAI est au-dessus du plancher."""
    cyc = np.asarray(cycles_true, float)
    soh = np.asarray(soh_true_pct, float)
    ok = np.isfinite(soh)
    cyc, soh = cyc[ok], soh[ok]
    if soh_floor is None:
        return cyc, soh
    # on coupe sur l'enveloppe monotone : une fois passe sous le plancher on
    # n'y remonte pas, meme si un point bruite repasse au-dessus.
    env = np.minimum.accumulate(soh)
    keep = env >= soh_floor
    return cyc[keep], soh[keep]


def evaluate_cell_scoped(traj_pred, cycles_true, soh_true_pct,
                         soh_floor=SOH_FLOOR, thr=70.0):
    """Toutes les metriques, restreintes aux points SOH_vrai >= `soh_floor`.

    `soh_floor=None` redonne la version non restreinte.
    """
    cyc, soh = _apply_floor(cycles_true, soh_true_pct, soh_floor)
    if len(cyc) < 5:
        return {}
    bounds = ZONE_BOUNDS_NOTE if soh_floor is not None else ZONE_BOUNDS
    out = evaluate_cell(traj_pred, cyc, soh, thr)
    out.update(zone_errors(traj_pred, cyc, soh, bounds=bounds))
    out.update(threshold_errors(traj_pred, cycles_true, soh_true_pct))
    out.update(knee_errors(traj_pred, cycles_true, soh_true_pct))
    # Le cycle d'atteinte du seuil se mesure TOUJOURS sur la verite complete :
    # sous le plancher, la trajectoire tronquee ne descend par construction
    # jamais jusqu'au seuil, et `evaluate_cell` renverrait NaN.
    plein = evaluate_cell(traj_pred, *_apply_floor(cycles_true, soh_true_pct,
                                                   None), thr)
    for k in ("n70_true", "n70_pred", "eol_ape", "eol_observe"):
        out[k] = plein[k]
    out["n_points_retenus"] = int(len(cyc))
    out["soh_min_retenu"] = float(np.min(soh))
    return out


def compare_scopes(traj_pred, cycles_true, soh_true_pct, keys=None):
    """Metriques avec et sans le plancher de notation, cote a cote."""
    keys = keys or ("rmse_uniforme_cycle", "rmse_uniforme_soh",
                    "rmse_sous_80", "eol_ape")
    note = evaluate_cell_scoped(traj_pred, cycles_true, soh_true_pct,
                                soh_floor=SOH_FLOOR)
    tout = evaluate_cell_scoped(traj_pred, cycles_true, soh_true_pct,
                                soh_floor=None)
    return {**{f"{k}__note": note.get(k, np.nan) for k in keys},
            **{f"{k}__tout": tout.get(k, np.nan) for k in keys}}
