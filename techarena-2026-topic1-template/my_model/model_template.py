# -*- coding: utf-8 -*-
# =============================================================================
#   TechArena 2026 - Challenge 1
#   Loi de degradation a knee ancre en SOH, avec regle de garde sur la
#   profondeur des donnees d'entrainement.
#
#   Graine fixe, aucun acces reseau, aucun chemin absolu, aucun poids externe.
# =============================================================================
"""Forme du modele

    base(n) = s0 - b * n^0.8
    n_k     = ( (s0 - SOH_k) / b )^(1/0.8)        cycle ou base(n) atteint SOH_k
    tau     = tau_frac * n_k
    SOH(n)  = base(n) - g * tau * ( exp((n - n_k)/tau) - 1 )_+      puis borne

Carte des conditions :  ln(b) = w0 + w1 * 1000 / T_K

Cinq choix, chacun appuye sur une mesure faite sur les cellules publiees. Ils
sont documentes ici parce que plusieurs contredisent le reglage intuitif.

1. LE KNEE EST ANCRE PAR UN NIVEAU DE SOH, PAS PAR UN NOMBRE DE CYCLES.
   Ajuste librement, n_k se disperse d'un facteur 6 entre cellules et sa carte
   (T, C) ne rend que R2 = 0.31. Le NIVEAU de SOH ou le knee demarre tient dans
   78-85 % sur les six cellules, sans dependance visible a T ni a C, et reste a
   +/- 1 % en validation leave-one-cell-out (dispersion x1.02, contre x7 pour
   l'amplitude et x7 pour la raideur). On estime donc le parametre identifiable.

2. LE KNEE EST PARAMETRE PAR SA PENTE INITIALE g = c / tau.
   Pour z petit, c*(exp(z)-1) ~ (c/tau)*(n - n_k) : seul le rapport agit tant
   que la trajectoire ne s'enfonce pas loin sous n_k. Dans la parametrisation
   (c, tau) la correlation entre les deux parametres de forme vaut 0.9998 - ils
   sont litteralement un seul parametre. En (g, tau) elle tombe a 0.919.

3. REGLE DE GARDE SUR LA PROFONDEUR. Le terme de knee n'est ajuste que si au
   moins deux cellules d'entrainement descendent a moins de GARDE_PTS points de
   SOH au-dessus de SOH_k. Mesure a l'origine de cette regle, sur refit tronque
   des six cellules (erreur mediane sur le cycle d'atteinte de 70 %) :

       donnees vues jusqu'a   avec terme de knee   sans terme de knee
       95 % de SOH                    61.5 %               5.9 %
       90 %                           22.6 %              37.8 %
       85 %                           10.4 %              53.8 %
       80 %                            7.4 %              45.0 %

   Ajouter un knee quand les donnees s'arretent quinze points au-dessus de son
   declenchement est activement nuisible : ses parametres n'y voient aucune
   donnee et prennent la valeur qu'imposent les bornes de l'optimiseur. Cette
   regle rend le modele robuste au re-entrainement a budget reduit (moins de
   cellules, cycles de debut de vie uniquement) que le pipeline officiel
   execute automatiquement.

4. s0 EST MESURE QUAND LES DONNEES SONT PROFONDES, LIBRE QUAND ELLES SONT
   COURTES. Sur trajectoire complete, l'etendue de s0 entre cellules est de
   0.84 point de SOH sans dependance a la temperature (pente -0.015 pt/degC,
   p = 0.16) : le mesurer evite qu'il absorbe une partie du knee. Mais en
   regime court, le laisser libre divise l'erreur par dix (3.8 % contre
   39-48 % sur refit tronque a 95 %), parce que le terme en n^0.8 doit sinon
   absorber seul le transitoire initial et surestime b de 49 a 64 %.

5. PAS DE TERME EN ln(C) DANS LA CARTE DES CONDITIONS.
   A nombre de cycles egal, les paires 0.5C / 1C a 25 et 45 degC different de
   moins de 1.5 point de SOH et l'ecart CHANGE DE SIGNE au cours de la vie. La
   pente en ln(C) n'est pas identifiable sur deux contrastes ; laissee libre
   elle prend une valeur negative - vieillir plus vite a 0.5C qu'a 1C, par
   cycle - qui s'extrapole en aberration vers le coin 55 degC / 0.5C, ou aucune
   cellule n'existe et qui tombe hors de l'enveloppe convexe du plan
   d'experience. L'effet du C-rate est reel mais passe par le TEMPS : a duree
   ecoulee egale, 0.5C conserve 1.6 a 5.6 points de SOH de plus que 1C.

Un auto-controle de plausibilite tourne en fin de `fit()` : monotonie stricte
en temperature, et rapport de duree de vie entre C-rates dans [0.80, 1.25]
(bande derivee des rapports mesures 0.935-1.029 sur les seuils ou les deux
cellules d'une paire ont une verite terrain, elargie pour l'extrapolation
jusqu'a 70 %). En cas d'echec le modele retombe sur sa forme sans knee.
"""
import numpy as np

RANDOM_SEED = 20260101

FADE_P = 0.8            # exposant du baseline officiel
N_MAX = 12000           # longueur de trajectoire renvoyee
SOH_LO, SOH_HI = 0.5, 119.9      # bornes exigees par framework/io.py
REF_CYCLES = 10         # fenetre de mesure de s0
SOH_K_INIT = 80.4       # niveau de declenchement, valeur de depart
GARDE_PTS = 10.0        # regle de garde : ecart max a SOH_k, en points de SOH
GARDE_MIN_CELLS = 2     # nb de cellules profondes exigees
RATIO_C_MIN, RATIO_C_MAX = 0.80, 1.25


# --------------------------------------------------------------- trajectoire
def _nk_from_sohk(s0, b, soh_k):
    return float(np.power(max((s0 - soh_k) / max(b, 1e-12), 1e-9), 1.0 / FADE_P))


def _traj(n, s0, b, soh_k=None, g=0.0, tau_frac=1.0):
    n = np.asarray(n, float)
    soh = s0 - b * np.power(n, FADE_P)
    if soh_k is not None and g > 0.0:
        nk = _nk_from_sohk(s0, b, soh_k)
        tau = max(tau_frac * nk, 1.0)
        z = np.clip((n - nk) / tau, -50.0, 50.0)
        soh = soh - np.where(n > nk, g * tau * np.expm1(z), 0.0)
    return np.clip(soh, SOH_LO, SOH_HI)


class MyModel:

    def __init__(self):
        self.s0_ = 102.47
        self.soh_k_ = SOH_K_INIT
        self.g_ = 0.0
        self.tau_frac_ = 1.0
        self.w_ = [0.0, 0.0]
        self.regime_ = "sans_knee"
        self.plausible_ = True

    # ------------------------------------------------------------------ fit
    def fit(self, cells):
        np.random.seed(RANDOM_SEED)
        from scipy.optimize import least_squares

        data = []
        for cell in cells:
            d = cell.soh.dropna()
            n = d["cycle_number"].to_numpy(float)
            y = d["soh_percent"].to_numpy(float)
            if len(n) < 20:
                continue
            early = y[n <= REF_CYCLES]
            s0 = float(np.median(early)) if early.size else float(np.median(y[:10]))
            data.append(dict(T=float(cell.temperature_degC), C=float(cell.c_rate),
                             n=n, y=y, s0=s0,
                             soh_min=float(np.minimum.accumulate(y).min())))
        if not data:
            raise RuntimeError("aucune cellule exploitable")

        # --- regle de garde : les donnees approchent-elles le declenchement ?
        profondes = [d for d in data
                     if d["soh_min"] - SOH_K_INIT <= GARDE_PTS]
        self.regime_ = ("avec_knee" if len(profondes) >= GARDE_MIN_CELLS
                        else "sans_knee")
        self.s0_ = float(np.mean([d["s0"] for d in data]))

        m = len(data)
        b0 = [max(d["s0"] - d["y"].min(), 1.0) / d["n"].max() ** FADE_P
              for d in data]

        if self.regime_ == "avec_knee":
            # s0 mesure par cellule ; (SOH_k, g, tau_frac) communs ; un b par cellule
            p0 = np.array([SOH_K_INIT, 0.005, 0.25] + b0, float)
            lo = np.array([60.0, 0.0, 0.02] + [1e-7] * m, float)
            hi = np.array([98.0, 1.0, 5.0] + [5.0] * m, float)

            def resid(p):
                out = []
                for i, d in enumerate(data):
                    pred = _traj(d["n"], d["s0"], p[3 + i], p[0], p[1], p[2])
                    out.append((pred - d["y"]) / np.sqrt(len(d["n"])))
                return np.concatenate(out)
        else:
            # regime court : pas de knee, et s0 LIBRE en commun (choix 4)
            p0 = np.array([self.s0_] + b0, float)
            lo = np.array([95.0] + [1e-7] * m, float)
            hi = np.array([112.0] + [5.0] * m, float)

            def resid(p):
                out = []
                for i, d in enumerate(data):
                    pred = _traj(d["n"], p[0], p[1 + i])
                    out.append((pred - d["y"]) / np.sqrt(len(d["n"])))
                return np.concatenate(out)

        rng = np.random.default_rng(RANDOM_SEED)
        best = None
        for k in range(6):
            st = p0 if k == 0 else np.clip(p0 * rng.uniform(0.5, 2.0, len(p0)),
                                           lo, hi)
            try:
                r = least_squares(resid, st, bounds=(lo, hi), max_nfev=20000)
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
        if best is None:
            raise RuntimeError("ajustement conjoint echoue")

        p = best.x
        if self.regime_ == "avec_knee":
            self.soh_k_, self.g_, self.tau_frac_ = (float(p[0]), float(p[1]),
                                                    float(p[2]))
            b_cell = np.asarray(p[3:], float)
        else:
            self.s0_ = float(p[0])
            self.g_ = 0.0
            b_cell = np.asarray(p[1:], float)

        # --- carte des conditions : Arrhenius seul (choix 5)
        Tk = np.array([d["T"] for d in data]) + 273.15
        X = np.column_stack([np.ones(m), 1000.0 / Tk])
        if m >= 2 and len(np.unique(Tk)) >= 2:
            w, *_ = np.linalg.lstsq(X, np.log(b_cell), rcond=None)
            self.w_ = [float(w[0]), float(w[1])]
        else:
            self.w_ = [float(np.log(b_cell.mean())), 0.0]

        self._auto_controle()
        print(f"  regime={self.regime_}  s0={self.s0_:.2f}  SOH_k={self.soh_k_:.2f}"
              f"  g={self.g_:.5f}  tau_frac={self.tau_frac_:.3f}"
              f"  w={np.round(self.w_, 4)}  plausible={self.plausible_}")
        return self

    # --------------------------------------------------------- auto-controle
    def _auto_controle(self):
        """Monotonie stricte en T et rapport entre C-rates dans l'enveloppe.

        En cas d'echec on retombe sur la forme sans knee, qui ne peut pas
        produire d'inversion.
        """
        T = np.arange(25.0, 55.01, 2.5)
        n70 = np.array([self._n_at(self.predict_soh(t, 1.0), 70.0) for t in T])
        ok = np.all(np.isfinite(n70)) and np.all(np.diff(n70) <= 1e-6)
        if ok:
            r = np.array([self._n_at(self.predict_soh(t, 0.5), 70.0) /
                          max(self._n_at(self.predict_soh(t, 1.0), 70.0), 1e-9)
                          for t in T])
            ok = bool(np.all((r >= RATIO_C_MIN) & (r <= RATIO_C_MAX)))
        self.plausible_ = bool(ok)
        if not ok and self.g_ > 0.0:
            self.g_ = 0.0
            self.regime_ = "sans_knee (repli apres auto-controle)"

    @staticmethod
    def _n_at(traj, thr):
        below = np.where(np.asarray(traj, float) < thr)[0]
        if not below.size or below[0] == 0:
            return float(N_MAX)
        j = int(below[0])
        return float(np.interp(thr, [traj[j], traj[j - 1]], [j + 1, j]))

    # -------------------------------------------------------------- predict
    def predict_soh(self, temperature_degC, c_rate):
        b = float(np.exp(self.w_[0]
                         + self.w_[1] * 1000.0 / (float(temperature_degC) + 273.15)))
        n = np.arange(1, N_MAX + 1, dtype=float)
        return _traj(n, self.s0_, b, self.soh_k_, self.g_, self.tau_frac_)
