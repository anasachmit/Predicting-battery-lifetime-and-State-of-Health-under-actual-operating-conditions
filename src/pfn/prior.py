# -*- coding: utf-8 -*-
r"""Prior du BatteryPFN : distribution sur les LABORATOIRES synthetiques.

Un tirage du prior = un "laboratoire" : une chimie fictive (theta communs), un
plan d'experience (6 conditions T x C), une realisation de bruit, et les six
trajectoires SOH qui en decoulent. Le PFN sera entraine EXCLUSIVEMENT sur ces
tirages ; aucune trajectoire reelle n'entre dans son apprentissage. Les donnees
reelles n'interviennent qu'ici, sous forme de BORNES (module `real_theta`), et
au moment du test de couverture.

--------------------------------------------------------------------------
CE QUE CE PRIOR EST CENSE REGLER
--------------------------------------------------------------------------
Le biais dominant du modele soumis (v3) est la forme supposee de b(T). Mesuree
sur les six cellules cible, l'energie d'activation apparente n'est pas
constante : elle vaut ~0 kJ/mol de 25 a 35 degC, puis 22 a 26 de 35 a 45, puis
32 a 36 de 45 a 55 (chiffres du banc ; la reproduction par ajustement conjoint
dans `real_theta` donne -3.5 / 26.3 / 36.1). Ce n'est pas une loi d'Arrhenius.

Mais aucune forme unique ne tranche : en validation leave-one-cell-out sur les
six cellules, Arrhenius rend 20.4 % de MAE, la quadratique 20.2 %, la lineaire
19.9 %. Trois formes incompatibles, un demi-point d'ecart. Choisir l'une des
trois, c'est asserter une structure que 6 cellules et 4 niveaux de temperature
ne peuvent pas departager.

Le PFN contourne le probleme : il n'en choisit aucune. Il est entraine sur un
MELANGE de structures, et son inference approche la moyenne a posteriori sur ce
melange. Le melange de l'etape A.1 est donc le mecanisme du gain attendu, pas
un detail de reglage.

--------------------------------------------------------------------------
LOI GENERATIVE (L3, identique au modele soumis)
--------------------------------------------------------------------------
Par cellule :

    base(n) = s0 - b * n^0.8
    n_k     = ((s0 - SOH_k) / b)^(1/0.8)
    tau     = tau_frac * n_k
    SOH(n)  = base(n) - g * tau * (exp((n - n_k)/tau) - 1)_+

Carte des conditions, ecrite en ECART a une condition de reference
(T_ref = 40 degC, C_ref = sqrt(0.5)) pour que l'echelle et la forme soient
independantes :

    ln b(T, C) = ln b_ref + f_struct(T) - f_struct(T_ref) + w2 * ln(C / C_ref)

`f_struct` est tiree parmi trois familles (A.1). C'est la seule difference de
fond avec le modele soumis, qui impose f_struct = Arrhenius.

--------------------------------------------------------------------------
DEUX ECARTS ASSUMES PAR RAPPORT AU CAHIER DES CHARGES
--------------------------------------------------------------------------
1. `a`. Le cahier des charges parle d'echelles log-uniformes "(a, b, g)" et
   d'un jitter cellule-a-cellule "sur a, b". L3 n'a pas de parametre `a` : sa
   seule amplitude de perte de base est `b` (le `a` de `laws.L3_user` appartient
   a une autre loi candidate, ecartee). Ajouter un terme en a*sqrt(n) a cote de
   b*n^0.8 serait de surcroit colineaire - `laws.py` documente deja que sqrt(t)
   et n^p ne coexistent jamais dans ce banc. Les echelles log-uniformes sont
   donc (b_ref, g), et le jitter log-normal porte sur les deux amplitudes que
   L3 possede reellement : b et g.
2. `sigma` de mesure. Le bruit point-a-point mesure sur les six cellules cible
   (residu apres filtre median 21 points) vaut 0.015 a 0.353 point de SOH. Le
   U[0.05, 0.4] demande est donc legerement plus bruite que le reel du cote
   silencieux. Conserve tel quel : sur-bruiter est le sens conservateur.

--------------------------------------------------------------------------
PROVENANCE DES BORNES
--------------------------------------------------------------------------
Toutes les constantes REAL_* ci-dessous sont mesurees, pas choisies. Elles
viennent de `results/pfn/theta_real.csv`, produit par `src.pfn.real_theta` sur
`data/unified.parquet` (44 cellules : 6 cible, 18 SNL, 20 Wheeler). Regle
d'elargissement : +/- 50 % de la LARGEUR de la plage reelle de chaque cote,
en echelle logarithmique pour les parametres d'echelle.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

R_GAS = 8.314e-3          # kJ / (mol K)
FADE_P = 0.8              # exposant du baseline officiel, conserve comme REPERE
N_REF = 1000.0            # cycle de reference de l'echelle de perte (voir L_REF)
T_REF_C = 40.0            # milieu du domaine note [25, 55] degC
C_REF = float(np.sqrt(0.5))   # centre geometrique de [0.5, 1.0]
T_DOMAIN = (25.0, 55.0)
C_DOMAIN = (0.5, 1.0)


# ==========================================================================
#  Mesures sur les 44 cellules reelles  (results/pfn/theta_real.csv)
# ==========================================================================
# --- echelle de perte -------------------------------------------------------
# L(1000) = s0 - SOH(1000), en points de SOH. C'est l'echelle retenue a la place
# de `b`, pour deux raisons.
#
# 1. Elle est INDEPENDANTE DE p. Avec base(n) = s0 - b*n^p, `b` et `p` sont
#    couples : a p = 0.8 les cellules cible donnent b dans [0.027, 0.056], a p
#    libre (~0.53) les MEMES cellules donnent b dans [0.20, 0.48]. Tirer b et p
#    independamment engendrerait des trajectoires absurdes. On tire donc
#    L_ref et p, et on pose b = L_ref / N_REF**p.
# 2. Elle se MESURE presque sans modele. Aucune cellule cible n'a de knee actif
#    au cycle 1000 (toutes >= 89.3 % de SOH), donc L(1000) lit directement la
#    loi de base, sans passer par un ajustement non lineaire.
#
# Mesure sur les 6 cellules cible (convention SOH_nom) :
#     25 degC/0.5C  8.57      45 degC/0.5C 10.08
#     25 degC/1C    9.07      45 degC/1C   10.16
#     35 degC/1C    8.51      55 degC/1C   12.79
REAL_L_TARGET = (8.51, 12.79)
REAL_L_REF_40C = 9.28      # interpole a 40 degC, moyenne geometrique par niveau
# Meme grandeur sur les autres sources. Elles NE CALIBRENT PAS l'echelle : la
# doctrine du banc (cf. src/wheeler_prior.py) est que les grandeurs portant une
# echelle ne se transferent pas entre formats. SNL est du 18650 1.1 Ah, Wheeler
# du 18650 a 50 degC. Elles servent de controle de couverture secondaire.
REAL_L_SNL = (2.55, 8.23)
REAL_L_WHEELER = (10.37, 23.34)

# Profil a deux regimes, lu sur L(1000) sans aucun ajustement :
#     Ea 25->35 = -2.7 kJ/mol,  35->45 = 14.1,  45->55 = 20.3
# Meme forme que la mesure sur les `b` ajustes (-3.5 / 26.3 / 36.1) : plate
# puis raide. C'est le fait que le melange de structures doit representer.

# --- exposant de la loi de base ---------------------------------------------
# Ajustements individuels des 44 cellules, exposant LIBRE. Le baseline officiel
# le fige a 0.8 ; la mesure ne soutient pas cette valeur.
#     cible   [0.44, 0.59]  mediane 0.53
#     SNL     [0.50, 0.95]  mediane 0.66
#     Wheeler [0.60, 0.82]  mediane 0.655
REAL_P = (0.439, 0.954)
REAL_P_TARGET = (0.439, 0.588)

# SOH_k, ajustements INDIVIDUELS des 6 cellules cible : 78.2 a 88.9,
# mediane 80.4. L'ajustement conjoint donne 79.56.
REAL_SOHK = (78.17, 88.87)

# g, ajustements individuels des 6 cellules cible (SOH_k libre).
REAL_G = (0.00307, 0.01192)
# tau_frac, memes ajustements : sature en butee haute sur 2 cellules sur 6.
# La saturation est le resultat, pas un echec : le knee reel est plus proche
# d'une RAMPE (limite tau -> infini) que d'une explosion exponentielle.
REAL_TAU_FRAC = (0.074, 20.0)

# s0 des 6 cellules cible, mediane des cycles <= 10, convention SOH_nom.
REAL_S0 = (102.06, 102.90)

# Dispersion cellule-a-cellule de ln b, mesuree sur les replicats SNL a
# conditions strictement identiques (4 cellules a 25 degC/1C, 4 a 35 degC/1C,
# 4 a 25 degC/3C) : 0.026, 0.055, 0.112.
REAL_SIGMA_CELL = (0.0256, 0.1115)

# Bruit point-a-point des 6 cellules cible (residu apres filtre median 21 pts).
REAL_SIGMA_MEAS = (0.015, 0.353)

# Plan reel : n_max, couverture n_points/n_max, profondeur d'arret.
REAL_N_MAX = (1375.0, 5356.0)
REAL_COVERAGE = (0.719, 1.000)
REAL_STOP_SOH = (42.75, 54.45, 65.70, 66.25, 84.78, 93.10)


def widen(lo, hi, frac=0.5, log=False, clip=None):
    """Elargit [lo, hi] de `frac` x sa largeur de chaque cote.

    `log=True` : elargissement en echelle logarithmique, pour un parametre
    d'echelle (une plage de facteur 2 devient une plage de facteur ~4, pas une
    plage decalee).
    """
    if log:
        a, b = np.log(lo), np.log(hi)
        w = b - a
        out = (float(np.exp(a - frac * w)), float(np.exp(b + frac * w)))
    else:
        w = hi - lo
        out = (float(lo - frac * w), float(hi + frac * w))
    if clip is not None:
        out = (max(out[0], clip[0]), min(out[1], clip[1]))
    return out


# ==========================================================================
#  A.1  -  Prior sur la STRUCTURE de b(T)
# ==========================================================================
STRUCTURES = ("arrhenius", "charniere", "quadratique")

STRUCTURE_WEIGHTS = (0.45, 0.45, 0.10)
"""Poids du melange de structures. Justification.

**Arrhenius, 0.45.** C'est le modele physique standard de la degradation
thermiquement activee : celui que suppose le baseline officiel, celui que
suppose la carte de conditions du modele soumis, et celui de la quasi-totalite
de la litterature LFP. Lui donner un poids faible reviendrait a affirmer que
notre mesure sur 6 cellules pese plus lourd que ce corpus. Elle ne le peut pas.

**Charniere, 0.45.** C'est ce que MESURENT nos 6 cellules : energie d'activation
apparente quasi nulle de 25 a 35 degC, puis 22 a 36 kJ/mol au-dela. Un tel
profil est le comportement attendu lorsqu'un second mecanisme (croissance de SEI
acceleree, dissolution du Fe, decomposition d'electrolyte) s'allume au-dessus
d'un seuil, le premier etant deja sature. C'est une hypothese physique, pas une
simple flexibilite de courbe.

**Pourquoi 0.45 / 0.45 et non 0.7 / 0.3 ou l'inverse.** Parce que les donnees ne
tranchent pas : en leave-one-cell-out sur les 6 cellules, Arrhenius rend 20.4 %
de MAE, la quadratique 20.2 %, la lineaire 19.9 %. Un demi-point d'ecart entre
trois formes incompatibles, sur 4 niveaux de temperature. L'EGALITE DES POIDS
EST L'ENONCE "les donnees ne choisissent pas". Toute asymetrie serait une
information qu'on n'a pas.

**Quadratique en 1/T, 0.10.** Ce n'est pas une troisieme hypothese physique :
c'est l'interpolation lisse entre les deux precedentes. Elle coute deux
parametres de forme et couvre les profils intermediaires - transition
progressive plutot que coude franc - qui laisseraient sinon un trou entre les
deux familles. 10 % suffit pour cela ; davantage diluerait le contraste qui
fait tout l'interet du melange.
"""

# Energie d'activation apparente, kJ/mol. Borne haute 45 : la mesure la plus
# raide du banc est 36 kJ/mol (45 -> 55 degC), la borne laisse 25 % de marge.
# Borne basse 0 : b ne decroit jamais avec T (monotonie exigee par la physique
# et par l'auto-controle de plausibilite du modele soumis).
EA_ARRHENIUS = (0.0, 45.0)
EA_HAUTE_CHARNIERE = (15.0, 45.0)   # pente haute de la charniere (cahier des charges)
T_BASCULE = (30.0, 45.0)            # point de bascule, degC (cahier des charges)
EA_QUAD_ENDPOINTS = (0.0, 45.0)     # Ea locale a chaque extremite du domaine


def _x_of(T_degC):
    """1000 / T_K. T croissant <=> x decroissant."""
    return 1000.0 / (np.asarray(T_degC, float) + 273.15)


def _f_arrhenius(T_degC, ea):
    """ln b non normalise, pente constante -ea/R en 1/T."""
    return -(ea / R_GAS) * _x_of(T_degC) / 1000.0


def _f_charniere(T_degC, t_b, ea_haute):
    """Pente basse NULLE sous t_b, pente -ea_haute/R au-dessus."""
    x = _x_of(T_degC)
    x_b = _x_of(t_b)
    # au-dessus de t_b : x < x_b, et la perte augmente
    return (ea_haute / R_GAS) * np.maximum(x_b - x, 0.0) / 1000.0


def _f_quadratique(T_degC, ea_froid, ea_chaud):
    """Quadratique en x = 1000/T, monotone croissante en T par construction.

    La derivee d(ln b)/dx d'une quadratique est LINEAIRE en x. Il suffit donc de
    l'imposer negative aux deux extremites du domaine pour qu'elle le soit
    partout entre les deux. On tire directement ces deux pentes sous forme
    d'energies d'activation locales (>= 0), ce qui rend le tirage interpretable :
    `ea_froid` est l'energie apparente au bord froid (25 degC), `ea_chaud` celle
    au bord chaud (55 degC). Le profil a deux regimes correspond a ea_froid ~ 0
    et ea_chaud grand - en version lissee, sans coude.
    """
    x = _x_of(T_degC)
    x_froid, x_chaud = _x_of(T_DOMAIN[0]), _x_of(T_DOMAIN[1])
    s_froid = -ea_froid / R_GAS / 1000.0     # d ln b / dx au bord froid
    s_chaud = -ea_chaud / R_GAS / 1000.0
    c2 = (s_froid - s_chaud) / (2.0 * (x_froid - x_chaud))
    c1 = s_chaud - 2.0 * c2 * x_chaud
    return c1 * x + c2 * x * x


@dataclass
class StructureB:
    """Structure tiree pour b(T). `log_ratio` la normalise a 0 en T_ref."""
    nom: str
    params: dict

    def f(self, T_degC):
        if self.nom == "arrhenius":
            return _f_arrhenius(T_degC, self.params["ea"])
        if self.nom == "charniere":
            return _f_charniere(T_degC, self.params["t_b"],
                                self.params["ea_haute"])
        if self.nom == "quadratique":
            return _f_quadratique(T_degC, self.params["ea_froid"],
                                  self.params["ea_chaud"])
        raise ValueError(self.nom)

    def log_ratio(self, T_degC):
        """ln b(T) - ln b(T_ref), a C-rate de reference."""
        return self.f(T_degC) - self.f(T_REF_C)

    def ea_apparente(self, T1, T2):
        """Energie d'activation apparente entre deux temperatures, kJ/mol.

        C'est la grandeur diagnostique du banc : celle qui vaut ~0 puis 22 puis
        32 sur les cellules reelles.
        """
        x1, x2 = _x_of(T1) / 1000.0, _x_of(T2) / 1000.0
        return float(-R_GAS * (self.f(T2) - self.f(T1)) / (x2 - x1))


def draw_structure(rng, weights=STRUCTURE_WEIGHTS):
    """Tire une structure de b(T), PUIS ses parametres (A.1)."""
    nom = STRUCTURES[int(rng.choice(len(STRUCTURES), p=np.asarray(weights)))]
    if nom == "arrhenius":
        p = dict(ea=float(rng.uniform(*EA_ARRHENIUS)))
    elif nom == "charniere":
        p = dict(t_b=float(rng.uniform(*T_BASCULE)),
                 ea_haute=float(rng.uniform(*EA_HAUTE_CHARNIERE)))
    else:
        p = dict(ea_froid=float(rng.uniform(*EA_QUAD_ENDPOINTS)),
                 ea_chaud=float(rng.uniform(*EA_QUAD_ENDPOINTS)))
    return StructureB(nom, p)


# ==========================================================================
#  A.2 / A.3 / A.4  -  configuration complete du prior
# ==========================================================================
@dataclass
class PriorConfig:
    """Toutes les constantes du prior, en un seul objet serialisable."""

    # --- A.2 parametres de la loi -----------------------------------------
    s0_mu: float = 102.5
    s0_sd: float = 0.4
    s0_trunc: tuple = (101.0, 104.0)

    soh_k_mu: float = 80.4
    soh_k_sd: float = 2.5
    soh_k_trunc: tuple = (74.0, 88.0)

    # echelles, log-uniformes, calibrees sur le reel puis elargies de 50 %.
    #
    # PASSE DE RECENTRAGE UNIQUE (et definitive). La version precedente tirait
    # `b_ref` = b a 40 degC sur la plage des b OBSERVES toutes temperatures
    # confondues [0.0269, 0.0562]. C'etait une erreur de categorie : cette
    # etendue est surtout la dependance en T, pas l'incertitude d'un labo a
    # l'autre. Elle gonflait la plage ET la decalait vers le haut, d'ou le
    # decentrage constate a l'etape B (rangs medians 56-68 au lieu de 50).
    # Le recentrage se fait sur L(1000) a 40 degC, grandeur lue sans modele, et
    # sur elle seule - jamais sur les rangs de couverture, qui sont le TEST.
    loss_ref_range: tuple = field(
        default_factory=lambda: widen(*REAL_L_TARGET, log=True))
    # Exposant de la loi de base. La regle des +/- 50 % est ecartee ici (3e et
    # derniere exception, avec tau_frac et - implicitement - loss_ref) : elle
    # donnerait [0.18, 1.21], or p > 1 fait doublon avec le terme de knee (la
    # loi de base accelererait toute seule) et p < 0.4 n'a pas de sens
    # physique. Les bornes retenues sont les bornes PHYSIQUES du mecanisme :
    # 0.5 = croissance de SEI limitee par diffusion, 0.8 = baseline officiel,
    # 1.0 = perte proportionnelle au cycle. La mesure [0.44, 0.95] y tient.
    #
    # PLAGE FIGEE le 2026-08-25. Decision prise sur la COUVERTURE seule :
    # U[0.40, 1.00] contient les 5 estimations propres des cellules cible
    # (0.454 a 0.588), la ou U[0.55, 0.95] n'en contiendrait que 2 et
    # U[0.60, 1.00] aucune. Sur l'identifiabilite de SOH_k, les trois plages
    # sont a egalite (err_SOH_k 1.33 / 1.37 / 1.41 pt sous estimateur a p
    # libre) : ce n'est donc pas un argument, et il n'est pas invoque.
    #
    # RECENTRAGE PAR LA DENSITE, 2026-08-26. Le SUPPORT reste [0.40, 1.00] :
    # aucun regime n'est exclu, la marginalisation reste possible partout.
    # Seule la DENSITE change, d'uniforme a normale tronquee N(0.55, 0.15).
    #
    # Pourquoi. Tant que p etait secondaire, un prior uniforme placant les
    # cibles (p mesure 0.45-0.59) dans son premier quart etait acceptable. Il ne
    # l'est plus : le decoupage de la borne C.2a par la VRAIE valeur de p montre
    # que figer l'exposant coute 3.93 points d'APE - davantage que les 2.28
    # points de l'ecart charniere/Arrhenius. p est devenu le levier dominant.
    #
    # Ou cela fait mal : dans le regime TRONQUE, les cycles de debut de vie ne
    # contraignent pas p, donc l'inference retombe sur le prior. Un prior de
    # mediane 0.70 face a une verite de 0.53 injecte un biais systematique dans
    # le run a budget reduit - celui qui porte le bonus de data-efficiency.
    #
    # Effet mesure sur les 5 cellules cible :
    #     uniforme [0.40, 1.00]    -> quantiles  9 a 31 %  (premier quart)
    #     N(0.55, 0.15) tronquee   -> quantiles 12 a 53 %  (coeur)
    # Masse du prior sous p = 0.70 : 50.0 % -> 81.3 %.
    p_range: tuple = (0.40, 1.00)      # SUPPORT, inchange
    p_mu: float = 0.55                 # DENSITE, recentree le 2026-08-26
    p_sd: float = 0.15
    g_range: tuple = field(
        default_factory=lambda: widen(*REAL_G, log=True))
    # SEUL parametre ou la regle des +/- 50 % en log est ecartee, et pourquoi.
    # La plage reelle [0.074, 20] est deja SATUREE en butee haute : elle ne
    # mesure pas une dispersion, elle mesure que 20 etait la borne. L'elargir
    # en log donnerait [0.0045, 330], c'est-a-dire mettre l'essentiel de la
    # masse au-dela de 20 - or au-dela de ~20, g*tau*(exp((n-n_k)/tau) - 1)
    # est numeriquement indiscernable de sa limite lineaire g*(n - n_k) sur
    # tout l'horizon utile. Cette masse serait donc du prior gaspille sur une
    # region ou le modele est degenere. On borne a 30, soit 50 % au-dela de la
    # butee observee, ce qui couvre la limite lineaire sans la sur-representer.
    tau_frac_range: tuple = (0.05, 30.0)

    # dependance en C-rate : centree sur 0 (nos donnees ne montrent aucun effet
    # a cycle egal) mais non degeneree, pour que le reseau apprenne a l'INFERER
    # au lieu de la supposer nulle.
    w2_mu: float = 0.0
    w2_sd: float = 0.15

    # --- A.3 realisme du bruit --------------------------------------------
    sigma_meas_range: tuple = (0.05, 0.40)          # cahier des charges
    sigma_cell_range: tuple = field(
        default_factory=lambda: widen(*REAL_SIGMA_CELL, clip=(0.005, 1.0)))
    s0_cell_sd: float = 0.30      # dispersion de s0 entre cellules d'un meme lab

    # --- A.3bis  ERREUR DE MODELE -----------------------------------------
    # Ce que L3 ne sait PAS representer, et qui n'est pas du bruit de mesure.
    #
    # Mesure qui impose ce terme : sur les 6 cellules cible, l'ajustement L3
    # conjoint laisse un residu de 1.27 point de RMSE - environ six fois le
    # bruit de mesure - et ce residu est quasi parfaitement AUTOCORRELE (0.92 a
    # 1.00 au lag 1, 0.78 a 0.99 au lag 50, jusqu'a 21 points d'amplitude crete
    # a crete). Sans ce terme le generateur produit un residu de 0.25 point,
    # blanc : un PFN entraine dessus apprendrait a inferer dans un monde ou la
    # loi est exacte, et serait surconfiant sur le reel.
    #
    # Forme : somme de modes lisses en u = n / n_fin, avec
    #     phi_k(u) = sin((k - 1/2) * pi * u)
    # qui vaut 0 en u = 0 (le debut de vie sert a mesurer s0, l'erreur ne doit
    # pas s'y loger) et reste libre en fin de trajectoire. Spectre en 1/k :
    # l'essentiel de l'energie est dans le mode 1, d'ou une longueur de
    # correlation de l'ordre de la duree de vie - ce que montre la mesure.
    #
    # Une fraction `rho` des coefficients est PARTAGEE par le laboratoire : la
    # mauvaise specification de L3 est une propriete de la chimie, pas de la
    # cellule. Le reste est propre a chaque cellule.
    # Amplitude calibree par balayage (12 configurations x 40 laboratoires) pour
    # que l'ajustement L3 sur donnees synthetiques laisse le meme residu que sur
    # les 6 cellules reelles : RMSE 1.27 pt, autocorrelation lag-1 0.92-1.00.
    # Retenu : RMSE 1.26, autocorrelation 0.94.
    sigma_modele_range: tuple = (0.5, 3.3)   # amplitude RMS, points de SOH
    n_modes_modele: int = 3
    rho_modele: float = 0.7                  # part partagee dans le laboratoire
    # Exposant du spectre : poids du mode k en 1/k**alpha. alpha grand = toute
    # l'energie dans le mode 1, donc un ecart lisse sur toute la vie - que le
    # terme de knee absorbe entierement. alpha petit = des modes plus courts,
    # que le knee ne peut PAS imiter. C'est le reglage qui arbitre entre
    # realisme du residu et destruction de l'identifiabilite de SOH_k.
    alpha_spectre_modele: float = 1.0

    p_derive: float = 0.15        # probabilite de derive lente de capteur
    derive_amp: tuple = (0.10, 0.50)   # points de SOH sur toute la trajectoire
    derive_exposant: tuple = (0.5, 2.0)

    # --- A.3 troncature et echantillonnage --------------------------------
    # Regle d'arret. La profondeur visee est tiree sur la distribution reelle
    # (2 cellules sur 6 s'arretent au-dessus de 70 %). Le BUDGET de cycles n'est
    # PAS tire independamment : ce serait le budget, et non la profondeur, qui
    # deciderait de l'arret, et la marginale demandee (1/3 au-dessus de 70 %)
    # serait perdue. Le budget est donc adosse au cycle ou la trajectoire
    # atteint sa profondeur visee, avec une marge, plus une petite proportion
    # d'essais reellement coupes avant terme (censure calendaire).
    p_arret_haut: float = 0.28             # regle pour que la marginale > 70 % vaille 1/3
    stop_haut_range: tuple = (78.0, 95.0)  # profondeurs reelles : 84.8 et 93.1
    stop_bas_range: tuple = (40.0, 70.0)   # profondeurs reelles : 42.8 a 66.2
    marge_budget: tuple = (1.0, 1.20)      # depassement au-dela du cycle vise
    p_budget_court: float = 0.10           # essai coupe avant d'atteindre sa cible
    marge_budget_court: tuple = (0.45, 0.90)
    couverture_range: tuple = field(
        default_factory=lambda: widen(*REAL_COVERAGE, clip=(0.3, 1.0)))
    n_trous_range: tuple = (0, 6)          # nb de blocs de cycles manquants
    n_max_absolu: int = 12000              # meme horizon que le modele soumis

    # --- A.4 design des points (T, C) -------------------------------------
    n_cellules: int = 6
    n_niveaux_T: tuple = (3, 4)      # le design reel en a 4
    n_niveaux_C: tuple = (2, 2)      # concentre sur 2 niveaux, comme le reel
    T_jitter: float = 0.5            # tolerance d'enceinte, degC
    espacement_T_min: float = 4.0
    p_grille_pleine: float = 0.15    # rarement : toutes les cases remplies

    def to_dict(self):
        return asdict(self)


DEFAULT = PriorConfig()


# --------------------------------------------------------------- tirages ---
def _trunc_normal(rng, mu, sd, lo, hi):
    for _ in range(100):
        v = float(rng.normal(mu, sd))
        if lo <= v <= hi:
            return v
    return float(np.clip(rng.normal(mu, sd), lo, hi))


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def draw_design(rng, cfg=DEFAULT):
    """A.4 - tire 6 points (T, C) avec la STRUCTURE du design reel.

    Le design reel est une grille 4 T x 2 C dont 6 cases sur 8 sont remplies,
    les C-rates concentres sur deux niveaux, et deux coins vides (35 degC/0.5C
    et 55 degC/0.5C). Un reseau entraine sur des grilles pleines et denses ne
    saurait pas traiter ce cas ; on reproduit donc explicitement :
      - peu de niveaux de temperature (3 ou 4), couvrant les bords du domaine ;
      - exactement deux niveaux de C-rate, l'un pres de 0.5, l'autre pres de 1 ;
      - une grille CREUSE : 6 cellules pour 6 a 8 cases, donc des cases vides ;
      - des effectifs desequilibres entre niveaux de C (le reel : 4 contre 2).
    """
    n_T = int(rng.integers(cfg.n_niveaux_T[0], cfg.n_niveaux_T[1] + 1))
    lo, hi = T_DOMAIN
    # les deux bords du domaine sont presque toujours instrumentes (c'est le
    # cas du design reel) ; les niveaux intermediaires sont libres.
    T = None
    for _ in range(200):
        cand = np.sort(np.concatenate([
            rng.uniform(lo, lo + 3.0, 1),
            rng.uniform(hi - 3.0, hi, 1),
            rng.uniform(lo + 3.0, hi - 3.0, max(n_T - 2, 0))]))
        if np.all(np.diff(cand) >= cfg.espacement_T_min):
            T = cand
            break
    if T is None:
        T = np.linspace(lo, hi, n_T)
    T = T[:n_T]

    n_C = int(rng.integers(cfg.n_niveaux_C[0], cfg.n_niveaux_C[1] + 1))
    C = np.sort(np.concatenate([rng.uniform(0.5, 0.65, 1),
                                rng.uniform(0.85, 1.0, 1)]))[:n_C]

    cases = [(float(t), float(c)) for t in T for c in C]
    k = cfg.n_cellules
    if len(cases) <= k or rng.random() < cfg.p_grille_pleine:
        # grille pleine, ou plus petite que l'effectif : on complete par des
        # replicats, comme le ferait un vrai plan a cellules multiples.
        idx = list(range(len(cases)))
        while len(idx) < k:
            idx.append(int(rng.integers(len(cases))))
        idx = idx[:k]
    else:
        # grille CREUSE et DESEQUILIBREE : on privilegie un niveau de C-rate,
        # ce qui laisse des coins vides sur l'autre.
        c_favori = float(C[int(rng.integers(len(C)))])
        poids = np.array([2.5 if abs(c - c_favori) < 1e-9 else 1.0
                          for _, c in cases])
        poids = poids / poids.sum()
        idx = list(rng.choice(len(cases), size=k, replace=False, p=poids))

    pts = [cases[i] for i in idx]
    T_out = np.array([p[0] for p in pts]) + rng.normal(0.0, cfg.T_jitter, k)
    C_out = np.array([p[1] for p in pts])
    return np.clip(T_out, lo - 1.0, hi + 1.0), C_out


def draw_lab(rng, cfg=DEFAULT):
    """Tire un laboratoire complet : theta communs, design, theta par cellule.

    `generate.simulate_lab` transforme ce dict en trajectoires observees.
    """
    struct = draw_structure(rng)

    s0 = _trunc_normal(rng, cfg.s0_mu, cfg.s0_sd, *cfg.s0_trunc)
    soh_k = _trunc_normal(rng, cfg.soh_k_mu, cfg.soh_k_sd, *cfg.soh_k_trunc)
    loss_ref = _loguniform(rng, *cfg.loss_ref_range)
    p = _trunc_normal(rng, cfg.p_mu, cfg.p_sd, *cfg.p_range)
    g = _loguniform(rng, *cfg.g_range)
    tau_frac = _loguniform(rng, *cfg.tau_frac_range)
    w2 = float(rng.normal(cfg.w2_mu, cfg.w2_sd))

    sigma_meas = float(rng.uniform(*cfg.sigma_meas_range))
    sigma_cell = float(rng.uniform(*cfg.sigma_cell_range))
    sigma_modele = float(rng.uniform(*cfg.sigma_modele_range))

    T, C = draw_design(rng, cfg)
    k = len(T)

    # Echelle de perte par cellule : carte des conditions x jitter log-normal.
    log_L = (np.log(loss_ref) + struct.log_ratio(T) + w2 * np.log(C / C_REF)
             + rng.normal(0.0, sigma_cell, k))
    # b = L / N_REF**p : c'est ici que l'echelle et l'exposant se recombinent.
    b_cell = np.exp(log_L) / N_REF ** p
    g_cell = g * np.exp(rng.normal(0.0, sigma_cell, k))
    s0_cell = s0 + rng.normal(0.0, cfg.s0_cell_sd, k)

    # Erreur de modele : part partagee par le labo + part propre a la cellule.
    K = int(cfg.n_modes_modele)
    rho = float(cfg.rho_modele)
    c_partage = rng.normal(0.0, 1.0, K)
    c_modele = (rho * c_partage[None, :]
                + np.sqrt(max(1.0 - rho ** 2, 0.0)) * rng.normal(0.0, 1.0, (k, K)))

    # regle d'arret par cellule (A.3) : 1/3 des essais s'arretent haut.
    haut = rng.random(k) < cfg.p_arret_haut
    stop_soh = np.where(haut,
                        rng.uniform(*cfg.stop_haut_range, k),
                        rng.uniform(*cfg.stop_bas_range, k))
    court = rng.random(k) < cfg.p_budget_court
    marge = np.where(court,
                     rng.uniform(*cfg.marge_budget_court, k),
                     rng.uniform(*cfg.marge_budget, k))
    couverture = rng.uniform(*cfg.couverture_range, k)
    n_trous = rng.integers(cfg.n_trous_range[0], cfg.n_trous_range[1] + 1, k)

    derive = rng.random(k) < cfg.p_derive
    derive_amp = rng.uniform(*cfg.derive_amp, k) * rng.choice([-1.0, 1.0], k)
    derive_exp = rng.uniform(*cfg.derive_exposant, k)

    return dict(
        structure=struct,
        structure_nom=struct.nom,
        structure_params=dict(struct.params),
        theta=dict(s0=s0, soh_k=soh_k, loss_ref=loss_ref, p=p, g=g,
                   tau_frac=tau_frac, w2=w2, sigma_meas=sigma_meas,
                   sigma_cell=sigma_cell, sigma_modele=sigma_modele),
        # diagnostic : les trois energies apparentes du banc, sur CE tirage
        ea_25_35=struct.ea_apparente(25.0, 35.0),
        ea_35_45=struct.ea_apparente(35.0, 45.0),
        ea_45_55=struct.ea_apparente(45.0, 55.0),
        cellules=[dict(T=float(T[i]), C=float(C[i]), b=float(b_cell[i]),
                       p=p, g=float(g_cell[i]), s0=float(s0_cell[i]),
                       soh_k=soh_k, tau_frac=tau_frac,
                       stop_soh=float(stop_soh[i]), marge_budget=float(marge[i]),
                       couverture=float(couverture[i]), n_trous=int(n_trous[i]),
                       derive=bool(derive[i]), derive_amp=float(derive_amp[i]),
                       derive_exp=float(derive_exp[i]),
                       sigma_modele=sigma_modele,
                       alpha_modele=float(cfg.alpha_spectre_modele),
                       c_modele=c_modele[i].tolist())
                  for i in range(k)],
    )
