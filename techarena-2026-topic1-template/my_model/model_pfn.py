# -*- coding: utf-8 -*-
# =============================================================================
#   TechArena 2026 - Challenge 1
#   BatteryPFN : reseau pre-entraine sur trajectoires synthetiques, avec repli
#   automatique sur le modele parametrique L3 si PyTorch est indisponible.
#
#   Graine fixe, aucun acces reseau, aucun chemin absolu, poids embarques.
# =============================================================================
"""BatteryPFN - inference bayesienne amortie sur la loi de degradation.

Le principe
-----------
Le reseau est pre-entraine EXCLUSIVEMENT sur des trajectoires synthetiques
tirees d'un prior explicite ; il n'a jamais vu de cellule reelle pendant son
apprentissage. A l'inference, il recoit les cellules d'entrainement comme
CONTEXTE et rend, en un seul passage avant, la trajectoire d'une condition
(T, C-rate) qu'il n'a pas vue. C'est ce qu'un ajustement par moindres carres
obtient par optimisation, mais amorti dans les poids.

L'interet ici est direct : les six cellules disponibles ne permettent de
departager ni la forme de b(T) (Arrhenius 20.4 %, quadratique 20.2 %, lineaire
19.9 % de MAE en leave-one-cell-out) ni l'exposant de la loi de base. Un
estimateur classique doit CHOISIR ; le reseau, entraine sur un melange, MOYENNE
sur les valeurs plausibles.

Pourquoi ce modele plutot que le L3 parametrique
------------------------------------------------
Ce n'est pas l'avantage moyen qui a decide, c'est l'absence d'effondrement dans
le regime a budget reduit - celui que le pipeline officiel execute
automatiquement et qui porte le bonus de data-efficiency. Mesure en
leave-one-cell-out sur les six cellules reelles, RMSE pire cas sur le perimetre
note (SOH >= 70 %) :

    regime                      PFN     L3      baseline
    complet / 5 cellules       4.64    12.40      7.20
    95 % / 5 cellules          7.03    14.71      9.57
    90 % / 5 cellules          4.86    66.38      7.80
    85 % / 5 cellules          5.04    56.32      7.98
    complet / 3 cellules       5.43    33.27     10.43
    95 % / 3 cellules          7.00    15.98     13.02

Le PFN reste borne a 7.03 dans les six regimes et n'a aucun fold au-dela de
100 % d'erreur relative sur le cycle de fin de vie, la ou L3 en a trois.

REPLI AUTOMATIQUE
-----------------
Si PyTorch est indisponible, si les poids ne se chargent pas, ou si le passage
avant echoue pour quelque raison que ce soit, le modele retombe SILENCIEUSEMENT
sur `model_template.MyModel` - c'est-a-dire exactement le modele parametrique
L3, deja valide de bout en bout. Le mode actif est journalise.

Le repli transforme un pari "tout ou rien" en "PFN si possible, L3 sinon".
"""
import os

import numpy as np

RANDOM_SEED = 20260101
N_MAX = 12000
SOH_LO, SOH_HI = 0.5, 119.9

# Normalisation des entrees - FIGEE, identique a l'entrainement.
T_MOY, T_ECH = 40.0, 15.0
LNC_MOY, LNC_ECH = -0.35, 0.35
LNN_MOY, LNN_ECH = 4.9, 2.1
SOH_MOY, SOH_ECH = 85.0, 15.0

N_CTX = 32                     # points de contexte par cellule
FICHIER_POIDS = "pfn_poids.pt"

# Grille sur laquelle le reseau est interroge, puis interpolee sur 1..N_MAX.
GRILLE = np.unique(np.round(np.geomspace(1.0, float(N_MAX), 96))).astype(float)


def _normalise(T, C, n, soh=None):
    z = [(np.asarray(T, np.float32) - T_MOY) / T_ECH,
         (np.log(np.asarray(C, np.float32)) - LNC_MOY) / LNC_ECH,
         (np.log(np.maximum(np.asarray(n, np.float32), 1.0)) - LNN_MOY) / LNN_ECH]
    if soh is not None:
        z.append((np.asarray(soh, np.float32) - SOH_MOY) / SOH_ECH)
    return np.stack(z, axis=-1)


# =============================================================================
#  Architecture - reproduite ici a l'identique pour que `my_model/` soit
#  autonome : la soumission ne peut importer aucun module du depot de recherche.
# =============================================================================
def _construit_reseau(torch, nn, F, quantiles):

    class Bloc(nn.Module):
        def __init__(self, d, n_tetes, mult_ff=4):
            super().__init__()
            self.n1 = nn.LayerNorm(d)
            self.att = nn.MultiheadAttention(d, n_tetes, batch_first=True)
            self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, mult_ff * d), nn.GELU(),
                                    nn.Linear(mult_ff * d, d))

        def forward(self, x):
            h = self.n1(x)
            a, _ = self.att(h, h, h, need_weights=False)
            x = x + a
            return x + self.ff(self.n2(x))

    class Reseau(nn.Module):
        """4 couches, d_model 256. Aucun encodage positionnel : le contexte est
        un ENSEMBLE non ordonne, permuter deux points ne doit rien changer."""

        def __init__(self, d_model=256, n_couches=4, n_tetes=8, mult_ff=4):
            super().__init__()
            self.quantiles = tuple(quantiles)
            nq = len(self.quantiles)
            # Encodeurs SEPARES : un token de contexte porte 4 variables, un
            # token de requete 3. Les confondre ferait lire "SOH absent" comme
            # la valeur 85 %.
            self.enc_ctx = nn.Sequential(nn.Linear(4, d_model), nn.GELU(),
                                         nn.Linear(d_model, d_model))
            self.enc_req = nn.Sequential(nn.Linear(3, d_model), nn.GELU(),
                                         nn.Linear(d_model, d_model))
            self.marqueur = nn.Parameter(torch.zeros(2, d_model))
            self.blocs = nn.ModuleList([Bloc(d_model, n_tetes, mult_ff)
                                        for _ in range(n_couches)])
            self.norme = nn.LayerNorm(d_model)
            self.tete = nn.Linear(d_model, nq)

        def forward(self, x_ctx, x_req):
            hc = self.enc_ctx(x_ctx) + self.marqueur[0]
            hq = self.enc_req(x_req) + self.marqueur[1]
            h = torch.cat([hc, hq], dim=1)
            for b in self.blocs:
                h = b(h)
            h = self.norme(h[:, x_ctx.shape[1]:])
            brut = self.tete(h)
            base = brut[..., :1]
            # quantiles croissants par construction : un croisement rendrait
            # l'incertitude ininterpretable
            incr = F.softplus(brut[..., 1:])
            return torch.cat([base, base + torch.cumsum(incr, dim=-1)], dim=-1)

    return Reseau


class BatteryPFN:
    """Interface de soumission. `fit` memorise le contexte, `predict_soh` infere."""

    def __init__(self):
        self.contexte = []          # [(T, C, cycles, soh)] des cellules vues
        self.mode = "non_ajuste"
        self._reseau = None
        self._torch = None
        self._replis = None         # instance L3 de secours

    # ------------------------------------------------------------ chargement
    def _charge_reseau(self):
        """Tente PyTorch + poids. Renvoie True si le mode PFN est disponible."""
        if self._reseau is not None:
            return True
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            torch.manual_seed(RANDOM_SEED)
            torch.set_grad_enabled(False)

            chemin = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  FICHIER_POIDS)
            paquet = torch.load(chemin, map_location="cpu", weights_only=False)
            quantiles = tuple(paquet.get("quantiles",
                                         (0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95)))
            Reseau = _construit_reseau(torch, nn, F, quantiles)
            net = Reseau()
            net.load_state_dict(paquet["state_dict"])
            net.eval()
            self._torch, self._reseau = torch, net
            return True
        except Exception as exc:                       # noqa: BLE001
            print(f"  [BatteryPFN] mode PFN indisponible ({type(exc).__name__}: "
                  f"{exc}) - repli sur le modele parametrique L3")
            self._reseau = None
            return False

    def _charge_repli(self, cells):
        from .model_template import MyModel
        self._replis = MyModel()
        self._replis.fit(cells)

    # ------------------------------------------------------------------ fit
    def fit(self, cells):
        """Memorise le contexte. Le reseau n'est PAS reentraine ici.

        Les poids viennent du pre-entrainement hors ligne sur le prior ; `fit`
        ne fait que preparer ce que le reseau lira comme contexte. C'est ce qui
        rend le modele robuste au run a budget reduit : moins de cellules ou des
        cellules plus courtes changent le contexte, pas les poids.
        """
        np.random.seed(RANDOM_SEED)
        self.contexte = []
        for cell in cells:
            d = cell.soh.dropna()
            n = d["cycle_number"].to_numpy(float)
            y = d["soh_percent"].to_numpy(float)
            if len(n) < 20:
                continue
            if len(n) > N_CTX:
                # grille GEOMETRIQUE : la courbure qui porte l'information est
                # concentree dans les premiers cycles
                idx = np.unique(np.round(
                    np.geomspace(1, len(n), N_CTX)).astype(int) - 1)
                idx = np.clip(idx, 0, len(n) - 1)
                idx[-1] = len(n) - 1
                n, y = n[idx], y[idx]
            self.contexte.append((float(cell.temperature_degC),
                                  float(cell.c_rate), n, y))

        if not self.contexte:
            raise RuntimeError("aucune cellule exploitable")

        # Le repli est ajuste SYSTEMATIQUEMENT, meme si le PFN est disponible :
        # il ne coute que quelques secondes et garantit qu'aucune defaillance a
        # la prediction ne puisse laisser le modele sans reponse.
        try:
            self._charge_repli(cells)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [BatteryPFN] repli L3 indisponible ({exc})")
            self._replis = None

        self.mode = "PFN" if self._charge_reseau() else "L3 (repli)"
        print(f"  [BatteryPFN] mode={self.mode}  cellules de contexte="
              f"{len(self.contexte)}")
        return self

    # -------------------------------------------------------------- predict
    def predict_soh(self, temperature_degC, c_rate):
        if self._reseau is not None:
            try:
                return self._predit_pfn(float(temperature_degC), float(c_rate))
            except Exception as exc:                   # noqa: BLE001
                print(f"  [BatteryPFN] passage avant echoue ({exc}) - repli L3")
        if self._replis is not None:
            return self._replis.predict_soh(temperature_degC, c_rate)
        raise RuntimeError("ni PFN ni repli disponibles")

    def _predit_pfn(self, T, C):
        torch = self._torch
        ctx = [_normalise(np.full(len(n), t), np.full(len(n), c), n, y)
               for (t, c, n, y) in self.contexte]
        Xc = torch.tensor(np.concatenate(ctx)[None], dtype=torch.float32)
        Xq = torch.tensor(
            _normalise(np.full(len(GRILLE), T), np.full(len(GRILLE), C),
                       GRILLE)[None], dtype=torch.float32)

        q = self._reseau(Xc, Xq)[0].numpy()
        i_med = list(self._reseau.quantiles).index(0.50)
        soh_grille = q[:, i_med] * SOH_ECH + SOH_MOY

        n_plein = np.arange(1, N_MAX + 1, dtype=float)
        traj = np.interp(n_plein, GRILLE, soh_grille)
        # Enveloppe cumulative : le reseau produit deja des trajectoires
        # monotones sur les 143 conditions de la grille notee (taux de violation
        # brut mesure = 0), mais la borne garantit la propriete plutot que de
        # l'esperer.
        traj = np.minimum.accumulate(traj)
        return np.clip(traj, SOH_LO, SOH_HI)

    # ------------------------------------------------- persistance explicite
    # persistence.py utilise ces deux methodes des qu'elles existent, ce qui
    # evite de pickler un module PyTorch et toutes les cornieres associees.
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        np.savez(os.path.join(folder, "contexte.npz"),
                 n_cellules=len(self.contexte),
                 **{f"T_{i}": np.array([t]) for i, (t, c, n, y)
                    in enumerate(self.contexte)},
                 **{f"C_{i}": np.array([c]) for i, (t, c, n, y)
                    in enumerate(self.contexte)},
                 **{f"n_{i}": n for i, (t, c, n, y) in enumerate(self.contexte)},
                 **{f"y_{i}": y for i, (t, c, n, y) in enumerate(self.contexte)})
        if self._replis is not None:
            import pickle
            with open(os.path.join(folder, "repli_l3.pkl"), "wb") as fh:
                pickle.dump(self._replis, fh)

    @classmethod
    def load(cls, folder):
        m = cls()
        d = np.load(os.path.join(folder, "contexte.npz"))
        k = int(d["n_cellules"])
        m.contexte = [(float(d[f"T_{i}"][0]), float(d[f"C_{i}"][0]),
                       d[f"n_{i}"], d[f"y_{i}"]) for i in range(k)]
        p = os.path.join(folder, "repli_l3.pkl")
        if os.path.exists(p):
            import pickle
            with open(p, "rb") as fh:
                m._replis = pickle.load(fh)
        m.mode = "PFN" if m._charge_reseau() else "L3 (repli)"
        print(f"  [BatteryPFN] recharge, mode={m.mode}")
        return m
