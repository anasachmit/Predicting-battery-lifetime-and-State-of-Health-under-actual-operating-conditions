# Challenge 1 — modèle d'ancrage : baseline refité + knee ancré en SOH

Soumission volontairement simple, destinée à établir une position de référence
et à mesurer l'écart entre notre proxy de métrique et le score officiel. Le
modèle complet (terme calendaire, quantification d'incertitude) suit.

## Méthode

```
base(n) = s0 − b · n^0.8                      (baseline officiel)
n_k     = cycle où base(n) franchit SOH_k     (ancrage en SOH, pas en cycles)
SOH(n)  = base(n) − c · (exp((n − n_k)/(n_k/β)) − 1)_+     puis borné
```

`SOH_k`, `c` et `β` sont ajustés **conjointement** sur toutes les cellules
d'entraînement (un jeu commun) ; seul `b` est propre à chaque cellule, puis
envoyé sur les conditions par `ln(b) = w0 + w1·1000/T_K`.

Trois choix, chacun appuyé sur une mesure et non sur une intuition.

**1. `s0` est mesuré, pas ajusté.** Médiane des cycles ≤ 10, moyennée sur les
cellules (102,47 % en convention nominale). Son étendue inter-cellules est de
0,84 point de SOH et il ne montre pas de dépendance à la température (pente
−0,015 pt/°C, p = 0,16, n = 6). Laissé libre, il absorbe une partie du knee et
descend vers 100 %, ce qui biaise toute la trajectoire précoce.

**2. Le knee est ancré par le niveau de SOH où il démarre, pas par un nombre de
cycles.** Ajusté librement, `n_k` se disperse d'un facteur 6 entre cellules et
sa carte (T, C) ne rend que R² = 0,31. Le niveau de SOH de déclenchement, lui,
tient dans **78–85 %** sur les 6 cellules, sans dépendance visible à T ni à C —
et reste à ±1 % en validation leave-one-cell-out. C'est le paramètre
identifiable, donc c'est celui qu'on estime.

**3. Pas de terme en ln(C) dans la carte des conditions.** À nombre de cycles
égal, les paires 0,5C / 1C à 25 °C et à 45 °C diffèrent de moins de 1,5 point de
SOH, et l'écart **change de signe** au cours de la vie. La pente en ln(C) n'est
pas identifiable sur deux contrastes ; laissée libre elle prend ici une valeur
négative — soit un vieillissement *par cycle* plus rapide à 0,5C qu'à 1C — qui
s'extrapole en aberration vers le coin 55 °C / 0,5C, coin où aucune cellule
n'existe et qui tombe hors de l'enveloppe convexe du plan d'expérience.

L'effet du C-rate est bien réel, mais il agit **par le temps passé** : à durée
écoulée égale, 0,5C conserve 1,6 à 5,6 points de SOH de plus que 1C. La durée
de cycle mesurée vaut `(2,08/C)·SOH + 1,09` heures (R² = 0,86, coefficient à 4 %
de sa valeur physique : charge CC + décharge CC consomment `2·SOH/C` heures).
Un terme calendaire est la bonne façon de capter cet effet ; il n'est pas dans
ce modèle d'ancrage.

## Jeux de données ouverts utilisés

Aucun n'entre dans cette soumission — le modèle est entraîné uniquement sur les
cellules publiées. Deux sources ont servi à cadrer la méthode et seront
utilisées pour les priors du modèle complet :

- **Wheeler et al. 2025**, 20 cellules A123 18650 Graphite/LFP 1,1 Ah,
  [DOI 10.57745/OLBXKT](https://doi.org/10.57745/OLBXKT), CC BY 4.0. Seule
  source publique où 20/20 cellules descendent sous 70 % de SOH (18/20 sous
  60 %) — donc la seule qui documente la forme du knee. Toutes à 50 °C.
- **SNL via BatteryLife** (Tan et al., KDD 2025),
  [Zenodo 19688272](https://zenodo.org/records/19688272), CC BY 4.0. 18 cellules
  18650 LFP sur une grille 15/25/35 °C × 0,5–3C. Aucune n'atteint 70 %.

MATR / Severson-Attia a été écarté : ses 169 cellules sont toutes à 30 °C avec
un protocole de décharge identique, et s'arrêtent à 80 % de SOH.

## Méthodologie de validation

**Ce qui est tenu à l'écart.** Leave-one-cell-out strict sur les 6 cellules.
Aucun partage de cycles entre entraînement et test.

**Pourquoi le LOCO ne suffit pas ici, et ce qu'on ajoute.** Dans l'espace
`(1000/T_K, ln C)` — celui où la loi Arrhenius × puissance est linéaire —
**4 des 6 folds sont extrapolants** et 3 retirent un coin du domaine noté. Pire,
sur les 4 cellules dont le franchissement de 70 % est observé, **3 sont
extrapolantes** : le seul fold à la fois interpolant et doté d'une vérité
terrain sur `n@70` est 35 °C/1C. Les erreurs sur `n@70` sont donc rapportées
**cellule par cellule**, jamais en médiane.

Le plan d'expérience complet lui-même n'est pas fermé : le coin **55 °C / 0,5C
ne contient aucune cellule et tombe hors de l'enveloppe convexe**. On ajoute
donc un **audit de plausibilité physique éliminatoire**, balayant toute la
grille 25–55 °C × 0,5–1,0 C : trajectoires finies et dans (0, 120], SOH ≤ 103 %,
décroissance monotone, `n@70` défini partout et décroissant en T et en C, aucun
knee avant 30 cycles. Le modèle soumis passe les 8 critères ; la variante à
exposant en C-rate libre en échoue un.

**Métriques.** La métrique officielle n'étant pas publiée, deux grilles de
cycles sont rapportées — uniforme en cycle (qui pondère la longue zone plate) et
uniforme en SOH (qui pondère la fin de vie) — et la sélection se fait sur le
**maximum** des deux, jamais sur leur moyenne. L'écart entre les deux grilles
sert d'indicateur de fragilité.

**Ce que le LOCO dit de cette soumission, sans arrondi favorable.** Contre le
baseline officiel **refité sur les mêmes 5 cellules** à chaque fold : RMSE pire
cas médian **4,84 contre 4,36** (ratio médian 1,00 — équivalent), mais erreur
médiane sur le cycle d'atteinte de 70 % de **19,7 % contre 62,9 %**. Le gain est
donc sur la prédiction de fin de vie, pas sur l'erreur moyenne de trajectoire,
et il est très inégal selon les cellules (11 %, 28 %, 8 %, 65 %).

**Limite connue.** La pente d'Arrhenius `w1` varie d'un facteur 3,6 selon la
cellule retirée (−0,61 à −2,17) : avec 4 niveaux de température et 6 cellules,
retirer une cellule ampute le contraste en T. Le LOCO sous-estime donc la
qualité du modèle entraîné sur les 6, mais signale une fragilité réelle de la
carte des conditions.

## Reproductibilité

Graine fixée (`RANDOM_SEED = 20260101`) dans `my_model/model_template.py`.
Aucun accès réseau, aucun chemin absolu, aucun poids pré-entraîné. Dépendances :
`numpy`, `pandas`, `scipy`. Entraînement complet sur les 6 cellules : **10 s**.
