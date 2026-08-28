# État BatteryPFN — arrêt du 27/08 au matin

Point de reprise. Tout ce qui suit est mesuré, pas estimé.

---

## 1. Où on en est

**Entraînement arrêté proprement au pas 2020**, checkpoint complet sur disque.

| | |
|---|---|
| laboratoires vus | **129 280** sur 400 000 visés (32 %) |
| pas | 2 020 sur 6 250 |
| débit réel mesuré | **36 lab/s** (benchmark pur : 42 ; le reste part en évaluations et chargement de shards) |
| perte train / valid | 0.0201 / 0.0182 |
| écart de calibration train / valid | 0.0218 / 0.0339 (dérive +0.0121) |
| jalons F.3 faits | 100 000 |

**Pool non épuisé.** La dérive de calibration oscille entre −0.010 et +0.014 sans
tendance ; le déclencheur d'arrêt (train < 0.02 **et** dérive > 0.03) n'a jamais
été proche. La règle reste armée dans `scripts/train_pfn.py`.

### Contenu du checkpoint — vérifié

```
cles           : ['batch','jalons_faits','labos_vus','modele','n_ctx','opt','pas']
pas            : 2020
labos_vus      : 129,280
jalons_faits   : [100000]
n_ctx / batch  : 32 / 64
tenseurs poids : 61 entrees
etat AdamW     : 61 parametres avec moments, step = 2020
```

### Reprise — TESTÉE, pas supposée

Test effectué le 27/08 : reprise sur 20 pas depuis le pas 2000.

```
pas        : 2000 -> 2020        (+20, attendu +20)
labos_vus  : 128,000 -> 129,280  (+1280, attendu +1280)
jalons     : [100000]            (conserve)
AdamW step : 2020                (CONTINUE depuis 2000, pas remis a zero)
REPRISE VALIDEE
```

---

## 2. Commande de reprise

```
cd c:/Users/anasa/Desktop/batterylife_health
.venv/Scripts/python.exe scripts/train_pfn.py --labos 400000
```

**Le `--labos 400000` est obligatoire.** Il fixe le total du programme de taux
d'apprentissage (cosinus). Le test de reprise a tourné avec `--labos 129280`,
ce qui a fait croire au programme qu'il était arrivé au bout et a écrasé le
taux à 1.5e-5. Relancer avec 400000 recalcule correctement le taux pour le pas
2020 sur un total de 6250 — rien n'est perdu, mais omettre l'argument
entraînerait à taux quasi nul.

Artefacts parasites laissés par ce test, à ignorer ou supprimer :
`f3_final.csv` et la ligne `"jalon": "final"` du journal, écrits au pas 2020.

Temps restant estimé pour aller au bout : (400000 − 129280) / 36 ≈ **2 h 05**.

---

## 3. LE PROBLÈME À TRAITER EN PRIORITÉ

### F.3 à 100k laboratoires

```
arrhenius 91.48 %   charniere 130.76 %   seuil 15.05 %   atteint : False
```

Contre une borne de 13.53 / 15.81. **Mais ce n'est pas un verdict sur le
mécanisme** — c'est un défaut de protocole que j'ai introduit, et le diagnostic
est sans ambiguïté :

```
n70 predit / n70 vrai : mediane 2.27
part predite TROP TOT : 0 %          <- systematique, jamais du bruit
RMSE zone notee       : 3.25         <- MEILLEUR que la borne (3.53)
n70 vrai   : med 4453   q05 1456  q95 8271
n70 predit : med 9768   q05 6335  q95 11390
```

Le réseau prédit **bien la trajectoire là où il a été entraîné** (RMSE 3.25 <
3.53) mais place le franchissement de 70 % deux fois trop tard, toujours dans le
même sens.

**Cause.** Dans `data.py`, `cycles_requete()` borne la grille de requête au
franchissement de 70 % (× U[1.00, 1.05]). Le réseau n'est donc jamais interrogé
au-delà, et le dernier point de chaque exemple vaut toujours ~70 %. À
l'évaluation, `evaluate.GRILLE_EVAL` interroge de 1 à 12 000 pour toutes les
cellules : pour une cellule dont n@70 vaut 3 000, les cycles 3 000–12 000 sont
hors distribution d'entraînement. Le réseau y maintient une valeur proche de
70 %, et le franchissement est repoussé.

J'avais borné la grille pour « ne pas gaspiller la capacité hors zone notée ».
C'était une erreur : l'interface de soumission exige `predict_soh` sur
**1..12 000**, donc l'entraînement doit couvrir la même plage de cycles que
l'inférence.

**Correctif à appliquer.** Étendre la grille de requête d'entraînement bien
au-delà de n@70 — jusqu'à SOH ≈ 40 % ou un multiple tiré de n@70, plafonné à
12 000 — puis **réentraîner depuis zéro** (le réseau actuel a appris la mauvaise
plage de `ln n`, le poursuivre ne la corrigera pas proprement).

Coût : ~2 h de réentraînement. À décider en fonction du temps restant.

---

## 4. Ce qui reste

| tâche | état | note |
|---|---|---|
| **F.3** aux jalons | 100k fait, invalidé par le défaut ci-dessus | à refaire après correctif |
| **F.2** | non commencé | **LOCO sur les 6 cellules RÉELLES**, pas de reconstruction synthétique |
| **F.4** | non commencé | |
| rapport | non commencé | le tableau 3.93 / 2.28 est déjà publiable |

**F.2 — définition à ne pas confondre.** C'est un leave-one-cell-out sur les
**six cellules cible réelles** : contexte = 5 cellules réelles observées,
requête = le (T, C) de la sixième, comparaison à sa trajectoire mesurée. Ce
n'est pas le test de reconstruction sur données synthétiques.

---

## 5. Limite dure

**F.3 et F.2 rendus le 28 au soir**, sinon la branche s'arrête et se documente
en perspectives. Gel de la soumission le 29 au soir.

Rappel de ce qui est déjà acquis et publiable sans réseau entraîné :

| effet mesuré sur la borne C.2a | coût en APE n@70 |
|---|---:|
| figer l'exposant `p` à 0.8 | **3.93 pts** |
| se tromper de structure de `b(T)` | 2.28 pts |

`L3` / `v3` reste la soumission. Rien n'a été basculé.

---

## 6. Arbitrage à faire demain matin

Avec ~1 jour avant la limite dure, deux voies :

1. **Corriger la grille et réentraîner** (~2 h) puis F.3, puis F.2. Tient si
   rien d'autre ne casse.
2. **Renoncer au réseau** et livrer la branche sur ses résultats analytiques —
   le tableau 3.93 / 2.28, la fermeture de L4, la borne recalculée — qui sont
   solides et déjà écrits dans `VERDICT.md`.

La voie 1 reste possible mais sans marge. La voie 2 est le repli sûr et ne perd
rien de ce qui a été établi.
