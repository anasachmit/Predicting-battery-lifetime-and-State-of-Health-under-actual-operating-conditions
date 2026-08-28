# Run v1 — archivé, défaut de protocole

Ce run n'est pas jeté : il DOCUMENTE un défaut de protocole, ce qui est un
résultat en soi pour le rapport.

## Le défaut

`data.cycles_requete()` bornait la grille de requête d'entraînement au
franchissement de 70 % de SOH (× U[1.00, 1.05]). Le réseau n'était donc jamais
interrogé au-delà, et le dernier point de chaque exemple valait toujours ~70 %.

À l'évaluation, `evaluate.GRILLE_EVAL` interroge de 1 à 12 000 cycles pour
toutes les cellules. Pour une cellule dont n@70 vaut 3 000, les cycles
3 000–12 000 sont hors distribution d'entraînement.

## La signature, mesurée sur 120 cellules

```
n70 predit / n70 vrai : mediane 2.27
part predite TROP TOT : 0 %          <- systematique, jamais du bruit
RMSE zone notee       : 3.25         <- MEILLEUR que la borne C.2a (3.53)
```

Le réseau prédit bien la trajectoire là où il a été entraîné, et place le
franchissement de 70 % deux fois trop tard, **toujours dans le même sens**.

## Pourquoi c'est un résultat

C'est précisément ce que les jalons intermédiaires servaient à attraper. Avec un
seul F.3 final, on aurait lu « APE 131 % » comme « le mécanisme ne marche pas »
et fermé la branche sur une conclusion fausse. Le jalon à 100k, croisé avec le
RMSE, a montré que le défaut était dans le protocole d'évaluation, pas dans le
modèle.

État à l'archivage : 129 280 laboratoires vus, pas 2020.
