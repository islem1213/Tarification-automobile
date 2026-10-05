# Tarification automobile 


| Fichier | Contenu |
|---|---|
| `Synthese_Pricing_Auto.docx` | La synthèse de 2 pages demandée (méthodologie, difficultés, résultats, comparaison des approches, conclusions) |
| `pricing_auto.ipynb` | Le notebook complet et commenté, **déjà exécuté** : il s'ouvre avec tous les résultats et graphiques visibles, sans rien relancer |


| § | Contenu | Point de l'énoncé |
|---|---|---|
| 1 | Fiabilisation et préparation des données | **A** |
| 2 | Séparation attritionnels / graves | **B** |
| 3 | Approche Fréquence × Coût | **C** |
| 4 | Séparation Train / Test | **F** |
| 5 | Modélisation GLM | **D** |
| 6 | Comparaison des approches de pricing | **E** |
| 7 | Validation hors échantillon | **D / F** |
| 8-9 | Tarif commercial, synthèse et limites | complément |

Le point F est traité avant le point D parce qu'un modèle doit être calibré sur
l'échantillon d'apprentissage seul : construire le découpage après l'estimation lui
retirerait toute valeur de validation.

## Réexécuter

```bash
python pricing_auto.py
```

Le fichier `pricing_auto.py` est le script complet ; le notebook en est la conversion.
Les deux contiennent exactement le même code, ce qui évite d'avoir deux versions à
maintenir. Pour régénérer le notebook après une modification du script :

```bash
python construire_notebook.py
```

Et pour régénérer la synthèse — dont tous les chiffres sont **relus** depuis les
tableaux exportés, et ne peuvent donc pas diverger du notebook :

```bash
python construire_synthese.py
```

## Organisation du dossier

```
projet_actuariat/
├── data/                        les deux bases + le cours (Chapter_II.xlsx)
├── pricing_auto.ipynb           notebook exécuté        ← livrable
├── Synthese_Pricing_Auto.docx   synthèse 2 pages        ← livrable
├── pricing_auto.py              même code, format script
├── construire_notebook.py       script → notebook
├── construire_synthese.py       tableaux → synthèse
└── outputs/
    ├── tables/                  42 tableaux .xlsx
    └── figures/                 12 graphiques .png
```

## Dépendances

`pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `openpyxl` pour l'analyse ;
`nbformat` et `nbclient` pour la conversion ; `python-docx` pour la synthèse.
**`scikit-learn` n'est pas nécessaire** : les GLM passent par `statsmodels`, la loi de
Pareto généralisée par `scipy`, et le découpage train/test comme les métriques de
validation sont codés explicitement.

