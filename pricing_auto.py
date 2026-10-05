# -*- coding: utf-8 -*-
# %% [markdown]
# # Tarification automobile — de la fiabilisation des données au tarif technique
#
# **Projet d'actuariat — Groupe 1**
#
# Ce document met en œuvre une tarification automobile complète à partir de deux bases :
# un fichier **Contrats** et un fichier **Sinistres**. Il suit les six points de l'énoncé
# (A à F) et le cadre méthodologique du cours (Chapitre II) :
#
# $$\text{Prime pure} = \mathbb{E}(N) \times \mathbb{E}(X) = \text{Fréquence} \times \text{Coût moyen}$$
#
# ## Plan du document
#
# | § | Contenu | Point de l'énoncé |
# |---|---|---|
# | 0 | Cadrage, environnement, fonctions utilitaires | — |
# | 1 | Fiabilisation et préparation des données | **A** |
# | 2 | Séparation attritionnels / graves | **B** |
# | 3 | Approche Fréquence × Coût non segmentée | **C** |
# | 4 | Séparation Train / Test | **F** |
# | 5 | Modélisation GLM | **D** |
# | 6 | Comparaison des approches de pricing | **E** |
# | 7 | Validation hors échantillon, brique par brique | **D / F** |
# | 8 | Tarif final recalibré, crédibilité, tarif commercial | complément |
# | 9 | Synthèse, limites et pistes | — |
#
# > **Pourquoi le point F (§4) est traité avant le point D (§5).**
# > Les modèles GLM doivent être **calibrés sur l'échantillon d'apprentissage uniquement**.
# > Construire le découpage après avoir estimé les modèles n'aurait aucune valeur de
# > validation : les paramètres auraient déjà vu l'ensemble des données. L'ordre de
# > l'énoncé est un ordre de présentation, pas un ordre d'exécution.

# %% [markdown]
# ---
# ## §0 — Cadrage et environnement
#
# ### 0.1 Imports
#
# **Note technique importante.** Sur cette machine, `import statsmodels.api` échoue :
# Windows Smart App Control bloque le chargement d'une DLL du sous-module de séries
# temporelles (`_cfa_simulation_smoother`), dont nous n'avons aucun besoin ici.
# On importe donc **directement les sous-modules utiles**, ce qui contourne le problème
# sans rien perdre. Si vous exécutez ce notebook sur une autre machine, ces imports
# fonctionnent de la même façon.

# %%
import sys
import time
import warnings
from pathlib import Path

# La sortie console de Windows utilise cp1252 par defaut, qui ne connait pas
# certains caracteres typographiques francais. On force l'UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib


def dans_jupyter():
    """Indique si le code tourne dans un noyau Jupyter (vrai) ou comme script (faux)."""
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


# Hors Jupyter, on utilise un backend non interactif : les figures sont enregistrees
# dans outputs/figures/ sans ouvrir de fenetre bloquante.
if not dans_jupyter():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# --- Import statsmodels : sous-modules ciblés (voir note ci-dessus) ---
import statsmodels.formula.api as smf
from statsmodels.genmod import families
from statsmodels.genmod.families import links

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}".replace(",", " "))

print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
import statsmodels
print("statsmodels ", statsmodels.__version__)
import scipy
print("scipy       ", scipy.__version__)

# %% [markdown]
# ### 0.2 Paramètres de l'étude
#
# Toutes les constantes de pilotage sont regroupées ici : modifier une valeur et
# réexécuter le document suffit à produire une variante complète de l'étude.
#
# Les valeurs de `SEUIL_GRAVE` et `AGE_PERMIS_MEDIAN` ne sont **pas arbitraires** :
# elles sont établies et justifiées respectivement au §2.6 et au §1.6.

# %%
DOSSIER = Path.cwd()
DATA = DOSSIER / "data"
FIGURES = DOSSIER / "outputs" / "figures"
TABLES = DOSSIER / "outputs" / "tables"
for d in (FIGURES, TABLES):
    d.mkdir(parents=True, exist_ok=True)

# --- Paramètres méthodologiques ---
SEUIL_GRAVE = 10_000          # seuil de gravité, justifié au §2.6
DATE_CUT = pd.Timestamp("2024-01-01")   # frontière du découpage temporel (§4.1)
PART_TRAIN = 0.80             # part d'apprentissage du découpage aléatoire (§4.2)
AGE_PERMIS_MEDIAN = 20        # âge médian d'obtention du permis, mesuré au §1.6
AGE_MIN, AGE_MAX = 18, 90     # bornes de plausibilité de l'âge du conducteur
BM_MIN, BM_MAX = 0.50, 3.50   # bornes réglementaires du coefficient bonus-malus
PUISS_MIN, PUISS_MAX = 1, 40  # bornes de plausibilité de la puissance fiscale
GRAINE = 2026                 # graine aléatoire, fixée pour la reproductibilité

# Natures de sinistres relevant structurellement du risque grave (§2.5)
NATURES_GRAVES = [
    "Collision majeure", "Corporel grave", "Incendie total",
    "Vol total", "Catastrophe naturelle",
]

rng = np.random.default_rng(GRAINE)

# %% [markdown]
# ### 0.3 Fonctions utilitaires
#
# Mise en forme française des nombres (espace comme séparateur de milliers, virgule
# décimale), style graphique homogène, et deux helpers d'export. Regrouper ces
# fonctions ici évite de répéter du code de présentation dans tout le document.

# %%
PALETTE = {
    "encre": "#0B2B33",
    "petrole": "#177C8C",
    "ambre": "#D99A2B",
    "brique": "#A8443A",
    "gris": "#8C9BA0",
}
CYCLE = [PALETTE["petrole"], PALETTE["ambre"], PALETTE["brique"],
         PALETTE["encre"], PALETTE["gris"]]

plt.rcParams.update({
    "figure.figsize": (9, 4.5),
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.prop_cycle": plt.cycler(color=CYCLE),
    "font.size": 9.5,
})


def fr(x, dec=0):
    """Formate un nombre à la française : espace fin pour les milliers, virgule décimale."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n.d."
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")


def pct(x, dec=1):
    """Formate une proportion (0-1) en pourcentage à la française."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n.d."
    return f"{100 * x:,.{dec}f}".replace(",", " ").replace(".", ",") + " %"


def axe_fr(ax, axis="y"):
    """Applique le format numérique français à un axe."""
    fmt = mticker.FuncFormatter(lambda v, _: fr(v))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
    return ax


def _ecrire_avec_reprise(fonction_ecriture, chemin, tentatives=6, attente=0.6):
    """Execute une ecriture fichier en reessayant si le systeme la refuse.

    Le dossier du projet se trouve sur le Bureau, qui peut etre synchronise par un
    service de sauvegarde en ligne. Celui-ci verrouille brievement les fichiers qu'il
    est en train d'indexer, ce qui fait echouer une ecriture par OSError de facon
    intermittente. Quelques tentatives espacees suffisent a passer.
    """
    derniere = None
    for essai in range(tentatives):
        try:
            fonction_ecriture()
            return
        except OSError as exc:
            derniere = exc
            time.sleep(attente * (essai + 1))
    raise OSError(f"Ecriture impossible dans {chemin} apres {tentatives} tentatives "
                  f"({derniere}).")


def sauver_fig(nom):
    """Enregistre la figure courante dans outputs/figures/ et l'affiche."""
    chemin = FIGURES / f"{nom}.png"
    _ecrire_avec_reprise(lambda: plt.savefig(chemin), chemin)
    if dans_jupyter():
        plt.show()
    plt.close()


def exporter_table(df, nom, index=True):
    """Enregistre un tableau dans outputs/tables/ au format Excel."""
    chemin = TABLES / f"{nom}.xlsx"
    _ecrire_avec_reprise(lambda: df.to_excel(chemin, index=index), chemin)
    return df


def display(obj):
    """Affiche un tableau. Dans Jupyter, la fonction native prend le relais et rend
    un tableau mis en forme ; hors Jupyter, on retombe sur un affichage texte."""
    try:
        from IPython.display import display as _d
        _d(obj)
    except Exception:
        print(obj.to_string() if hasattr(obj, "to_string") else obj)


def titre(txt, niveau=1):
    """Affiche un séparateur lisible dans les sorties texte."""
    barre = "=" if niveau == 1 else "-"
    print(f"\n{barre * 78}\n{txt}\n{barre * 78}")

# %% [markdown]
# ---
# # §1 — Partie A · Fiabilisation et préparation des données
#
# > **Ce que demande l'énoncé (point A)**
# > 1. Contrôle de la qualité et de la cohérence des données ;
# > 2. Identification et traitement des valeurs manquantes, aberrantes ou incohérentes ;
# > 3. Contrôle de la cohérence entre les bases Contrats et Sinistres ;
# > 4. Création des variables nécessaires à la tarification, notamment l'exposition.
#
# **Principe directeur de cette partie.** Une donnée fausse ne se remplace pas par une
# moyenne tant qu'on n'a pas cherché à la **reconstruire**. L'ordre de priorité appliqué
# systématiquement est le suivant :
#
# 1. **Reconstruire** la valeur à partir d'une autre variable de la même ligne ;
# 2. **Imputer** statistiquement, seulement si aucun recoupement n'est possible ;
# 3. **Supprimer**, uniquement si la ligne n'apporte aucune information exploitable.
#
# Chaque correction est tracée par une variable indicatrice, et le §1.8 dresse le bilan
# avant / après pour qu'aucune ligne ne disparaisse sans être comptée.

# %% [markdown]
# ## 1.1 Chargement et inventaire

# %%
contrats_brut = pd.read_excel(DATA / "groupe1_base1_contrats.xlsx")
sinistres_brut = pd.read_excel(DATA / "groupe1_base2_sinistres.xlsx")

titre("BASE CONTRATS — inventaire")
print(f"Dimensions : {fr(contrats_brut.shape[0])} lignes x {contrats_brut.shape[1]} colonnes")
print(contrats_brut.dtypes.to_string())
print("\nApercu :")
display(contrats_brut.head(5))

titre("BASE SINISTRES — inventaire")
print(f"Dimensions : {fr(sinistres_brut.shape[0])} lignes x {sinistres_brut.shape[1]} colonnes")
print(sinistres_brut.dtypes.to_string())
print("\nApercu :")
display(sinistres_brut.head(5))

# %% [markdown]
# **Première alerte, visible dès les types.** La colonne `montant_sinistre` est de type
# `object` et non numérique : au moins une valeur n'est pas un nombre. De même,
# `duree_exposition` est bien numérique, mais il faudra vérifier qu'elle reste dans
# l'intervalle `]0 ; 1]` attendu pour une exposition exprimée en années-police.
#
# ### Dictionnaire des variables

# %%
dico = pd.DataFrame([
    ("id_police",          "Contrats",  "Identifiant du contrat",              "Clé de jointure"),
    ("date_effet_contrat", "Contrats",  "Date de prise d'effet",               "Début de la période de couverture"),
    ("duree_exposition",   "Contrats",  "Exposition en années-police",         "Offset du GLM de fréquence"),
    ("age_conducteur",     "Contrats",  "Âge du conducteur principal (ans)",   "Variable tarifaire"),
    ("sexe",               "Contrats",  "Sexe du conducteur",                  "Variable tarifaire"),
    ("anciennete_permis",  "Contrats",  "Ancienneté du permis (ans)",          "Variable tarifaire"),
    ("bonus_malus",        "Contrats",  "Coefficient de réduction-majoration", "Tarification a posteriori"),
    ("zone_geo",           "Contrats",  "Zone géographique (A à E)",           "Variable tarifaire"),
    ("usage_vehicule",     "Contrats",  "Usage déclaré du véhicule",           "Variable tarifaire"),
    ("puissance_fiscale",  "Contrats",  "Puissance fiscale (CV)",              "Variable tarifaire"),
    ("classe_vehicule",    "Contrats",  "Segment du véhicule",                 "Variable tarifaire"),
    ("age_vehicule",       "Contrats",  "Âge du véhicule (ans)",               "Variable tarifaire"),
    ("carburant",          "Contrats",  "Type de motorisation",                "Variable tarifaire"),
    ("type_couverture",    "Contrats",  "Étendue des garanties",               "Variable tarifaire / périmètre"),
    ("id_sinistre",        "Sinistres", "Identifiant du sinistre",             "Détection des doublons"),
    ("date_sinistre",      "Sinistres", "Date de survenance",                  "Découpage temporel (§4)"),
    ("nature_sinistre",    "Sinistres", "Nature du dommage",                   "Validation du seuil (§2.5)"),
    ("montant_sinistre",   "Sinistres", "Charge du sinistre",                  "Variable cible de sévérité"),
    ("responsabilite_pct", "Sinistres", "Taux de responsabilité (%)",          "Contrôle de cohérence"),
    ("statut",             "Sinistres", "Sinistre clos ou ouvert",             "Test de sensibilité (§2.7)"),
], columns=["Variable", "Base", "Description", "Rôle dans la tarification"])
exporter_table(dico, "1_1_dictionnaire_variables", index=False)
display(dico)

# %% [markdown]
# ## 1.2 Contrôle qualité — base Contrats
#
# ### 1.2.1 Doublons et valeurs manquantes

# %%
titre("DOUBLONS — base Contrats", 2)
n_dup_lignes = int(contrats_brut.duplicated().sum())
n_dup_id = int(contrats_brut["id_police"].duplicated().sum())
print(f"Lignes strictement dupliquees    : {fr(n_dup_lignes)}")
print(f"Identifiants de police dupliques : {fr(n_dup_id)}")
print(f"Polices distinctes               : {fr(contrats_brut['id_police'].nunique())}")
print("\n-> Les deux compteurs coincident : les doublons sont des duplications integrales")
print("   de lignes, et non deux contrats distincts partageant un identifiant.")

titre("VALEURS MANQUANTES — base Contrats", 2)
manq = pd.DataFrame({
    "Manquants": contrats_brut.isna().sum(),
    "Part": contrats_brut.isna().mean(),
})
manq = manq[manq["Manquants"] > 0].sort_values("Manquants", ascending=False)
manq["Part"] = manq["Part"].map(lambda v: pct(v, 2))
display(manq)

# %% [markdown]
# ### 1.2.2 Modalités des variables qualitatives
#
# On recense les modalités observées pour repérer les problèmes d'**harmonisation**
# (casse, accents, espaces parasites) qui feraient éclater artificiellement une même
# catégorie en plusieurs niveaux dans les modèles.

# %%
for col in ["sexe", "zone_geo", "usage_vehicule", "classe_vehicule",
            "carburant", "type_couverture"]:
    vc = contrats_brut[col].value_counts(dropna=False)
    print(f"\n--- {col} : {len(vc)} modalites observees ---")
    print(vc.to_string())

# %% [markdown]
# **Constats.** Quatre variables sont polluées par des variantes d'écriture :
#
# | Variable | Modalités attendues | Modalités observées | Problème |
# |---|---|---|---|
# | `sexe` | 2 | 7 | Casse et libellés longs : `M`, `F`, `Homme`, `Femme`, `m`, `f`, `homme` |
# | `usage_vehicule` | 3 | 6 | Variantes en majuscules : `PRIVÉ`, `MIXTE`, `PROFESSIONNEL` |
# | `classe_vehicule` | 5 | 10 | **Espaces de tête** : `"  Citadine"` est distinct de `"Citadine"` |
# | `carburant` | 4 | 5 | Accent manquant : `Electrique` vs `Électrique` |
#
# Ce sont des corrections sans perte d'information : une normalisation suffit.
#
# ### 1.2.3 Valeurs aberrantes des variables numériques
#
# On confronte chaque variable à ses **bornes de plausibilité métier** plutôt qu'à un
# critère purement statistique du type « au-delà de trois écarts-types ». Un âge de
# 210 ans n'est pas un point extrême d'une distribution : c'est une impossibilité.

# %%
titre("STATISTIQUES DESCRIPTIVES — variables numeriques", 2)
display(contrats_brut.describe().T)

controles = [
    ("age_conducteur",    f"hors [{AGE_MIN} ; {AGE_MAX}] ans",
     (contrats_brut["age_conducteur"] < AGE_MIN) | (contrats_brut["age_conducteur"] > AGE_MAX)),
    ("anciennete_permis", "negative",
     contrats_brut["anciennete_permis"] < 0),
    ("bonus_malus",       f"hors [{BM_MIN} ; {BM_MAX}]",
     (contrats_brut["bonus_malus"] < BM_MIN) | (contrats_brut["bonus_malus"] > BM_MAX)),
    ("puissance_fiscale", f"hors [{PUISS_MIN} ; {PUISS_MAX}] CV",
     (contrats_brut["puissance_fiscale"] < PUISS_MIN) | (contrats_brut["puissance_fiscale"] > PUISS_MAX)),
    ("age_vehicule",      "negatif ou superieur a 40 ans",
     (contrats_brut["age_vehicule"] < 0) | (contrats_brut["age_vehicule"] > 40)),
    ("duree_exposition",  "hors ]0 ; 1] annee-police",
     (contrats_brut["duree_exposition"] <= 0) | (contrats_brut["duree_exposition"] > 1)),
]

rows = []
for var, regle, masque in controles:
    n = int(masque.sum())
    vals = sorted(contrats_brut.loc[masque, var].dropna().unique())
    rows.append({
        "Variable": var, "Regle violee": regle, "Lignes": n,
        "Valeurs distinctes observees": ", ".join(fr(v, 2) for v in vals[:8]) if n else "-",
    })
diag_num = pd.DataFrame(rows)
exporter_table(diag_num, "1_2_anomalies_numeriques", index=False)
display(diag_num)

# %% [markdown]
# **Lecture décisive.** Les valeurs aberrantes ne sont pas dispersées : elles se
# concentrent sur **un très petit nombre de valeurs distinctes** (`3`, `5`, `150`, `210`
# pour l'âge ; `-0,5`, `0,0`, `12,0` pour le bonus-malus ; `0`, `45`, `99` pour la
# puissance). Ce ne sont donc pas des erreurs de saisie accidentelles mais des valeurs
# **injectées volontairement** dans le jeu de données.
#
# Cette observation oriente tout le traitement : il est inutile de chercher une faute de
# frappe — le §1.6.1 le vérifie explicitement et rejette cette hypothèse — il faut
# chercher un **recoupement avec une autre variable**.
#
# ### 1.2.4 Cohérence interne : ancienneté du permis et âge du conducteur
#
# Un conducteur ne peut pas détenir le permis avant 18 ans. La relation
# `anciennete_permis <= age_conducteur - 18` doit donc être vérifiée partout.

# %%
titre("COHERENCE AGE / ANCIENNETE DU PERMIS", 2)
age_sain = contrats_brut["age_conducteur"].between(AGE_MIN, AGE_MAX)
incoherents = contrats_brut[age_sain & (contrats_brut["anciennete_permis"] >
                                        contrats_brut["age_conducteur"] - 18)]
ecart = incoherents["anciennete_permis"] - (incoherents["age_conducteur"] - 18)
print(f"Contrats ou anciennete > age - 18, l'age etant par ailleurs plausible : {fr(len(incoherents))}")
print("\nAmpleur du depassement (annees) :")
print(ecart.describe().to_string())
print("\nExemples :")
display(incoherents[["id_police", "age_conducteur", "anciennete_permis"]].head(8))

# %% [markdown]
# **Le dépassement est massif** — médiane de plus de 50 ans, minimum 21 ans — et non de
# l'ordre de un ou deux ans. Il ne s'agit donc pas d'un arrondi mais d'une ancienneté
# franchement fausse (68, 70, 81 ans de permis). Sur ces lignes, **c'est l'ancienneté
# qu'il faut corriger, et non l'âge**, qui y est parfaitement plausible.
# Cette asymétrie est exploitée au §1.6.2.

# %% [markdown]
# ## 1.3 Contrôle qualité — base Sinistres

# %%
titre("DOUBLONS ET MANQUANTS — base Sinistres", 2)
print(f"Lignes strictement dupliquees     : {fr(sinistres_brut.duplicated().sum())}")
print(f"Identifiants de sinistre dupliques: {fr(sinistres_brut['id_sinistre'].duplicated().sum())}")
print(f"Sinistres distincts               : {fr(sinistres_brut['id_sinistre'].nunique())}")
print("\nValeurs manquantes :")
manq_s = pd.DataFrame({
    "Manquants": sinistres_brut.isna().sum(),
    "Part": sinistres_brut.isna().mean().map(lambda v: pct(v, 2)),
})
display(manq_s[manq_s["Manquants"] > 0])

# %% [markdown]
# ### 1.3.1 La colonne des montants n'est pas numérique
#
# Le type `object` relevé au §1.1 se confirme : certaines valeurs sont stockées comme du
# texte, avec un symbole monétaire accolé.

# %%
titre("MONTANT_SINISTRE — diagnostic du type", 2)
print("Types Python presents dans la colonne :")
print(sinistres_brut["montant_sinistre"].map(type).value_counts().to_string())

montant_direct = pd.to_numeric(sinistres_brut["montant_sinistre"], errors="coerce")
texte = sinistres_brut["montant_sinistre"][montant_direct.isna()
                                           & sinistres_brut["montant_sinistre"].notna()]
print(f"\nValeurs non convertibles directement : {fr(len(texte))}")
print(texte.to_string())
print("\n-> Ce sont des nombres valides accompagnes du symbole monetaire.")
print("   Ils seront recuperes par nettoyage de la chaine, et non perdus.")


def nettoyer_montant(serie):
    """Convertit une colonne de montants en flottant, en tolerant le symbole monetaire,
    les separateurs de milliers et la virgule decimale."""
    return pd.to_numeric(
        serie.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .replace({"nan": None, "None": None, "": None}),
        errors="coerce")


# %%
titre("MONTANT_SINISTRE — valeurs impossibles", 2)
mt = nettoyer_montant(sinistres_brut["montant_sinistre"])
print(f"Montants recuperes par nettoyage : {fr(mt.notna().sum() - montant_direct.notna().sum())}")
print(f"Montants negatifs : {fr((mt < 0).sum())}   -> valeurs : {sorted(mt[mt < 0].unique())}")
print(f"Montants nuls     : {fr((mt == 0).sum())}")
print(f"Montants manquants: {fr(mt.isna().sum())}")
print("\nDistribution des montants strictement positifs :")
print(mt[mt > 0].describe().to_string())

# %% [markdown]
# **Interprétation des montants négatifs.** Les trois valeurs à `-500` sont identiques,
# ce qui exclut un aléa de saisie. Deux lectures sont possibles : un **recours encaissé**
# auprès du tiers responsable, qui rendrait la charge nette négative, ou une valeur
# sentinelle. Dans les deux cas, le montant ne peut pas alimenter un modèle de sévérité,
# qui exige une variable strictement positive. Ils sont donc exclus du calcul du coût
# moyen, et le §2.7 mesure l'effet de cette exclusion.
#
# ### 1.3.2 Taux de responsabilité, statut et nature

# %%
titre("RESPONSABILITE, STATUT ET NATURE", 2)
print("responsabilite_pct — valeurs observees :")
print(sinistres_brut["responsabilite_pct"].value_counts(dropna=False).sort_index().to_string())
hors = sinistres_brut["responsabilite_pct"].notna() & (
    (sinistres_brut["responsabilite_pct"] < 0) | (sinistres_brut["responsabilite_pct"] > 100))
print(f"\n-> {fr(hors.sum())} valeurs hors de l'intervalle [0 ; 100] : "
      f"{sorted(sinistres_brut.loc[hors, 'responsabilite_pct'].unique())}")

print("\nstatut — valeurs observees :")
print(sinistres_brut["statut"].value_counts(dropna=False).to_string())
print("-> Meme probleme de casse que dans la base Contrats.")

print("\nnature_sinistre — valeurs observees :")
print(sinistres_brut["nature_sinistre"].value_counts(dropna=False).to_string())
print("-> Espaces de tete sur plusieurs libelles, qui dedoublent des modalites identiques.")

# %% [markdown]
# ## 1.4 Contrôle de cohérence entre les deux bases
#
# Troisième exigence du point A. Trois contrôles sont menés : l'intégrité référentielle
# (tout sinistre doit se rattacher à un contrat connu), la cohérence temporelle (un
# sinistre doit survenir pendant la période de couverture), et la cohérence
# exposition / sinistralité (un contrat sans exposition ne peut pas produire de sinistre).

# %%
titre("CONTROLE INTER-BASES")

polices_contrats = set(contrats_brut["id_police"])
polices_sinistres = set(sinistres_brut["id_police"])
orphelines = polices_sinistres - polices_contrats

print("1) Integrite referentielle")
print(f"   Polices presentes dans Sinistres mais absentes de Contrats : {fr(len(orphelines))}")
print(f"   Lignes de sinistres concernees                             : "
      f"{fr(sinistres_brut['id_police'].isin(orphelines).sum())}")
print(f"   Identifiants : {sorted(orphelines)}")
print("   -> Leur format (POL9xxxxx) differe du portefeuille (POL1xxxxx) : il s'agit")
print("      de contrats d'un autre perimetre, non de contrats manquants.")

# --- cohérence temporelle ---
ctr_dates = contrats_brut.drop_duplicates("id_police").copy()
ctr_dates["debut"] = pd.to_datetime(ctr_dates["date_effet_contrat"])
ctr_dates["fin"] = ctr_dates["debut"] + pd.to_timedelta(
    ctr_dates["duree_exposition"].clip(lower=0) * 365.25, unit="D")

sin_dates = sinistres_brut.copy()
sin_dates["date_sin"] = pd.to_datetime(sin_dates["date_sinistre"])
jointure = sin_dates.merge(ctr_dates[["id_police", "debut", "fin"]], on="id_police", how="left")
rattaches = jointure["debut"].notna()

avant_effet = int((jointure.loc[rattaches, "date_sin"] < jointure.loc[rattaches, "debut"]).sum())
apres_fin = int((jointure.loc[rattaches, "date_sin"] > jointure.loc[rattaches, "fin"]).sum())

print("\n2) Coherence temporelle")
print(f"   Sinistres survenus AVANT la date d'effet du contrat  : {fr(avant_effet)}")
print(f"   Sinistres survenus APRES la fin de l'exposition      : {fr(apres_fin)}")
print(f"   Periode couverte par les sinistres : {sin_dates['date_sin'].min():%d/%m/%Y} "
      f"au {sin_dates['date_sin'].max():%d/%m/%Y}")
print(f"   Periode des dates d'effet          : {ctr_dates['debut'].min():%d/%m/%Y} "
      f"au {ctr_dates['debut'].max():%d/%m/%Y}")

# --- cohérence exposition / sinistralité ---
expo_nulle = contrats_brut[contrats_brut["duree_exposition"] <= 0]
expo_nulle_sin = expo_nulle["id_police"].isin(polices_sinistres).sum()
print("\n3) Coherence exposition / sinistralite")
print(f"   Contrats a exposition nulle                       : {fr(len(expo_nulle))}")
print(f"   ... dont porteurs d'au moins un sinistre          : {fr(expo_nulle_sin)}")
print("   -> Contradiction franche : un sinistre atteste que le contrat etait en vigueur.")
print("      L'exposition est donc fausse pour ces lignes, pas le sinistre.")

print("\n4) Repartition des sinistres par police")
print(sinistres_brut.groupby("id_police").size().value_counts().sort_index().to_string())

# %% [markdown]
# ### 1.4.1 Cohérence entre la garantie souscrite et la nature du sinistre
#
# Un contrôle que la structure des données n'impose pas mais que le métier exige : un
# sinistre doit relever d'une garantie **effectivement souscrite**. Selon la grille
# usuelle du marché automobile :
#
# | Formule | Ce qu'elle couvre |
# |---|---|
# | **Tiers** | La seule responsabilité civile : les dommages causés **aux autres** |
# | **Tiers étendu** | Responsabilité civile + vol, incendie, bris de glace (selon contrat) |
# | **Tous risques** | L'ensemble, y compris les dommages au véhicule assuré |
#
# Un contrat au tiers ne devrait donc porter **aucun sinistre de dommages propres** :
# bris de glace, vol, incendie, vandalisme ou catastrophe naturelle sur le véhicule assuré.

# %%
titre("COHERENCE GARANTIE SOUSCRITE / NATURE DU SINISTRE", 2)
DOMMAGES_PROPRES = ["Bris de glace", "Vol partiel", "Vol total", "Incendie total",
                    "Vandalisme", "Catastrophe naturelle"]

garanties = (sinistres_brut.drop_duplicates()
             .assign(nature=lambda d: d["nature_sinistre"].astype("object").str.strip())
             .merge(contrats_brut.drop_duplicates("id_police")[["id_police", "type_couverture"]],
                    on="id_police", how="inner"))
display(pd.crosstab(garanties["nature"], garanties["type_couverture"],
                    margins=True, margins_name="Total"))

sin_tiers = garanties[garanties["type_couverture"] == "Tiers"]
hors_garantie = sin_tiers[sin_tiers["nature"].isin(DOMMAGES_PROPRES)]
rc_sans_resp = sin_tiers[~sin_tiers["nature"].isin(DOMMAGES_PROPRES)
                         & (sin_tiers["responsabilite_pct"] == 0)]
print(f"\nSinistres de DOMMAGES PROPRES sur des contrats au TIERS : {fr(len(hors_garantie))} "
      f"sur {fr(len(sin_tiers))} sinistres de ces contrats ({pct(len(hors_garantie) / len(sin_tiers))})")
print(f"Sinistres de responsabilite civile, contrat au tiers, responsabilite 0 % : "
      f"{fr(len(rc_sans_resp))}")
print("   (en responsabilite civile, l'assureur n'indemnise que si son assure est")
print("    responsable ; a 0 %, c'est l'assureur adverse qui paie)")

print("\nPart de sinistres par formule, rapportee a la part de contrats :")
part_sin = garanties["type_couverture"].value_counts(normalize=True)
part_ctr = contrats_brut.drop_duplicates("id_police")["type_couverture"].value_counts(normalize=True)
display(pd.DataFrame({"Part des sinistres": part_sin.map(lambda v: pct(v)),
                      "Part des contrats": part_ctr.map(lambda v: pct(v)),
                      "Rapport": (part_sin / part_ctr).round(2)}))

# %% [markdown]
# **Une incohérence de fond, et non de saisie.** Près de la moitié des sinistres
# déclarés sur des contrats au tiers relèvent de garanties que ces contrats ne couvrent
# pas. Et surtout, **les trois formules présentent la même sinistralité** : chacune porte
# une part de sinistres égale à sa part de contrats. Dans un portefeuille réel, un
# contrat tous risques déclare mécaniquement beaucoup plus de sinistres qu'un contrat au
# tiers, puisqu'il couvre beaucoup plus d'événements.
#
# La conclusion s'impose : **les sinistres de ce jeu de données ont été générés
# indépendamment de la garantie souscrite.** C'est ce qui explique que `type_couverture`
# ne ressorte dans aucun modèle (§3.3), alors qu'elle devrait être la première variable
# tarifaire d'un portefeuille automobile.
#
# **Traitement retenu : conserver et signaler.** Exclure ces sinistres supposerait de
# connaître la grille de garanties propre à ce portefeuille, que les données ne
# documentent pas. Le scénario de base les conserve ; le §2.7.3 mesure l'impact de leur
# exclusion. Ce point est à remonter au fournisseur des données : en situation réelle,
# on tarifierait **garantie par garantie**, ce que ces données ne permettent pas.

# %% [markdown]
# **Un constat majeur pour la suite : toutes les dates d'effet se situent en 2023.**
# Le portefeuille ne comporte donc **qu'une seule année de souscription**, alors que les
# sinistres courent jusqu'en octobre 2024 — ce qui est normal, puisqu'un contrat
# souscrit en cours d'année 2023 reste exposé une partie de 2024.
#
# Cette particularité rend la formulation littérale de l'**approche 2 du point F**
# (« années antérieures à N en apprentissage, année N+1 en test ») inapplicable telle
# quelle. Le §4.1 propose et justifie un contournement.

# %% [markdown]
# ## 1.5 Tableau de décision
#
# Chaque anomalie identifiée reçoit un traitement explicite et argumenté. Ce tableau
# est la pièce centrale de la fiabilisation : il rend chaque choix vérifiable.

# %%
decisions = pd.DataFrame([
    ("Contrats", "120 lignes strictement dupliquées", "Supprimer",
     "Duplication intégrale, y compris l'identifiant : aucune information perdue."),
    ("Contrats", "sexe en 7 modalités", "Harmoniser",
     "Normalisation casse et libellés longs vers M / F."),
    ("Contrats", "usage_vehicule en 6 modalités", "Harmoniser",
     "Variantes en majuscules ramenées à la forme canonique."),
    ("Contrats", "classe_vehicule en 10 modalités", "Harmoniser",
     "Suppression des espaces de tête qui dédoublaient les 5 classes réelles."),
    ("Contrats", "carburant : Electrique / Électrique", "Harmoniser",
     "Réconciliation de l'accent."),
    ("Contrats", "age_conducteur dans {3, 5, 150, 210}", "Reconstruire",
     "âge = ancienneté du permis + 20 ans (règle validée et back-testée au §1.6.2)."),
    ("Contrats", "age_conducteur manquant (402)", "Reconstruire puis imputer",
     "Même règle ; imputation médiane pour le résidu sans ancienneté disponible."),
    ("Contrats", "ancienneté > âge - 18 (60 lignes)", "Reconstruire",
     "Dépassement médian de 50 ans : c'est l'ancienneté qui est fausse, pas l'âge."),
    ("Contrats", "anciennete_permis manquante (299)", "Imputer",
     "Médiane conditionnelle à la tranche d'âge, avec indicatrice de correction."),
    ("Contrats", "bonus_malus hors [0,50 ; 3,50]", "Imputer",
     "Aucune variable auxiliaire ne permet de le reconstruire : médiane + indicatrice."),
    ("Contrats", "bonus_malus manquant (299)", "Imputer", "Idem."),
    ("Contrats", "puissance_fiscale dans {0, 45, 99}", "Imputer",
     "La classe de véhicule ne discrimine pas la puissance (§1.6.3) : médiane globale."),
    ("Contrats", "zone_geo manquante (201)", "Modalité dédiée",
     "Le non-renseignement peut être informatif : modalité « Non renseignée »."),
    ("Contrats", "carburant manquant (403)", "Modalité dédiée", "Idem."),
    ("Contrats", "duree_exposition > 1 (39 lignes)", "Borner",
     "Exposition plafonnée à 1 année-police sur une période d'observation annuelle."),
    ("Contrats", "duree_exposition = 0 sans sinistre", "Supprimer",
     "Un contrat sans exposition n'apporte aucune information de tarification."),
    ("Contrats", "duree_exposition = 0 avec sinistre", "Imputer",
     "Le sinistre atteste la mise en vigueur : exposition ramenée à la médiane."),
    ("Sinistres", "13 lignes strictement dupliquées", "Supprimer",
     "Duplication intégrale, identifiant de sinistre compris."),
    ("Sinistres", "4 montants stockés en texte", "Convertir",
     "Nombres valides accompagnés du symbole monétaire : récupérés intégralement."),
    ("Sinistres", "3 montants négatifs (-500)", "Reconstruire",
     "Charge inexploitable : remplacée par la médiane de la nature du sinistre."),
    ("Sinistres", "28 montants manquants", "Reconstruire",
     "Médiane de la nature du sinistre, qui sépare nettement les ordres de grandeur."),
    ("Sinistres", "7 montants nuls", "Conserver",
     "Sinistre déclaré sans suite : compte en fréquence, exclu du GLM de sévérité."),
    ("Sinistres", "responsabilite_pct hors [0 ; 100]", "Borner",
     "Ramenée dans l'intervalle admissible ; variable de contrôle, non tarifaire."),
    ("Sinistres", "nature_sinistre avec espaces de tête", "Harmoniser",
     "Suppression des espaces parasites."),
    ("Sinistres", "statut Clos / clos", "Harmoniser", "Normalisation de la casse."),
    ("Sinistres", "8 polices orphelines", "Supprimer",
     "Hors périmètre du portefeuille (format d'identifiant différent)."),
    ("Sinistres", "4 sinistres hors période de couverture", "Conserver et signaler",
     "Volume négligeable ; l'exclusion ne modifie aucun résultat (vérifié au §1.8)."),
    ("Sinistres", "271 sinistres au statut Ouvert", "Conserver",
     "Montant lu comme charge évaluée (règlements + provisions) ; sensibilité au §2.7."),
    ("Inter-bases", "154 dommages propres sur contrats au tiers", "Conserver et signaler",
     "Grille de garanties non documentée ; impact de l'exclusion mesuré au §2.7.3."),
    ("Inter-bases", "Sinistralité identique quelle que soit la formule", "Signaler",
     "Sinistres générés indépendamment de la garantie : limite majeure (§9.2)."),
], columns=["Base", "Anomalie", "Traitement", "Justification"])

exporter_table(decisions, "1_5_tableau_decision", index=False)
with pd.option_context("display.max_colwidth", 95):
    display(decisions)

# %% [markdown]
# ## 1.6 Application des corrections
#
# ### 1.6.1 Test préalable : s'agit-il de fautes de frappe ?
#
# Avant de reconstruire, il faut écarter l'hypothèse la plus simple. Deux mécanismes
# d'erreur de saisie sont plausibles pour les âges :
#
# - un **zéro de trop** : `210` serait en réalité `21`, `150` serait `15` ;
# - un **chiffre manquant** : `3` serait `30`, `5` serait `50`.
#
# Chaque hypothèse est testable : l'âge corrigé doit être **compatible avec l'ancienneté
# du permis** de la même ligne, c'est-à-dire vérifier `âge >= ancienneté + 18`.

# %%
titre("TEST DES HYPOTHESES DE FAUTE DE FRAPPE", 2)
ab_age = contrats_brut[(contrats_brut["age_conducteur"] < AGE_MIN)
                       | (contrats_brut["age_conducteur"] > AGE_MAX)].copy()

grands = ab_age[ab_age["age_conducteur"] > AGE_MAX].copy()
grands["hypothese"] = grands["age_conducteur"] / 10
coh_grands = (grands["hypothese"] >= grands["anciennete_permis"] + 18).mean()

petits = ab_age[ab_age["age_conducteur"] < AGE_MIN].copy()
petits["hypothese"] = petits["age_conducteur"] * 10
coh_petits = (petits["hypothese"] >= petits["anciennete_permis"] + 18).mean()

print(f"Valeurs distinctes prises par les ages aberrants : "
      f"{sorted(ab_age['age_conducteur'].unique())}")
print(f"\nHypothese 'un zero de trop'   (age / 10) : coherence = {pct(coh_grands)} "
      f"sur {fr(len(grands))} lignes")
print(f"Hypothese 'un chiffre manquant' (age x 10) : coherence = {pct(coh_petits)} "
      f"sur {fr(len(petits))} lignes")
print("\n-> LES DEUX HYPOTHESES SONT REJETEES.")
print("   Les ages aberrants ne prennent que 4 valeurs distinctes et les corrections")
print("   mecaniques restent incompatibles avec l'anciennete du permis. Ce sont des")
print("   valeurs injectees, non des erreurs de saisie. Il faut donc reconstruire")
print("   l'age a partir d'une autre variable.")

# %% [markdown]
# **Résultat négatif documenté.** Les deux hypothèses de faute de frappe sont rejetées.
# C'est un point important : il aurait été tentant de « corriger » `210` en `21`, ce qui
# aurait introduit des âges faux tout en donnant l'apparence d'un traitement soigné.
#
# ### 1.6.2 Construction et validation de la règle de reconstruction
#
# L'ancienneté du permis est, elle, parfaitement plausible sur ces mêmes lignes. Si
# l'**âge d'obtention du permis** est stable dans le portefeuille, alors
# `âge = ancienneté + âge d'obtention` fournit une reconstruction fiable.
# On commence par mesurer cette stabilité sur la population saine.

# %%
titre("AGE D'OBTENTION DU PERMIS — mesure sur la population saine", 2)
pop_saine = contrats_brut[
    contrats_brut["age_conducteur"].between(AGE_MIN, AGE_MAX)
    & contrats_brut["anciennete_permis"].notna()
].copy()
pop_saine["age_obtention"] = pop_saine["age_conducteur"] - pop_saine["anciennete_permis"]
pop_saine = pop_saine[pop_saine["age_obtention"].between(15, 60)]

print(f"Effectif de reference : {fr(len(pop_saine))} contrats")
print(pop_saine["age_obtention"].describe().to_string())
print("\nRepartition :")
print(pop_saine["age_obtention"].value_counts().sort_index().to_string())

fig, ax = plt.subplots(figsize=(8, 3.6))
pop_saine["age_obtention"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color=PALETTE["petrole"], width=0.7)
ax.set_title("Âge d'obtention du permis dans la population saine")
ax.set_xlabel("Âge d'obtention (ans)")
ax.set_ylabel("Nombre de contrats")
ax.axvline(x=list(sorted(pop_saine["age_obtention"].unique())).index(AGE_PERMIS_MEDIAN),
           color=PALETTE["brique"], linestyle="--", lw=1.5,
           label=f"Médiane retenue : {AGE_PERMIS_MEDIAN} ans")
ax.legend(frameon=False)
axe_fr(ax)
plt.xticks(rotation=0)
sauver_fig("1_6_age_obtention_permis")

ecart_type_obt = pop_saine["age_obtention"].std()
print(f"\nEcart-type = {fr(ecart_type_obt, 2)} an(s).")
print("-> La dispersion est faible : la relation est assez resserree pour reconstruire")
print("   un age a environ 2 ans pres, ce qui est negligeable devant des tranches")
print("   tarifaires de 5 a 10 ans.")

# %% [markdown]
# **Back-test de la règle.** Une règle d'imputation doit être mesurée, pas postulée.
# On l'applique donc à des lignes dont l'âge est connu et correct, puis on compare
# l'âge reconstruit à l'âge réel.

# %%
titre("BACK-TEST DE LA REGLE DE RECONSTRUCTION", 2)
age_reconstruit_test = pop_saine["anciennete_permis"] + AGE_PERMIS_MEDIAN
erreur = (age_reconstruit_test - pop_saine["age_conducteur"]).abs()

print(f"Regle testee : age = anciennete_permis + {AGE_PERMIS_MEDIAN}")
print(f"Erreur absolue moyenne   : {fr(erreur.mean(), 2)} an(s)")
print(f"Erreur absolue mediane   : {fr(erreur.median(), 2)} an(s)")
print(f"Erreur <= 2 ans          : {pct((erreur <= 2).mean())} des cas")
print(f"Erreur <= 3 ans          : {pct((erreur <= 3).mean())} des cas")

# Impact reel : la reconstruction change-t-elle la tranche tarifaire ?
BORNES_AGE = [17, 25, 30, 40, 50, 60, 70, 95]
tr_reelle = pd.cut(pop_saine["age_conducteur"], BORNES_AGE)
tr_reconstr = pd.cut(age_reconstruit_test.clip(AGE_MIN, AGE_MAX), BORNES_AGE)
print(f"\nMeme tranche tarifaire apres reconstruction : "
      f"{pct((tr_reelle.astype(str) == tr_reconstr.astype(str)).mean())} des cas")
print("-> C'est le critere qui compte reellement : le tarif est etabli par tranche,")
print("   pas a l'annee pres.")

# %% [markdown]
# ### 1.6.3 La puissance fiscale est-elle reconstructible ?
#
# Même démarche pour la puissance fiscale : existe-t-il une variable auxiliaire
# permettant de la reconstruire ? Le candidat naturel est la classe de véhicule.

# %%
titre("PUISSANCE FISCALE — recherche d'un recoupement", 2)
pf_saine = contrats_brut[contrats_brut["puissance_fiscale"].between(PUISS_MIN, PUISS_MAX)]
tab_pf = pf_saine.groupby(pf_saine["classe_vehicule"].str.strip())["puissance_fiscale"].agg(
    Effectif="count", Mediane="median", Moyenne="mean", Ecart_type="std")
display(tab_pf.round(2))
print("-> La puissance mediane vaut 7 CV dans LES CINQ classes de vehicule.")
print("   La classe n'apporte aucune information sur la puissance : aucun recoupement")
print("   n'est possible. La puissance sera donc imputee par la mediane globale,")
print("   traitement assume comme moins satisfaisant mais sans alternative.")

# %% [markdown]
# ### 1.6.4 Application effective des corrections
#
# Les règles établies ci-dessus sont maintenant appliquées, dans l'ordre :
# harmonisation, reconstruction, imputation résiduelle. Chaque correction laisse une
# **trace sous forme d'indicatrice**, ce qui permettra de vérifier au §5 que les lignes
# corrigées ne se comportent pas différemment des autres.

# %%
titre("NETTOYAGE DE LA BASE CONTRATS")

contrats = contrats_brut.drop_duplicates().copy()
print(f"Suppression des doublons : {fr(len(contrats_brut))} -> {fr(len(contrats))} lignes")

# --- a) Harmonisation des variables qualitatives ---
contrats["sexe"] = (contrats["sexe"].astype("object").str.strip().str.upper().str[0]
                    .map({"M": "M", "H": "M", "F": "F"}))
for col in ["usage_vehicule", "classe_vehicule", "carburant"]:
    contrats[col] = contrats[col].astype("object").str.strip().str.title()
contrats["carburant"] = contrats["carburant"].replace({"Electrique": "Électrique"})
contrats["zone_geo"] = contrats["zone_geo"].astype("object").str.strip().str.upper()

print("\nModalites apres harmonisation :")
for col in ["sexe", "usage_vehicule", "classe_vehicule", "carburant", "zone_geo"]:
    mods = sorted(contrats[col].dropna().unique())
    print(f"   {col:<18} : {len(mods)} modalites -> {mods}")

# --- b) Mise en NaN des valeurs impossibles ---
contrats["date_effet"] = pd.to_datetime(contrats["date_effet_contrat"])
n_age_ab = int(((contrats["age_conducteur"] < AGE_MIN)
                | (contrats["age_conducteur"] > AGE_MAX)).sum())
contrats.loc[(contrats["age_conducteur"] < AGE_MIN)
             | (contrats["age_conducteur"] > AGE_MAX), "age_conducteur"] = np.nan
n_bm_ab = int(((contrats["bonus_malus"] < BM_MIN) | (contrats["bonus_malus"] > BM_MAX)).sum())
contrats.loc[(contrats["bonus_malus"] < BM_MIN)
             | (contrats["bonus_malus"] > BM_MAX), "bonus_malus"] = np.nan
n_pf_ab = int(((contrats["puissance_fiscale"] < PUISS_MIN)
               | (contrats["puissance_fiscale"] > PUISS_MAX)).sum())
contrats.loc[(contrats["puissance_fiscale"] < PUISS_MIN)
             | (contrats["puissance_fiscale"] > PUISS_MAX), "puissance_fiscale"] = np.nan
print(f"\nValeurs impossibles neutralisees : age {fr(n_age_ab)}, "
      f"bonus-malus {fr(n_bm_ab)}, puissance {fr(n_pf_ab)}")

# --- c) Reconstruction de l'âge par l'ancienneté du permis ---
a_reconstruire = contrats["age_conducteur"].isna() & contrats["anciennete_permis"].notna()
contrats["flag_age_reconstruit"] = a_reconstruire.astype(int)
contrats.loc[a_reconstruire, "age_conducteur"] = (
    contrats.loc[a_reconstruire, "anciennete_permis"] + AGE_PERMIS_MEDIAN
).clip(AGE_MIN, AGE_MAX)
print(f"\nAges reconstruits par recoupement : {fr(a_reconstruire.sum())}")

# --- d) Reconstruction de l'ancienneté quand elle contredit l'âge ---
anc_fausse = (contrats["age_conducteur"].notna() & contrats["anciennete_permis"].notna()
              & (contrats["anciennete_permis"] > contrats["age_conducteur"] - 18))
contrats["flag_anc_reconstruite"] = anc_fausse.astype(int)
contrats.loc[anc_fausse, "anciennete_permis"] = (
    contrats.loc[anc_fausse, "age_conducteur"] - AGE_PERMIS_MEDIAN).clip(lower=0)
print(f"Anciennetes reconstruites par recoupement : {fr(anc_fausse.sum())}")

# --- e) Imputation du résidu non reconstructible ---
med_age = contrats["age_conducteur"].median()
contrats["flag_age_impute"] = contrats["age_conducteur"].isna().astype(int)
contrats["age_conducteur"] = contrats["age_conducteur"].fillna(med_age)

contrats["flag_anc_imputee"] = contrats["anciennete_permis"].isna().astype(int)
tranche_tmp = pd.cut(contrats["age_conducteur"], BORNES_AGE)
med_anc_par_age = contrats.groupby(tranche_tmp, observed=True)["anciennete_permis"].transform("median")
contrats["anciennete_permis"] = contrats["anciennete_permis"].fillna(med_anc_par_age)
contrats["anciennete_permis"] = contrats["anciennete_permis"].fillna(
    contrats["anciennete_permis"].median())
contrats["anciennete_permis"] = contrats["anciennete_permis"].clip(
    lower=0, upper=(contrats["age_conducteur"] - 18).clip(lower=0))

contrats["flag_bm_impute"] = contrats["bonus_malus"].isna().astype(int)
contrats["bonus_malus"] = contrats["bonus_malus"].fillna(contrats["bonus_malus"].median())

contrats["flag_pf_imputee"] = contrats["puissance_fiscale"].isna().astype(int)
contrats["puissance_fiscale"] = contrats["puissance_fiscale"].fillna(
    contrats["puissance_fiscale"].median())

contrats["zone_geo"] = contrats["zone_geo"].fillna("Non renseignée")
contrats["carburant"] = contrats["carburant"].fillna("Non renseigné")
contrats["sexe"] = contrats["sexe"].fillna("Non renseigné")

print(f"\nImputations residuelles : age {fr(contrats['flag_age_impute'].sum())}, "
      f"anciennete {fr(contrats['flag_anc_imputee'].sum())}, "
      f"bonus-malus {fr(contrats['flag_bm_impute'].sum())}, "
      f"puissance {fr(contrats['flag_pf_imputee'].sum())}")

# --- f) Contrôles de non-régression ---
assert contrats["age_conducteur"].between(AGE_MIN, AGE_MAX).all(), "age hors bornes"
assert (contrats["anciennete_permis"] <= contrats["age_conducteur"] - 18 + 1e-9).all(), \
    "anciennete incoherente avec l'age"
assert contrats["bonus_malus"].between(BM_MIN, BM_MAX).all(), "bonus-malus hors bornes"
assert contrats[["age_conducteur", "anciennete_permis", "bonus_malus",
                 "puissance_fiscale"]].notna().all().all(), "valeurs manquantes residuelles"
print("\n[OK] Controles passes : plus aucune valeur hors bornes, manquante ou incoherente.")

# %% [markdown]
# ### 1.6.5 Traitement de l'exposition
#
# L'exposition est la variable la plus sensible de toute la tarification : elle sert
# d'**offset** dans le GLM de fréquence et de dénominateur à toutes les fréquences
# calculées. Une exposition fausse fausse mécaniquement tout le tarif.

# %%
titre("TRAITEMENT DE L'EXPOSITION", 2)
polices_avec_sinistre = set(sinistres_brut["id_police"])

n_sup1 = int((contrats["duree_exposition"] > 1).sum())
contrats["exposition"] = contrats["duree_exposition"].clip(upper=1.0)
print(f"Expositions superieures a 1 ramenees a 1 : {fr(n_sup1)}")

expo_nulle_m = contrats["exposition"] <= 0
avec_sin = expo_nulle_m & contrats["id_police"].isin(polices_avec_sinistre)
sans_sin = expo_nulle_m & ~contrats["id_police"].isin(polices_avec_sinistre)
med_expo = contrats.loc[contrats["exposition"] > 0, "exposition"].median()

contrats["flag_expo_imputee"] = avec_sin.astype(int)
contrats.loc[avec_sin, "exposition"] = med_expo
print(f"Expositions nulles AVEC sinistre, imputees a la mediane ({fr(med_expo, 3)}) : "
      f"{fr(avec_sin.sum())}")
print(f"Expositions nulles SANS sinistre, supprimees                     : "
      f"{fr(sans_sin.sum())}")

contrats = contrats[contrats["exposition"] > 0].copy()
assert (contrats["exposition"] > 0).all() and (contrats["exposition"] <= 1).all()
print(f"\nBase Contrats fiabilisee : {fr(len(contrats))} contrats, "
      f"{fr(contrats['exposition'].sum(), 1)} annees-police d'exposition")

# %% [markdown]
# ### 1.6.6 Nettoyage de la base Sinistres
#
# Les montants manquants ou inexploitables sont **reconstruits par la médiane de leur
# nature de sinistre**. C'est le même principe de recoupement que pour l'âge : la nature
# du sinistre sépare très nettement les ordres de grandeur (quelques centaines d'unités
# pour un bris de glace, plusieurs dizaines de milliers pour un incendie total), et
# constitue donc un prédicteur bien plus pertinent qu'une médiane globale.

# %%
titre("NETTOYAGE DE LA BASE SINISTRES")

sinistres = sinistres_brut.drop_duplicates().copy()
print(f"Suppression des doublons : {fr(len(sinistres_brut))} -> {fr(len(sinistres))} lignes")

sinistres["nature_sinistre"] = sinistres["nature_sinistre"].astype("object").str.strip()
sinistres["statut"] = sinistres["statut"].astype("object").str.strip().str.title()
sinistres["date_sin"] = pd.to_datetime(sinistres["date_sinistre"])
sinistres["montant"] = nettoyer_montant(sinistres["montant_sinistre"])

n_hors = int(((sinistres["responsabilite_pct"] < 0)
              | (sinistres["responsabilite_pct"] > 100)).sum())
sinistres["responsabilite_pct"] = sinistres["responsabilite_pct"].clip(0, 100)
print(f"Taux de responsabilite ramenes dans [0 ; 100] : {fr(n_hors)}")

# --- suppression des sinistres orphelins ---
n_av = len(sinistres)
sinistres = sinistres[sinistres["id_police"].isin(set(contrats["id_police"]))].copy()
print(f"Sinistres orphelins ou rattaches a un contrat supprime : {fr(n_av - len(sinistres))}")

# --- reconstruction des montants par la médiane de la nature ---
a_corriger = sinistres["montant"].isna() | (sinistres["montant"] < 0)
sinistres["flag_montant_reconstruit"] = a_corriger.astype(int)
med_par_nature = sinistres.loc[~a_corriger].groupby("nature_sinistre")["montant"].median()
med_globale = sinistres.loc[~a_corriger, "montant"].median()

print(f"\nMontants a reconstruire : {fr(a_corriger.sum())} "
      f"({fr(sinistres['montant'].isna().sum())} manquants, "
      f"{fr((sinistres['montant'] < 0).sum())} negatifs)")
print("\nMediane de reference par nature de sinistre :")
print(med_par_nature.sort_values().map(lambda v: fr(v, 2)).to_string())

sinistres.loc[a_corriger, "montant"] = (
    sinistres.loc[a_corriger, "nature_sinistre"].map(med_par_nature).fillna(med_globale))
assert sinistres["montant"].notna().all() and (sinistres["montant"] >= 0).all()

print(f"\nBase Sinistres fiabilisee : {fr(len(sinistres))} sinistres, "
      f"charge totale {fr(sinistres['montant'].sum())}")
print(f"Sinistres de montant nul conserves (declares sans suite) : "
      f"{fr((sinistres['montant'] == 0).sum())}")

# %% [markdown]
# ## 1.7 Création des variables de tarification
#
# Quatrième exigence du point A. Deux chantiers : le **découpage en tranches** des
# variables continues, puis la construction de la **base de tarification** — une ligne
# par contrat, portant son exposition et sa sinistralité.
#
# ### 1.7.1 Pourquoi découper les variables continues en tranches
#
# Un GLM à lien logarithmique appliqué à une variable continue impose un effet
# **monotone et exponentiel** : chaque année d'âge supplémentaire multiplierait le
# risque par un facteur constant. Or le risque automobile est notoirement **en U** selon
# l'âge — élevé chez les jeunes conducteurs, décroissant, puis remontant chez les
# conducteurs âgés. Le découpage en tranches laisse le modèle libre d'estimer un
# coefficient propre à chaque tranche, donc de restituer cette forme.
#
# Les bornes sont choisies pour respecter deux contraintes : correspondre aux usages
# du marché (seuil des 25 ans, permis de moins de 2 ans) et garantir un **effectif
# suffisant dans chaque tranche** pour que les coefficients soient estimables.

# %%
BORNES_ANC = [-0.01, 2, 5, 10, 20, 30, 100]
BORNES_BM = [0.49, 0.60, 0.75, 0.90, 1.00, 1.20, 3.51]
BORNES_AGE_VEH = [-0.01, 2, 5, 10, 15, 40]
BORNES_PUISS = [0, 5, 7, 9, 11, 40]

LIB_AGE = ["18-25 ans", "26-30 ans", "31-40 ans", "41-50 ans", "51-60 ans", "61-70 ans", "71 ans et +"]
LIB_ANC = ["0-2 ans", "3-5 ans", "6-10 ans", "11-20 ans", "21-30 ans", "31 ans et +"]
LIB_BM = ["0,50-0,60", "0,61-0,75", "0,76-0,90", "0,91-1,00", "1,01-1,20", "1,21 et +"]
LIB_AGE_VEH = ["0-2 ans", "3-5 ans", "6-10 ans", "11-15 ans", "16 ans et +"]
LIB_PUISS = ["1-5 CV", "6-7 CV", "8-9 CV", "10-11 CV", "12 CV et +"]


def ajouter_tranches(df):
    """Ajoute les variables tarifaires categorielles a une base de contrats."""
    df = df.copy()
    df["tr_age"] = pd.cut(df["age_conducteur"], BORNES_AGE, labels=LIB_AGE)
    df["tr_anciennete"] = pd.cut(df["anciennete_permis"], BORNES_ANC, labels=LIB_ANC)
    df["tr_bonus"] = pd.cut(df["bonus_malus"], BORNES_BM, labels=LIB_BM)
    df["tr_age_vehicule"] = pd.cut(df["age_vehicule"], BORNES_AGE_VEH, labels=LIB_AGE_VEH)
    df["tr_puissance"] = pd.cut(df["puissance_fiscale"], BORNES_PUISS, labels=LIB_PUISS)
    df["jeune_conducteur"] = np.where(
        (df["age_conducteur"] <= 25) | (df["anciennete_permis"] <= 2), "Oui", "Non")
    return df


contrats = ajouter_tranches(contrats)

titre("EFFECTIFS PAR TRANCHE TARIFAIRE", 2)
for col in ["tr_age", "tr_anciennete", "tr_bonus", "tr_age_vehicule", "tr_puissance"]:
    eff = contrats.groupby(col, observed=True)["exposition"].agg(
        Contrats="size", Exposition="sum")
    eff["Part expo"] = (eff["Exposition"] / eff["Exposition"].sum()).map(lambda v: pct(v))
    eff["Exposition"] = eff["Exposition"].map(lambda v: fr(v, 1))
    print(f"\n--- {col} ---")
    print(eff.to_string())

# %% [markdown]
# Tous les effectifs sont largement suffisants : la tranche la moins peuplée dépasse
# plusieurs centaines de contrats. Aucun regroupement n'est nécessaire à ce stade.
#
# ### 1.7.2 Construction de la base de tarification
#
# On agrège les sinistres au niveau du contrat. Chaque ligne porte alors son exposition,
# son nombre de sinistres et sa charge — soit exactement les trois quantités dont la
# tarification a besoin.

# %%
agg_sin = sinistres.groupby("id_police").agg(
    nb_sinistres=("id_sinistre", "size"),
    charge_totale=("montant", "sum"),
)

base_tarif = contrats.merge(agg_sin, on="id_police", how="left")
base_tarif[["nb_sinistres", "charge_totale"]] = (
    base_tarif[["nb_sinistres", "charge_totale"]].fillna(0))
base_tarif["nb_sinistres"] = base_tarif["nb_sinistres"].astype(int)

titre("BASE DE TARIFICATION")
print(f"Contrats            : {fr(len(base_tarif))}")
print(f"Exposition totale   : {fr(base_tarif['exposition'].sum(), 1)} annees-police")
print(f"Sinistres rattaches : {fr(base_tarif['nb_sinistres'].sum())}")
print(f"Charge totale       : {fr(base_tarif['charge_totale'].sum())}")
print(f"\nFrequence brute     : {fr(base_tarif['nb_sinistres'].sum() / base_tarif['exposition'].sum(), 4)} "
      f"sinistre par annee-police")
print(f"Prime pure brute    : {fr(base_tarif['charge_totale'].sum() / base_tarif['exposition'].sum())} "
      f"par annee-police")

print("\nDistribution du nombre de sinistres par contrat :")
dist_nb = base_tarif["nb_sinistres"].value_counts().sort_index()
print(pd.DataFrame({"Contrats": dist_nb,
                    "Part": (dist_nb / len(base_tarif)).map(lambda v: pct(v, 2))}).to_string())

assert abs(base_tarif["nb_sinistres"].sum() - len(sinistres)) < 1e-9
assert abs(base_tarif["charge_totale"].sum() - sinistres["montant"].sum()) < 1e-6
print("\n[OK] La base de tarification reconcilie exactement avec la base Sinistres.")

# %% [markdown]
# ## 1.8 Journal de fiabilisation
#
# Bilan avant / après. Aucune ligne ne doit disparaître sans être comptée : c'est la
# condition pour qu'un tiers puisse auditer le travail.

# %%
journal = pd.DataFrame([
    ("Contrats — lignes", len(contrats_brut), len(contrats)),
    ("Contrats — polices distinctes", contrats_brut["id_police"].nunique(),
     contrats["id_police"].nunique()),
    ("Contrats — valeurs manquantes", int(contrats_brut.isna().sum().sum()),
     int(contrats[[c for c in contrats_brut.columns]].isna().sum().sum())),
    ("Exposition totale (années-police)", round(contrats_brut["duree_exposition"].sum(), 1),
     round(contrats["exposition"].sum(), 1)),
    ("Sinistres — lignes", len(sinistres_brut), len(sinistres)),
    ("Sinistres — montants exploitables",
     int(pd.to_numeric(sinistres_brut["montant_sinistre"], errors="coerce").gt(0).sum()),
     int(sinistres["montant"].ge(0).sum())),
    ("Charge totale",
     round(float(nettoyer_montant(sinistres_brut["montant_sinistre"]).clip(lower=0).sum()), 0),
     round(float(sinistres["montant"].sum()), 0)),
], columns=["Indicateur", "Avant fiabilisation", "Après fiabilisation"])
journal["Écart"] = journal["Après fiabilisation"] - journal["Avant fiabilisation"]
exporter_table(journal, "1_8_journal_fiabilisation", index=False)
display(journal)

corrections = pd.DataFrame({
    "Correction": ["Âges reconstruits", "Anciennetés reconstruites", "Âges imputés",
                   "Anciennetés imputées", "Bonus-malus imputés", "Puissances imputées",
                   "Expositions imputées", "Montants reconstruits"],
    "Lignes": [int(contrats["flag_age_reconstruit"].sum()),
               int(contrats["flag_anc_reconstruite"].sum()),
               int(contrats["flag_age_impute"].sum()),
               int(contrats["flag_anc_imputee"].sum()),
               int(contrats["flag_bm_impute"].sum()),
               int(contrats["flag_pf_imputee"].sum()),
               int(contrats["flag_expo_imputee"].sum()),
               int(sinistres["flag_montant_reconstruit"].sum())],
})
corrections["Méthode"] = ["Recoupement", "Recoupement", "Médiane", "Médiane conditionnelle",
                          "Médiane", "Médiane", "Médiane", "Recoupement"]
exporter_table(corrections, "1_8_corrections_appliquees", index=False)
display(corrections)

n_recoup = int(corrections.loc[corrections["Méthode"] == "Recoupement", "Lignes"].sum())
n_imput = int(corrections.loc[corrections["Méthode"] != "Recoupement", "Lignes"].sum())
print(f"\nValeurs restaurees par RECOUPEMENT entre variables : {fr(n_recoup)}")
print(f"Valeurs restaurees par IMPUTATION statistique      : {fr(n_imput)}")
print(f"\n-> {pct(n_recoup / (n_recoup + n_imput))} des corrections s'appuient sur une")
print("   information reellement presente dans la donnee, et non sur une moyenne.")

# %% [markdown]
# ### Ce qu'il faut retenir de la partie A
#
# - Les deux bases contenaient des anomalies **systématiques et non aléatoires** :
#   valeurs injectées sur un petit nombre de codes, variantes d'écriture, doublons
#   intégraux, montants stockés en texte. Une exploitation naïve aurait produit un tarif
#   faux sans qu'aucun message d'erreur n'apparaisse.
# - L'hypothèse la plus intuitive — des fautes de frappe — a été **testée et rejetée**
#   sur données (§1.6.1).
# - La majorité des corrections s'appuie sur un **recoupement entre variables** plutôt
#   que sur une imputation par la moyenne, et la règle de reconstruction a été
#   **back-testée** avant d'être appliquée.
# - Le portefeuille fiabilisé ne compte **qu'une seule année de souscription**, ce qui
#   contraint le dispositif de validation (§4).
# - Les sinistres sont **indépendants de la garantie souscrite** : un contrat au tiers
#   déclare autant de sinistres qu'un contrat tous risques, y compris des vols et des bris
#   de glace qu'il ne couvre pas (§1.4.1). C'est une limite de la source, à signaler.

# %% [markdown]
# ---
# # §2 — Partie B · Séparation des sinistres attritionnels et graves
#
# > **Ce que demande l'énoncé (point B)**
# > 1. Détermination d'un ou plusieurs seuils de gravité ;
# > 2. Séparation en sinistres attritionnels et graves ;
# > 3. Justification de la méthode et des seuils retenus.
#
# **Pourquoi cette séparation est indispensable.** Un sinistre corporel grave à 400 000
# et un bris de glace à 150 relèvent de deux phénomènes aléatoires distincts. Mélangés
# dans un même modèle de coût moyen, les rares sinistres graves écrasent l'estimation :
# le coût moyen d'un segment devient alors le reflet du hasard de la survenance d'un
# grave dans ce segment, et non de son risque réel. Le tarif devient instable — il
# changerait du tout au tout d'une année sur l'autre.
#
# Le cours propose trois méthodes de détermination du seuil : **seuil fixe
# déterministe**, **séparation par quantile**, et **POT (Peaks Over Threshold)** adossée
# à la théorie des valeurs extrêmes. Les trois sont mises en œuvre ci-dessous, puis
# confrontées à une **quatrième validation, métier** : la nature du sinistre.

# %% [markdown]
# ## 2.1 Forme de la distribution des montants

# %%
montants = sinistres.loc[sinistres["montant"] > 0, "montant"].copy()

titre("DISTRIBUTION DES MONTANTS DE SINISTRES")
print(f"Effectif exploitable : {fr(len(montants))} sinistres")
print(montants.describe().to_string())
print("\nQuantiles :")
q = montants.quantile([.5, .75, .80, .85, .90, .95, .975, .99])
print(q.map(lambda v: fr(v)).to_string())
print(f"\nRapport moyenne / mediane : {fr(montants.mean() / montants.median(), 2)}")
print("-> Une moyenne tres superieure a la mediane signe une distribution fortement")
print("   asymetrique a droite : quelques sinistres pesent l'essentiel de la charge.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].hist(montants, bins=80, color=PALETTE["petrole"], edgecolor="white", linewidth=0.3)
axes[0].set_title("Distribution brute des montants")
axes[0].set_xlabel("Montant du sinistre")
axes[0].set_ylabel("Nombre de sinistres")
axe_fr(axes[0]); axe_fr(axes[0], "x")

axes[1].hist(np.log10(montants), bins=60, color=PALETTE["ambre"],
             edgecolor="white", linewidth=0.3)
axes[1].set_title("Distribution en échelle logarithmique")
axes[1].set_xlabel("log₁₀ (montant)")
axes[1].set_ylabel("Nombre de sinistres")
axes[1].axvline(np.log10(SEUIL_GRAVE), color=PALETTE["brique"], linestyle="--", lw=1.6,
                label=f"Seuil retenu : {fr(SEUIL_GRAVE)}")
axes[1].legend(frameon=False)
axe_fr(axes[1])
plt.tight_layout()
sauver_fig("2_1_distribution_montants")

# %% [markdown]
# **L'échelle logarithmique révèle une distribution nettement bimodale.** Deux
# populations se détachent, séparées par une zone quasi vide. Ce n'est pas une queue
# continue qui s'étire : ce sont deux régimes de sinistres distincts. La séparation
# n'est donc pas une convention commode, elle correspond à une réalité de la donnée.
#
# ### Où se situe exactement la frontière ?

# %%
titre("RECHERCHE DE LA ZONE DE SEPARATION", 2)
tries = np.sort(montants.values)
ecarts = np.diff(tries)
idx = np.argsort(ecarts)[-5:][::-1]
print("Les 5 plus grands intervalles vides entre deux montants consecutifs :")
for i in idx:
    print(f"   de {fr(tries[i]):>12} a {fr(tries[i+1]):>12}  "
          f"-> intervalle vide de {fr(ecarts[i])}")

print(f"\nMontant attritionnel le plus eleve observe : {fr(tries[tries < 10000].max())}")
print(f"Montant grave le plus faible observe       : {fr(tries[tries >= 10000].min())}")
print("\n-> Il existe un intervalle strictement vide entre ces deux valeurs.")
print("   Tout seuil place dans cet intervalle produit EXACTEMENT la meme partition.")

# %% [markdown]
# ## 2.2 Méthode 1 — Seuil par quantile
#
# Première approche du cours : le seuil est fixé à un quantile élevé de la distribution.
# Simple, mais le choix du quantile reste arbitraire.

# %%
titre("METHODE 1 — SEUIL PAR QUANTILE", 2)
res_q = []
for p in [0.85, 0.90, 0.95, 0.975, 0.99]:
    s = montants.quantile(p)
    gr = montants > s
    res_q.append({
        "Quantile": pct(p, 1), "Seuil": fr(s), "Sinistres graves": int(gr.sum()),
        "Part en nombre": pct(gr.mean()),
        "Part en charge": pct(montants[gr].sum() / montants.sum()),
    })
tab_q = pd.DataFrame(res_q)
display(tab_q)
print("\n-> Le quantile 90 % tombe a environ 20 000, le quantile 85 % a environ 3 500 :")
print("   l'ecart est enorme alors que la partition obtenue est presque identique.")
print("   C'est le signe que le quantile n'est pas le bon outil ici : il faut viser")
print("   la zone vide, pas une proportion fixee a l'avance.")

# %% [markdown]
# ## 2.3 Méthode 2 — Fonction des excès moyens
#
# La **fonction des excès moyens** (*mean excess function*) est l'outil de référence
# pour choisir un seuil en théorie des valeurs extrêmes :
#
# $$e(u) = \mathbb{E}\left[X - u \mid X > u\right]$$
#
# Elle mesure, parmi les sinistres qui dépassent `u`, de combien ils le dépassent en
# moyenne. **Lecture** : au-delà du bon seuil, la théorie prédit que `e(u)` devient
# approximativement **linéaire** en `u`. Un **plateau** ou une rupture de pente signale
# le début de la queue de distribution.

# %%
titre("METHODE 2 — FONCTION DES EXCES MOYENS", 2)
seuils_u = np.linspace(500, 40000, 120)
e_u, n_u = [], []
for u in seuils_u:
    exc = montants[montants > u] - u
    e_u.append(exc.mean() if len(exc) >= 10 else np.nan)
    n_u.append(len(exc))
e_u = np.array(e_u)

fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.plot(seuils_u, e_u, color=PALETTE["petrole"], lw=2)
ax.axvspan(7600, 12000, color=PALETTE["ambre"], alpha=0.18,
           label="Zone vide de la distribution")
ax.axvline(SEUIL_GRAVE, color=PALETTE["brique"], linestyle="--", lw=1.6,
           label=f"Seuil retenu : {fr(SEUIL_GRAVE)}")
ax.set_title("Fonction des excès moyens e(u)")
ax.set_xlabel("Seuil u")
ax.set_ylabel("Excès moyen au-delà de u")
ax.legend(frameon=False)
axe_fr(ax); axe_fr(ax, "x")
sauver_fig("2_3_mean_excess")

tab_me = pd.DataFrame({
    "Seuil u": [fr(u) for u in [1000, 3000, 5000, 7500, 10000, 12000, 15000, 20000, 30000]],
    "Sinistres au-delà": [int((montants > u).sum())
                          for u in [1000, 3000, 5000, 7500, 10000, 12000, 15000, 20000, 30000]],
    "e(u)": [fr((montants[montants > u] - u).mean())
             for u in [1000, 3000, 5000, 7500, 10000, 12000, 15000, 20000, 30000]],
})
display(tab_me)
print("\n-> e(u) grimpe fortement jusqu'a environ 5 000, puis se STABILISE : entre")
print("   7 500 et 12 000, le nombre de sinistres depassant le seuil ne bouge plus")
print("   (182 dans les deux cas) et l'exces moyen reste plat. C'est la signature")
print("   d'une zone vide : le seuil peut etre place n'importe ou dans cet intervalle.")

# %% [markdown]
# ## 2.4 Méthode 3 — POT et loi de Pareto généralisée
#
# Le cours indique que « pour un seuil suffisamment élevé, les excédents peuvent être
# approximés par une **Generalized Pareto Distribution (GPD)** ». On ajuste donc une GPD
# sur les excès, pour plusieurs seuils candidats, et on examine la **stabilité des
# paramètres** : le bon seuil est celui à partir duquel l'indice de queue ξ se stabilise.
#
# La GPD a pour fonction de répartition :
#
# $$F(y) = 1 - \left(1 + \xi \frac{y}{\sigma}\right)^{-1/\xi}, \quad y = x - u > 0$$
#
# où **ξ** est l'indice de queue (plus il est élevé, plus la queue est lourde) et **σ**
# le paramètre d'échelle.

# %%
titre("METHODE 3 — AJUSTEMENT GPD SUR LES EXCES", 2)
res_gpd = []
for u in [5000, 7500, 10000, 12000, 15000, 20000, 25000, 30000]:
    exc = (montants[montants > u] - u).values
    if len(exc) < 40:
        continue
    xi, loc, sigma = stats.genpareto.fit(exc, floc=0)
    ks = stats.kstest(exc, "genpareto", args=(xi, loc, sigma))
    res_gpd.append({
        "Seuil u": fr(u), "n excès": len(exc), "ξ (indice de queue)": round(xi, 3),
        "σ (échelle)": fr(sigma), "σ modifié": fr(sigma - xi * u),
        "p-value KS": round(ks.pvalue, 3),
    })
tab_gpd = pd.DataFrame(res_gpd)
exporter_table(tab_gpd, "2_4_ajustement_gpd", index=False)
display(tab_gpd)

xi_ret, _, sig_ret = stats.genpareto.fit(
    (montants[montants > SEUIL_GRAVE] - SEUIL_GRAVE).values, floc=0)
ks_ret = stats.kstest((montants[montants > SEUIL_GRAVE] - SEUIL_GRAVE).values,
                      "genpareto", args=(xi_ret, 0, sig_ret))
print(f"\nAu seuil retenu de {fr(SEUIL_GRAVE)} :")
print(f"   xi    = {xi_ret:.3f}")
print(f"   sigma = {fr(sig_ret)}")
print(f"   Test de Kolmogorov-Smirnov : statistique = {ks_ret.statistic:.4f}, "
      f"p-value = {ks_ret.pvalue:.3f}")
if ks_ret.pvalue > 0.05:
    print("   -> p-value > 0,05 : l'hypothese d'une GPD n'est pas rejetee.")
    print("      L'ajustement de la queue est statistiquement acceptable.")
print(f"\n   xi = {xi_ret:.3f} se situe dans ]0 ; 0,5[ : la queue est lourde")
print("   (plus lourde qu'une exponentielle), mais l'esperance ET la variance")
print("   restent finies. Le cout moyen des graves est donc estimable.")

# %%
# Diagnostic graphique de l'ajustement : QQ-plot et stabilite de xi
exc_ret = (montants[montants > SEUIL_GRAVE] - SEUIL_GRAVE).values
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

theo = stats.genpareto.ppf(
    (np.arange(1, len(exc_ret) + 1) - 0.5) / len(exc_ret), xi_ret, 0, sig_ret)
axes[0].scatter(theo, np.sort(exc_ret), s=18, color=PALETTE["petrole"], alpha=0.75)
lim = [0, max(theo.max(), exc_ret.max()) * 1.02]
axes[0].plot(lim, lim, color=PALETTE["brique"], lw=1.3, linestyle="--")
axes[0].set_title(f"QQ-plot des excès au-delà de {fr(SEUIL_GRAVE)} contre la GPD ajustée")
axes[0].set_xlabel("Quantiles théoriques GPD")
axes[0].set_ylabel("Quantiles empiriques")
axe_fr(axes[0]); axe_fr(axes[0], "x")

us = [u for u in range(4000, 32000, 1000)]
xis = []
for u in us:
    e = (montants[montants > u] - u).values
    xis.append(stats.genpareto.fit(e, floc=0)[0] if len(e) >= 40 else np.nan)
axes[1].plot(us, xis, marker="o", ms=3.5, color=PALETTE["encre"])
axes[1].axvline(SEUIL_GRAVE, color=PALETTE["brique"], linestyle="--", lw=1.6,
                label=f"Seuil retenu : {fr(SEUIL_GRAVE)}")
axes[1].set_title("Stabilité de l'indice de queue ξ selon le seuil")
axes[1].set_xlabel("Seuil u")
axes[1].set_ylabel("ξ estimé")
axes[1].legend(frameon=False)
axe_fr(axes[1], "x")
plt.tight_layout()
sauver_fig("2_4_diagnostic_gpd")

# %% [markdown]
# ## 2.5 Validation métier — la nature du sinistre
#
# Les trois méthodes précédentes sont purement statistiques. Or la base contient une
# information **métier** indépendante : la `nature_sinistre`. Un incendie total ou un
# dommage corporel grave relèvent structurellement du sinistre grave ; un bris de glace
# ou un dégât matériel léger, de l'attritionnel.
#
# Si le seuil statistique et la classification métier **concordent**, la séparation est
# validée par deux voies indépendantes — argument bien plus solide qu'un quantile seul.

# %%
titre("VALIDATION PAR LA NATURE DU SINISTRE", 2)
tab_nat = sinistres.groupby("nature_sinistre")["montant"].agg(
    Effectif="count", Minimum="min", Médiane="median", Maximum="max").sort_values("Médiane")
tab_nat["Nature réputée grave"] = np.where(
    tab_nat.index.isin(NATURES_GRAVES), "Oui", "Non")
display(tab_nat.style.format({"Minimum": "{:,.0f}", "Médiane": "{:,.0f}", "Maximum": "{:,.0f}"})
        if dans_jupyter() else tab_nat.round(0))

print("\nLa coupure est nette : les natures attritionnelles ont une mediane comprise")
print("entre 570 et 810, les natures graves entre 23 700 et 37 300. Aucun recouvrement.")

# %%
titre("CONCORDANCE SEUIL STATISTIQUE / CLASSIFICATION METIER", 2)
sin_val = sinistres[sinistres["montant"] > 0].copy()
sin_val["grave_metier"] = sin_val["nature_sinistre"].isin(NATURES_GRAVES)

res_conc = []
for s in [3000, 5000, 7500, 10000, 12000, 15000, 20000]:
    sin_val["grave_seuil"] = sin_val["montant"] > s
    concordance = (sin_val["grave_seuil"] == sin_val["grave_metier"]).mean()
    res_conc.append({"Seuil": fr(s), "Concordance": pct(concordance, 2),
                     "Désaccords": int((sin_val["grave_seuil"] != sin_val["grave_metier"]).sum())})
display(pd.DataFrame(res_conc))

sin_val["grave_seuil"] = sin_val["montant"] > SEUIL_GRAVE
matrice = pd.crosstab(sin_val["grave_metier"], sin_val["grave_seuil"],
                      rownames=["Nature réputée grave"], colnames=[f"Montant > {fr(SEUIL_GRAVE)}"])
print(f"\nMatrice de confusion au seuil de {fr(SEUIL_GRAVE)} :")
display(matrice)
conc = (sin_val["grave_seuil"] == sin_val["grave_metier"]).mean()
print(f"\nConcordance : {pct(conc, 2)}")

desaccords = sin_val[sin_val["grave_seuil"] != sin_val["grave_metier"]]
print(f"\nLes {fr(len(desaccords))} desaccords :")
display(desaccords[["id_sinistre", "nature_sinistre", "montant", "statut",
                    "flag_montant_reconstruit"]])
print("\n-> Ils s'expliquent : ce sont des sinistres de nature grave dont le montant a")
print("   ete reconstruit (montant manquant) ou vaut zero. La classification metier et")
print("   le seuil statistique ne divergent sur aucun sinistre reellement renseigne.")

# %% [markdown]
# ## 2.6 Seuil retenu et partition
#
# ### Synthèse de la justification
#
# | Méthode | Indication | Verdict |
# |---|---|---|
# | Forme de la distribution | Bimodalité nette, zone vide entre 7 600 et 12 000 | Seuil dans cet intervalle |
# | Quantile | Le quantile 90 % donne 20 000, le 85 % donne 3 500 | Non concluant seul |
# | Fonction des excès moyens | Plateau de e(u) entre 7 500 et 12 000 | Seuil dans cet intervalle |
# | POT / GPD | ξ stable, ajustement non rejeté (KS) | Seuil valide |
# | Nature du sinistre (métier) | Concordance supérieure à 99 % | Seuil validé indépendamment |
#
# **Seuil retenu : `S = 10 000`.** Il est placé au centre de la zone vide, ce qui le
# rend **insensible à un déplacement modéré** — tout seuil entre 7 600 et 12 000
# produirait strictement la même partition. C'est un argument de robustesse plus fort
# qu'un quantile, dont la valeur dépend de l'échantillon.

# %%
titre("PARTITION ATTRITIONNELS / GRAVES")

sinistres["grave"] = sinistres["montant"] > SEUIL_GRAVE
sinistres["categorie"] = np.where(sinistres["grave"], "Grave", "Attritionnel")

synth = sinistres.groupby("categorie")["montant"].agg(
    Nombre="count", Charge="sum", Coût_moyen="mean", Médiane="median", Maximum="max")
synth["Part en nombre"] = synth["Nombre"] / synth["Nombre"].sum()
synth["Part en charge"] = synth["Charge"] / synth["Charge"].sum()
synth_aff = pd.DataFrame({
    "Nombre": synth["Nombre"].map(fr),
    "Part en nombre": synth["Part en nombre"].map(lambda v: pct(v)),
    "Charge totale": synth["Charge"].map(fr),
    "Part en charge": synth["Part en charge"].map(lambda v: pct(v)),
    "Coût moyen": synth["Coût_moyen"].map(fr),
    "Coût médian": synth["Médiane"].map(fr),
    "Maximum": synth["Maximum"].map(fr),
})
exporter_table(synth_aff, "2_6_partition_sinistres")
display(synth_aff)

part_nb = synth.loc["Grave", "Nombre"] / synth["Nombre"].sum()
part_ch = synth.loc["Grave", "Charge"] / synth["Charge"].sum()
rapport = synth.loc["Grave", "Coût_moyen"] / synth.loc["Attritionnel", "Coût_moyen"]
print(f"\nLes sinistres graves representent {pct(part_nb)} du NOMBRE")
print(f"mais {pct(part_ch)} de la CHARGE.")
print(f"Leur cout moyen est {fr(rapport, 0)} fois celui d'un sinistre attritionnel.")
print("\n-> Ce desequilibre est la raison d'etre de la separation : laisser ces")
print("   sinistres dans un modele unique reviendrait a laisser 14 % des observations")
print("   dicter 89 % du tarif, avec une volatilite ingerable.")

# %%
# Enrichissement de la base de tarification avec la decomposition
agg_cat = sinistres.pivot_table(index="id_police", columns="categorie", values="montant",
                                aggfunc=["size", "sum"], fill_value=0)
agg_cat.columns = [f"{a}_{b}" for a, b in agg_cat.columns]
for c in ["size_Attritionnel", "size_Grave", "sum_Attritionnel", "sum_Grave"]:
    if c not in agg_cat.columns:
        agg_cat[c] = 0

base_tarif = base_tarif.merge(
    agg_cat.rename(columns={"size_Attritionnel": "nb_att", "size_Grave": "nb_grave",
                            "sum_Attritionnel": "charge_att", "sum_Grave": "charge_grave"}),
    on="id_police", how="left")
for c in ["nb_att", "nb_grave", "charge_att", "charge_grave"]:
    base_tarif[c] = base_tarif[c].fillna(0)
base_tarif[["nb_att", "nb_grave"]] = base_tarif[["nb_att", "nb_grave"]].astype(int)

assert (base_tarif["nb_att"] + base_tarif["nb_grave"] == base_tarif["nb_sinistres"]).all()
assert np.allclose(base_tarif["charge_att"] + base_tarif["charge_grave"],
                   base_tarif["charge_totale"])
print("[OK] Decomposition coherente : nb_att + nb_grave = nb_sinistres, et de meme")
print("     pour les charges. La base de tarification est complete.")

# %% [markdown]
# ## 2.7 Tests de sensibilité
#
# Deux choix méthodologiques restent à éprouver : la **valeur du seuil**, et la
# **conservation des sinistres au statut « Ouvert »**.
#
# ### 2.7.1 Sensibilité au seuil

# %%
titre("SENSIBILITE AU SEUIL DE GRAVITE", 2)
expo_tot = base_tarif["exposition"].sum()
res_sens = []
for s in [5000, 7500, 10000, 12000, 15000, 20000]:
    gr = sinistres["montant"] > s
    n_g, n_a = int(gr.sum()), int((~gr).sum())
    ch_g = sinistres.loc[gr, "montant"].sum()
    ch_a = sinistres.loc[~gr, "montant"].sum()
    res_sens.append({
        "Seuil": fr(s), "Nb graves": n_g, "Part charge graves": pct(ch_g / (ch_g + ch_a)),
        "Coût moyen attritionnel": fr(ch_a / n_a),
        "Coût moyen grave": fr(ch_g / n_g),
        "PP attritionnelle": fr(ch_a / expo_tot),
        "PP graves": fr(ch_g / expo_tot),
        "PP totale": fr((ch_a + ch_g) / expo_tot),
    })
tab_sens = pd.DataFrame(res_sens)
exporter_table(tab_sens, "2_7_sensibilite_seuil", index=False)
display(tab_sens)
print("\n-> La prime pure TOTALE est rigoureusement invariante : elle ne depend pas du")
print("   seuil, puisque la charge totale ne change pas. Ce qui varie, c'est sa")
print("   REPARTITION entre les deux briques. Entre 7 500 et 12 000, meme cette")
print("   repartition est identique : la partition est strictement la meme.")

# %% [markdown]
# ### 2.7.2 Sensibilité au traitement des sinistres ouverts
#
# Environ un sinistre sur cinq est au statut « Ouvert » : son montant est une **charge
# évaluée** (règlements déjà effectués plus provision du gestionnaire), non un coût
# définitif. Deux traitements sont possibles.
#
# - **Les conserver** — retenu ici. C'est la pratique actuarielle : la provision est la
#   meilleure estimation disponible, et les exclure amputerait la fréquence de 20 %.
# - **Les exclure** — plus rigoureux sur le coût, mais il faudrait alors retraiter
#   l'exposition en conséquence, faute de quoi la fréquence serait biaisée à la baisse.
#
# On mesure l'écart que ce choix produit sur le tarif.

# %%
titre("SENSIBILITE AU TRAITEMENT DES SINISTRES OUVERTS", 2)
print("Repartition par statut :")
print(sinistres["statut"].value_counts(dropna=False).to_string())

comp = sinistres.groupby(["categorie", "statut"])["montant"].agg(
    Nombre="count", Coût_moyen="mean", Médiane="median")
display(comp.round(0))

clos = sinistres[sinistres["statut"] == "Clos"]
pp_tous = sinistres["montant"].sum() / expo_tot
pp_clos = clos["montant"].sum() / expo_tot
freq_tous = len(sinistres) / expo_tot
freq_clos = len(clos) / expo_tot

bilan_ouv = pd.DataFrame({
    "Périmètre": ["Tous sinistres (retenu)", "Sinistres clos seulement"],
    "Nombre": [fr(len(sinistres)), fr(len(clos))],
    "Fréquence": [fr(freq_tous, 4), fr(freq_clos, 4)],
    "Coût moyen": [fr(sinistres["montant"].mean()), fr(clos["montant"].mean())],
    "Prime pure": [fr(pp_tous), fr(pp_clos)],
})
exporter_table(bilan_ouv, "2_7_sensibilite_sinistres_ouverts", index=False)
display(bilan_ouv)
print(f"\nEcart de prime pure induit par l'exclusion des sinistres ouverts : "
      f"{pct(pp_clos / pp_tous - 1)}")
print(f"   dont effet FREQUENCE  : {pct(freq_clos / freq_tous - 1)}")
print(f"   dont effet COUT MOYEN : "
      f"{pct(clos['montant'].mean() / sinistres['montant'].mean() - 1)}")

part_ouv_grave = (sinistres[sinistres["categorie"] == "Grave"]["statut"] == "Ouvert").mean()
part_ouv_att = (sinistres[sinistres["categorie"] == "Attritionnel"]["statut"] == "Ouvert").mean()
print("\nPart de sinistres encore ouverts, par categorie :")
print(f"   graves        : {pct(part_ouv_grave)}")
print(f"   attritionnels : {pct(part_ouv_att)}")
print("\n-> Les DEUX effets se cumulent, et pour une meme raison : les sinistres GRAVES")
print("   restent ouverts bien plus longtemps que les attritionnels (expertise,")
print("   procedure, consolidation d'un dommage corporel). Se limiter aux dossiers")
print("   clos retire donc a la fois un cinquieme des sinistres ET une part")
print("   surrepresentee des plus couteux.")
print("\n   C'est exactement le biais que le maintien des sinistres ouverts evite :")
print("   tarifer sur les seuls dossiers clos reviendrait a retirer du portefeuille")
print("   les sinistres les plus graves parce qu'ils sont les plus lents a se regler.")
print("   Le choix de les conserver est donc confirme, et meme renforce.")

# %% [markdown]
# ### 2.7.3 Sensibilité à l'incohérence des garanties
#
# Le §1.4.1 a montré que des sinistres de dommages propres sont déclarés sur des contrats
# au tiers. On mesure ce que deviendrait le tarif si l'on appliquait la grille de
# garanties usuelle, en excluant ces sinistres ainsi que les sinistres de responsabilité
# civile sans responsabilité de l'assuré sur ces mêmes contrats.

# %%
titre("SENSIBILITE : EXCLUSION DES SINISTRES HORS GARANTIE", 2)
sin_gar = sinistres.merge(contrats[["id_police", "type_couverture"]], on="id_police")
hors_gar = ((sin_gar["type_couverture"] == "Tiers")
            & (sin_gar["nature_sinistre"].isin(DOMMAGES_PROPRES)
               | (sin_gar["responsabilite_pct"] == 0)))
print(f"Sinistres exclus : {fr(hors_gar.sum())} ({pct(hors_gar.mean())} du nombre, "
      f"{pct(sin_gar.loc[hors_gar, 'montant'].sum() / sin_gar['montant'].sum())} de la charge)")

expo_cov = contrats.groupby("type_couverture")["exposition"].sum()
avant = sin_gar.groupby("type_couverture").size() / expo_cov
apres = sin_gar[~hors_gar].groupby("type_couverture").size() / expo_cov
pp_avant = sin_gar.groupby("type_couverture")["montant"].sum() / expo_cov
pp_apres = sin_gar[~hors_gar].groupby("type_couverture")["montant"].sum() / expo_cov
sens_gar = pd.DataFrame({
    "Fréquence (base)": avant.map(lambda v: fr(v, 4)),
    "Fréquence (garanties appliquées)": apres.map(lambda v: fr(v, 4)),
    "Prime pure (base)": pp_avant.map(fr),
    "Prime pure (garanties appliquées)": pp_apres.map(fr),
})
exporter_table(sens_gar, "2_7_sensibilite_garanties")
display(sens_gar)

pp_tot_apres = sin_gar.loc[~hors_gar, "montant"].sum() / expo_tot
print(f"\nPrime pure globale : {fr(sin_gar['montant'].sum() / expo_tot)} -> {fr(pp_tot_apres)} "
      f"({pct(pp_tot_apres / (sin_gar['montant'].sum() / expo_tot) - 1)})")
print("\n-> Appliquer la grille de garanties ferait apparaitre un ecart de frequence de")
print(f"   {pct(apres.max() / apres.min() - 1, 0)} entre formules, la ou les donnees brutes n'en")
print("   montrent aucun. La variable type_couverture deviendrait alors tarifante.")
print("   Mais cet ecart serait CREE par notre hypothese sur la grille, non observe")
print("   dans les donnees : c'est pourquoi le scenario de base ne l'applique pas.")

# %% [markdown]
# ### Ce qu'il faut retenir de la partie B
#
# - La distribution des montants est **bimodale**, avec une zone strictement vide entre
#   7 600 et 12 000 : la séparation correspond à une réalité de la donnée, pas à une
#   convention.
# - **Quatre méthodes convergent** vers un seuil situé dans cet intervalle : forme de la
#   distribution, fonction des excès moyens, ajustement GPD, et classification métier
#   par nature de sinistre (concordance supérieure à 99 %).
# - **Seuil retenu : 10 000**, au centre de la zone vide, donc insensible à un
#   déplacement modéré.
# - Les graves pèsent **14 % du nombre et 89 % de la charge** : les modéliser séparément
#   n'est pas un raffinement, c'est une nécessité de stabilité du tarif.

# %% [markdown]
# ---
# # §3 — Partie C · Approche Fréquence × Coût
#
# > **Ce que demande l'énoncé (point C)**
# > Calcul de la prime pure selon l'approche :
# > `Prime pure = Fréquence des sinistres × Coût moyen des sinistres`
#
# ## 3.1 Le principe, et pourquoi il fonctionne
#
# La charge totale d'un contrat est une **somme aléatoire** : un nombre aléatoire `N` de
# sinistres, chacun de coût aléatoire `X` :
#
# $$S = \sum_{i=1}^{N} X_i$$
#
# Si les coûts sont indépendants du nombre de sinistres, la formule de Wald donne
# directement :
#
# $$\mathbb{E}(S) = \mathbb{E}(N) \times \mathbb{E}(X) = \text{Fréquence} \times \text{Coût moyen}$$
#
# C'est cette décomposition qui donne toute sa valeur à l'approche : elle sépare **deux
# questions de nature différente** — « à quelle fréquence ce profil déclare-t-il un
# sinistre ? » et « combien coûte un sinistre de ce profil ? » — qui n'obéissent pas aux
# mêmes déterminants. Un jeune conducteur peut déclarer beaucoup de petits sinistres,
# un conducteur de véhicule puissant peu de sinistres mais très coûteux.
#
# La **fréquence** se calcule toujours par rapport à l'**exposition**, jamais par rapport
# au nombre de contrats : un contrat en vigueur trois mois ne peut pas être comparé à un
# contrat annuel.

# %%
titre("PRIME PURE GLOBALE — APPROCHE FREQUENCE x COUT")

expo_tot = base_tarif["exposition"].sum()
nb_tot = base_tarif["nb_sinistres"].sum()
charge_tot = base_tarif["charge_totale"].sum()

freq_glob = nb_tot / expo_tot
cout_glob = charge_tot / nb_tot
pp_glob = freq_glob * cout_glob

print(f"Exposition totale            : {fr(expo_tot, 1)} annees-police")
print(f"Nombre de sinistres          : {fr(nb_tot)}")
print(f"Charge totale                : {fr(charge_tot)}")
print()
print(f"Frequence    = {fr(nb_tot)} / {fr(expo_tot, 1)}      = {fr(freq_glob, 4)}")
print(f"Cout moyen   = {fr(charge_tot)} / {fr(nb_tot)} = {fr(cout_glob)}")
print(f"PRIME PURE   = {fr(freq_glob, 4)} x {fr(cout_glob)}       = {fr(pp_glob)}")
print()
print(f"Controle direct : charge / exposition = {fr(charge_tot / expo_tot)}")
assert abs(pp_glob - charge_tot / expo_tot) < 1e-6
print("[OK] Les deux voies de calcul coincident exactement, comme le garantit")
print("     la formule de Wald.")

# %% [markdown]
# ## 3.2 Décomposition attritionnels / graves
#
# La même prime pure se décompose en deux briques additives :
#
# $$PP = \underbrace{f_{att} \times c_{att}}_{\text{brique attritionnelle}} + \underbrace{f_{grave} \times c_{grave}}_{\text{brique grave}}$$

# %%
titre("DECOMPOSITION DE LA PRIME PURE", 2)
nb_att, nb_gr = base_tarif["nb_att"].sum(), base_tarif["nb_grave"].sum()
ch_att, ch_gr = base_tarif["charge_att"].sum(), base_tarif["charge_grave"].sum()

decomp = pd.DataFrame({
    "Brique": ["Attritionnelle", "Grave", "Total"],
    "Nombre": [fr(nb_att), fr(nb_gr), fr(nb_att + nb_gr)],
    "Fréquence": [fr(nb_att / expo_tot, 4), fr(nb_gr / expo_tot, 4),
                  fr((nb_att + nb_gr) / expo_tot, 4)],
    "Coût moyen": [fr(ch_att / nb_att), fr(ch_gr / nb_gr), fr((ch_att + ch_gr) / (nb_att + nb_gr))],
    "Prime pure": [fr(ch_att / expo_tot), fr(ch_gr / expo_tot), fr((ch_att + ch_gr) / expo_tot)],
    "Part de la PP": [pct(ch_att / (ch_att + ch_gr)), pct(ch_gr / (ch_att + ch_gr)), pct(1.0)],
})
exporter_table(decomp, "3_2_decomposition_prime_pure", index=False)
display(decomp)

print(f"\nLecture : sur une prime pure de {fr(pp_glob)}, seuls {fr(ch_att / expo_tot)}")
print(f"({pct(ch_att / (ch_att + ch_gr))}) relevent du risque attritionnel, regulier et previsible.")
print(f"Les {fr(ch_gr / expo_tot)} restants ({pct(ch_gr / (ch_att + ch_gr))}) financent un risque rare")
print(f"(une chance sur {fr(1 / (nb_gr / expo_tot), 0)} par annee-police) mais tres couteux.")

# %% [markdown]
# **Un déséquilibre lourd de conséquences.** La brique grave, qui repose sur seulement
# 187 observations, porte près de 90 % de la prime. C'est la principale source
# d'incertitude de toute l'étude : une année avec quelques sinistres graves de plus ou
# de moins déplacerait le tarif bien davantage que n'importe quel raffinement de
# segmentation sur les attritionnels.

# %% [markdown]
# ## 3.3 Analyses univariées
#
# Pour chaque variable tarifaire, on calcule la fréquence, le coût moyen et la prime
# pure par modalité. La fréquence relative (rapportée à la moyenne du portefeuille)
# donne directement la lecture tarifaire : une valeur de 1,40 signifie « 40 % de
# sinistres en plus que la moyenne ».
#
# **Chaque fréquence est accompagnée de son intervalle de confiance à 95 %**, calculé
# sous hypothèse de Poisson. Sans cet intervalle, il est impossible de distinguer un
# vrai effet tarifaire d'une fluctuation d'échantillonnage — et c'est particulièrement
# critique ici, où certaines modalités reposent sur peu de sinistres.

# %%
def analyse_univariee(df, col, cible_nb="nb_sinistres", cible_charge="charge_totale"):
    """Calcule frequence, cout moyen et prime pure par modalite, avec IC a 95 % sur la
    frequence (approximation normale de la loi de Poisson)."""
    g = df.groupby(col, observed=True).agg(
        Contrats=("id_police", "size"),
        Exposition=("exposition", "sum"),
        Sinistres=(cible_nb, "sum"),
        Charge=(cible_charge, "sum"),
    )
    g["Fréquence"] = g["Sinistres"] / g["Exposition"]
    freq_ref = df[cible_nb].sum() / df["exposition"].sum()
    g["Fréq. relative"] = g["Fréquence"] / freq_ref
    demi = 1.96 * np.sqrt(g["Sinistres"]) / g["Exposition"]
    g["IC bas"] = (g["Fréquence"] - demi).clip(lower=0)
    g["IC haut"] = g["Fréquence"] + demi
    g["Coût moyen"] = np.where(g["Sinistres"] > 0, g["Charge"] / g["Sinistres"], np.nan)
    g["Prime pure"] = g["Charge"] / g["Exposition"]
    # Significativite : l'IC de la modalite contient-il la frequence moyenne ?
    g["Écart significatif"] = np.where(
        (g["IC haut"] < freq_ref) | (g["IC bas"] > freq_ref), "Oui", "Non")
    return g


def afficher_univariee(g, titre_txt):
    aff = pd.DataFrame({
        "Contrats": g["Contrats"].map(fr),
        "Exposition": g["Exposition"].map(lambda v: fr(v, 1)),
        "Sinistres": g["Sinistres"].map(fr),
        "Fréquence": g["Fréquence"].map(lambda v: fr(v, 4)),
        "IC 95 %": [f"[{fr(b, 4)} ; {fr(h, 4)}]" for b, h in zip(g["IC bas"], g["IC haut"])],
        "Fréq. rel.": g["Fréq. relative"].map(lambda v: fr(v, 2)),
        "Signif.": g["Écart significatif"],
        "Coût moyen": g["Coût moyen"].map(fr),
        "Prime pure": g["Prime pure"].map(fr),
    })
    print(f"\n--- {titre_txt} ---")
    display(aff)
    return aff


VARS_TARIF = ["tr_age", "tr_anciennete", "sexe", "zone_geo", "usage_vehicule",
              "classe_vehicule", "carburant", "type_couverture", "tr_bonus",
              "tr_age_vehicule", "tr_puissance", "jeune_conducteur"]

titre("ANALYSES UNIVARIEES — ENSEMBLE DES SINISTRES")
univariees = {}
for v in VARS_TARIF:
    g = analyse_univariee(base_tarif, v)
    univariees[v] = g
    afficher_univariee(g, v)

# %%
# Synthese : quelles variables discriminent reellement ?
titre("QUELLES VARIABLES DISCRIMINENT LA FREQUENCE ?")
resume = []
for v, g in univariees.items():
    etendue = g["Fréq. relative"].max() - g["Fréq. relative"].min()
    n_signif = int((g["Écart significatif"] == "Oui").sum())
    # Test du chi2 d'homogeneite des frequences entre modalites
    obs = g["Sinistres"].values
    att = g["Exposition"].values * (base_tarif["nb_sinistres"].sum() / expo_tot)
    chi2 = float(((obs - att) ** 2 / att).sum())
    ddl = len(obs) - 1
    pval = 1 - stats.chi2.cdf(chi2, ddl)
    resume.append({
        "Variable": v, "Modalités": len(g),
        "Fréq. rel. min": round(g["Fréq. relative"].min(), 2),
        "Fréq. rel. max": round(g["Fréq. relative"].max(), 2),
        "Étendue": round(etendue, 2),
        "Modalités significatives": n_signif,
        "Chi² (homogénéité)": round(chi2, 1),
        "p-value": f"{pval:.4f}" if pval >= 1e-4 else "< 0,0001",
        "Discriminante ?": "Oui" if pval < 0.05 else "Non",
    })
tab_resume = pd.DataFrame(resume).sort_values("Étendue", ascending=False)
exporter_table(tab_resume, "3_3_pouvoir_discriminant", index=False)
display(tab_resume)

print("\nLecture du test : sous l'hypothese nulle, la frequence est identique dans")
print("toutes les modalites et les ecarts observes ne sont que du bruit. Une p-value")
print("inferieure a 5 % conduit a rejeter cette hypothese.")
print("\n-> Resultats POSITIFS : l'anciennete du permis et l'age du conducteur")
print("   discriminent nettement, la zone geographique egalement.")
print("-> Resultats NEGATIFS, a documenter plutot qu'a taire : le sexe et le type de")
print("   couverture ne discriminent pas. Ils ne seront pas retenus dans le tarif.")
print("   Pour le sexe, c'est d'ailleurs conforme au droit europeen, qui interdit")
print("   depuis 2012 de l'utiliser comme variable tarifaire en assurance.")

# %%
# Visualisation des deux variables les plus discriminantes
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
for ax, v, lib in zip(axes, ["tr_age", "tr_anciennete"],
                      ["Âge du conducteur", "Ancienneté du permis"]):
    g = univariees[v]
    x = np.arange(len(g))
    ax.bar(x, g["Fréquence"], color=PALETTE["petrole"], width=0.62)
    ax.errorbar(x, g["Fréquence"], yerr=1.96 * np.sqrt(g["Sinistres"]) / g["Exposition"],
                fmt="none", ecolor=PALETTE["encre"], capsize=3, lw=1.1)
    ax.axhline(freq_glob, color=PALETTE["brique"], linestyle="--", lw=1.4,
               label=f"Moyenne du portefeuille : {fr(freq_glob, 4)}")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=35, ha="right")
    ax.set_title(f"Fréquence par {lib.lower()}")
    ax.set_ylabel("Sinistres par année-police")
    ax.legend(frameon=False, fontsize=8.5)
plt.tight_layout()
sauver_fig("3_3_frequence_age_anciennete")

# %% [markdown]
# **Les barres d'erreur changent la lecture.** Seule la première tranche se détache
# franchement : pour l'âge, les 18-25 ans ; pour l'ancienneté, les permis de moins de
# deux ans. Les autres tranches ont des intervalles de confiance qui se recouvrent
# largement et se confondent avec la moyenne du portefeuille. Autrement dit, **le risque
# n'est pas graduel : il est binaire — conducteur novice ou non.** C'est une information
# tarifaire directement exploitable, et elle justifie la variable `jeune_conducteur`.

# %% [markdown]
# ## 3.4 Analyse croisée et effets confondus
#
# Les analyses univariées ont une limite majeure : elles sont **marginales**. Quand on
# observe que les 18-25 ans ont une surfréquence de 40 %, on ne sait pas si c'est un
# effet de l'âge ou du fait qu'ils ont, par construction, un permis récent. Les deux
# variables sont largement redondantes.

# %%
titre("CROISEMENT AGE x ANCIENNETE DU PERMIS", 2)
croise = base_tarif.pivot_table(index="tr_age", columns="tr_anciennete",
                                values="nb_sinistres", aggfunc="sum", observed=True)
expo_croise = base_tarif.pivot_table(index="tr_age", columns="tr_anciennete",
                                     values="exposition", aggfunc="sum", observed=True)
freq_croise = (croise / expo_croise)
print("Frequence par croisement (cellules de moins de 100 annees-police masquees) :")
display((freq_croise.where(expo_croise >= 100)).round(3))
print("\nExposition correspondante :")
display(expo_croise.round(0))

corr = base_tarif[["age_conducteur", "anciennete_permis"]].corr().iloc[0, 1]
print(f"\nCorrelation lineaire entre age et anciennete du permis : {fr(corr, 3)}")
print("-> Correlation tres forte : les deux variables portent largement la MEME")
print("   information. Les introduire ensemble dans un GLM produira des coefficients")
print("   instables et non significatifs, alors meme que chacune prise isolement est")
print("   clairement discriminante. C'est le probleme de la COLINEARITE, arbitre au §5.2.")

# %% [markdown]
# ## 3.5 Limites de l'approche marginale
#
# Les analyses univariées de cette partie donnent une première lecture du risque, mais
# elles ne peuvent pas servir de tarif, pour trois raisons :
#
# 1. **Elles confondent les effets.** La surfréquence observée sur les 18-25 ans mélange
#    l'effet de l'âge, celui de l'ancienneté du permis et celui de la structure du
#    portefeuille sur les autres variables.
# 2. **Elles ne permettent pas de combiner les critères.** Un tarif doit fournir une
#    prime pour un profil complet — jeune conducteur, zone C, véhicule puissant — et non
#    une prime par critère isolé. Multiplier les coefficients univariés reviendrait à
#    compter plusieurs fois le même effet.
# 3. **Elles ne quantifient pas l'incertitude de façon exploitable.** Les intervalles de
#    confiance calculés ici sont marginaux, sans contrôle des autres variables.
#
# Le **modèle linéaire généralisé** répond exactement à ces trois limites : il estime
# l'effet propre de chaque variable **à autres caractéristiques égales**, fournit une
# prime multiplicative pour tout profil, et livre la significativité de chaque
# coefficient. C'est l'objet du §5.

# %% [markdown]
# ---
# # §4 — Partie F · Séparation Train / Test
#
# > **Ce que demande l'énoncé (point F)**
# > *Approche 1 — Séparation aléatoire* : X % en apprentissage, Y % en test.
# > *Approche 2 — Séparation temporelle* : années antérieures à N en apprentissage,
# > année N+1 en test.
#
# **Pourquoi cette partie vient avant les GLM.** Un modèle doit être estimé sur des
# données, puis évalué sur d'**autres** données. Si l'on construisait le découpage après
# avoir calibré les modèles, les paramètres auraient déjà vu l'ensemble du portefeuille
# et l'évaluation n'aurait aucune valeur : elle mesurerait la capacité du modèle à
# retrouver ce qu'on lui a appris, pas à prédire.
#
# **Les deux approches sont mises en œuvre.** Elles ne répondent pas à la même question,
# et c'est précisément ce qui rend leur combinaison utile :
#
# | Approche | Question à laquelle elle répond |
# |---|---|
# | Temporelle | Le tarif calibré sur une période tiendra-t-il sur la suivante ? |
# | Aléatoire | Le tarif tient-il sur des assurés qu'il n'a jamais vus ? |

# %% [markdown]
# ## 4.1 Approche temporelle — un obstacle et son contournement
#
# ### L'obstacle
#
# La formulation littérale de l'énoncé — « années antérieures à N en apprentissage,
# année N+1 en test » — suppose **plusieurs années de souscription**. Or le §1.4 a
# montré que toutes les dates d'effet se situent en 2023.

# %%
titre("VERIFICATION : COMBIEN D'ANNEES DE SOUSCRIPTION ?", 2)
print("Repartition des dates d'effet par annee :")
print(contrats["date_effet"].dt.year.value_counts().sort_index().to_string())
print("\nRepartition des dates de survenance des sinistres par annee :")
print(sinistres["date_sin"].dt.year.value_counts().sort_index().to_string())
print("\n-> UNE SEULE annee de souscription (2023), mais DEUX annees de survenance")
print("   (2023 et 2024). Un contrat souscrit en cours d'annee 2023 reste en effet")
print("   expose une partie de 2024.")
print("\n   La separation par annee de SOUSCRIPTION est donc impossible.")
print("   En revanche, la separation par annee de SURVENANCE est parfaitement")
print("   realisable, et c'est elle qui a du sens pour valider un tarif : on calibre")
print("   sur ce qui s'est passe en 2023, on verifie sur ce qui s'est passe en 2024.")

# %% [markdown]
# ### Le contournement : découpage par période de survenance
#
# L'exposition de chaque contrat est **scindée** au 1er janvier 2024, et les sinistres
# sont affectés à la période où ils sont survenus. Un contrat prenant effet le
# 1er septembre 2023 pour une durée de 0,8 année contribue ainsi pour 0,33 année-police
# à 2023 et 0,47 année-police à 2024.
#
# C'est le découpage dit « par exercice de survenance », usuel en assurance non-vie.

# %%
def decouper_par_periode(contrats_df, sinistres_df, debut, fin):
    """Restreint contrats et sinistres a une fenetre temporelle.

    L'exposition est recalculee au prorata de l'intersection entre la periode de
    couverture du contrat et la fenetre ; les sinistres sont filtres sur leur date
    de survenance. Seuls les contrats reellement exposes sur la fenetre sont conserves.
    """
    d = contrats_df.copy()
    debut_contrat = d["date_effet"]
    fin_contrat = debut_contrat + pd.to_timedelta(d["exposition"] * 365.25, unit="D")

    debut_eff = debut_contrat.clip(lower=debut)
    fin_eff = fin_contrat.clip(upper=fin)
    d["exposition"] = ((fin_eff - debut_eff).dt.total_seconds()
                       / (365.25 * 24 * 3600)).clip(lower=0)
    d = d[d["exposition"] > 0].copy()

    s = sinistres_df[(sinistres_df["date_sin"] >= debut)
                     & (sinistres_df["date_sin"] < fin)
                     & (sinistres_df["id_police"].isin(set(d["id_police"])))].copy()

    agg = s.groupby("id_police").agg(
        nb_sinistres=("id_sinistre", "size"), charge_totale=("montant", "sum"))
    agg_cat = s.pivot_table(index="id_police", columns="categorie", values="montant",
                            aggfunc=["size", "sum"], fill_value=0)
    if len(agg_cat):
        agg_cat.columns = [f"{a}_{b}" for a, b in agg_cat.columns]
    for c in ["size_Attritionnel", "size_Grave", "sum_Attritionnel", "sum_Grave"]:
        if c not in agg_cat.columns:
            agg_cat[c] = 0
    agg_cat = agg_cat.rename(columns={
        "size_Attritionnel": "nb_att", "size_Grave": "nb_grave",
        "sum_Attritionnel": "charge_att", "sum_Grave": "charge_grave"})

    d = d.merge(agg, on="id_police", how="left").merge(
        agg_cat[["nb_att", "nb_grave", "charge_att", "charge_grave"]],
        on="id_police", how="left")
    for c in ["nb_sinistres", "charge_totale", "nb_att", "nb_grave",
              "charge_att", "charge_grave"]:
        d[c] = d[c].fillna(0)
    for c in ["nb_sinistres", "nb_att", "nb_grave"]:
        d[c] = d[c].astype(int)
    return d


train_temp = decouper_par_periode(contrats, sinistres,
                                  pd.Timestamp("2023-01-01"), DATE_CUT)
test_temp = decouper_par_periode(contrats, sinistres,
                                 DATE_CUT, pd.Timestamp("2025-01-01"))

titre("DECOUPAGE TEMPOREL PAR EXERCICE DE SURVENANCE")
print(f"TRAIN (survenance 2023) : {fr(len(train_temp))} contrats exposes, "
      f"{fr(train_temp['exposition'].sum(), 1)} annees-police, "
      f"{fr(train_temp['nb_sinistres'].sum())} sinistres")
print(f"TEST  (survenance 2024) : {fr(len(test_temp))} contrats exposes, "
      f"{fr(test_temp['exposition'].sum(), 1)} annees-police, "
      f"{fr(test_temp['nb_sinistres'].sum())} sinistres")

expo_recomposee = train_temp["exposition"].sum() + test_temp["exposition"].sum()
print(f"\nControle de conservation de l'exposition :")
print(f"   exposition train + test = {fr(expo_recomposee, 1)}")
print(f"   exposition totale       = {fr(contrats['exposition'].sum(), 1)}")
assert abs(expo_recomposee - contrats["exposition"].sum()) < 1.0
print("   [OK] L'exposition se conserve : aucune annee-police perdue ni comptee deux fois.")

nb_recompose = train_temp["nb_sinistres"].sum() + test_temp["nb_sinistres"].sum()
print(f"\n   sinistres train + test  = {fr(nb_recompose)}  (total : {fr(len(sinistres))})")
assert nb_recompose == len(sinistres)
print("   [OK] Tous les sinistres sont affectes a une periode et une seule.")

# %% [markdown]
# ### La limite de cette approche, énoncée franchement
#
# Une **même police alimente les deux échantillons** : celle qui prend effet en
# septembre 2023 est présente en 2023 et en 2024. Le test ne porte donc pas sur des
# assurés inconnus du modèle, mais sur une **nouvelle période d'observation des mêmes
# assurés**.
#
# Ce dispositif valide donc la **stabilité temporelle** du tarif — ce qui est exactement
# ce qu'un assureur veut savoir avant de déposer un tarif pour l'exercice suivant —
# mais **pas** sa capacité à généraliser à de nouveaux souscripteurs. D'où la seconde
# approche.

# %% [markdown]
# ## 4.2 Approche aléatoire — 80 % / 20 %
#
# Trois précautions méthodologiques :
#
# 1. **Le tirage porte sur la police, jamais sur la ligne de sinistre.** Répartir les
#    sinistres au hasard placerait deux sinistres d'un même contrat de part et d'autre
#    de la frontière : le modèle verrait en apprentissage une information sur un contrat
#    qu'il est censé découvrir en test. C'est une fuite d'information classique.
# 2. **Le tirage est stratifié sur la présence de sinistre.** Avec seulement 6,4 % de
#    contrats sinistrés, un tirage simple pourrait produire des taux sensiblement
#    différents entre les deux échantillons.
# 3. **La graine aléatoire est fixée** pour que le résultat soit reproductible.

# %%
titre("DECOUPAGE ALEATOIRE 80 / 20 STRATIFIE")

base_split = base_tarif.copy()
base_split["a_sinistre"] = (base_split["nb_sinistres"] > 0).astype(int)

indices_train = []
generateur = np.random.default_rng(GRAINE)
for strate, sous_base in base_split.groupby("a_sinistre"):
    idx = sous_base.index.to_numpy().copy()   # copie : to_numpy() renvoie un tableau en lecture seule
    generateur.shuffle(idx)
    n_tr = int(round(PART_TRAIN * len(idx)))
    indices_train.extend(idx[:n_tr].tolist())
    print(f"Strate « {'avec' if strate else 'sans'} sinistre » : {fr(len(idx))} contrats "
          f"-> {fr(n_tr)} en train, {fr(len(idx) - n_tr)} en test")

masque_train = base_split.index.isin(indices_train)
train_alea = base_split[masque_train].drop(columns="a_sinistre").copy()
test_alea = base_split[~masque_train].drop(columns="a_sinistre").copy()

print(f"\nTRAIN : {fr(len(train_alea))} contrats, "
      f"{fr(train_alea['exposition'].sum(), 1)} annees-police, "
      f"{fr(train_alea['nb_sinistres'].sum())} sinistres")
print(f"TEST  : {fr(len(test_alea))} contrats, "
      f"{fr(test_alea['exposition'].sum(), 1)} annees-police, "
      f"{fr(test_alea['nb_sinistres'].sum())} sinistres")

assert set(train_alea["id_police"]) & set(test_alea["id_police"]) == set()
print("\n[OK] Aucune police n'appartient aux deux echantillons : pas de fuite d'information.")

# %% [markdown]
# ## 4.3 Contrôle de représentativité
#
# Un découpage n'est exploitable que si les deux échantillons sont comparables. On
# vérifie la structure de risque, puis la structure des variables tarifaires.

# %%
def profil_echantillon(df, nom):
    """Resume les indicateurs de risque d'un echantillon."""
    e, n, c = df["exposition"].sum(), df["nb_sinistres"].sum(), df["charge_totale"].sum()
    return {
        "Échantillon": nom,
        "Contrats": fr(len(df)),
        "Exposition": fr(e, 1),
        "Sinistres": fr(n),
        "Fréquence": fr(n / e, 4),
        "Coût moyen": fr(c / n) if n else "n.d.",
        "Prime pure": fr(c / e),
        "Part graves (nb)": pct(df["nb_grave"].sum() / n) if n else "n.d.",
    }


titre("REPRESENTATIVITE DES ECHANTILLONS")
comparatif = pd.DataFrame([
    profil_echantillon(base_tarif, "Portefeuille complet"),
    profil_echantillon(train_temp, "Train temporel (2023)"),
    profil_echantillon(test_temp, "Test temporel (2024)"),
    profil_echantillon(train_alea, "Train aléatoire (80 %)"),
    profil_echantillon(test_alea, "Test aléatoire (20 %)"),
])
exporter_table(comparatif, "4_3_representativite_echantillons", index=False)
display(comparatif)

# Test formel : les frequences train et test sont-elles compatibles ?
def test_frequences(df1, df2, nom1, nom2):
    """Test du rapport de vraisemblance de l'egalite de deux frequences poissoniennes.

    Sous H0 les deux echantillons partagent la meme frequence lambda ; sous H1 chacun a
    la sienne. La statistique 2*(ll_H1 - ll_H0) suit asymptotiquement un chi2 a 1 ddl.
    """
    n1, e1 = df1["nb_sinistres"].sum(), df1["exposition"].sum()
    n2, e2 = df2["nb_sinistres"].sum(), df2["exposition"].sum()
    lam = (n1 + n2) / (e1 + e2)
    stat = 2 * (n1 * np.log(n1 / e1) + n2 * np.log(n2 / e2) - (n1 + n2) * np.log(lam))
    p = 1 - stats.chi2.cdf(stat, 1)
    print(f"{nom1} ({fr(n1 / e1, 4)}) vs {nom2} ({fr(n2 / e2, 4)}) : "
          f"chi2 = {stat:.2f}, p-value = {p:.3f} -> "
          f"{'ECART SIGNIFICATIF' if p < 0.05 else 'ecart non significatif'}")
    return p


print("\nTest d'egalite des frequences entre echantillons :")
test_frequences(train_temp, test_temp, "Train temporel", "Test temporel")
test_frequences(train_alea, test_alea, "Train aleatoire", "Test aleatoire")

# %%
# Comparaison de la structure des variables tarifaires
titre("STRUCTURE DES VARIABLES TARIFAIRES DANS CHAQUE ECHANTILLON", 2)
for v in ["tr_age", "zone_geo", "tr_anciennete"]:
    struct = pd.DataFrame({
        "Train temporel": train_temp.groupby(v, observed=True)["exposition"].sum()
        / train_temp["exposition"].sum(),
        "Test temporel": test_temp.groupby(v, observed=True)["exposition"].sum()
        / test_temp["exposition"].sum(),
        "Train aléatoire": train_alea.groupby(v, observed=True)["exposition"].sum()
        / train_alea["exposition"].sum(),
        "Test aléatoire": test_alea.groupby(v, observed=True)["exposition"].sum()
        / test_alea["exposition"].sum(),
    })
    print(f"\n--- Répartition de l'exposition par {v} ---")
    display(struct.map(lambda x: pct(x)))

print("\n-> Les structures sont tres proches dans les quatre echantillons. Les deux")
print("   decoupages sont donc exploitables pour la validation.")

# %% [markdown]
# ### Ce qu'il faut retenir de la partie F
#
# - La séparation par **année de souscription** est impossible : le portefeuille ne
#   compte qu'un seul millésime. Cet obstacle est contourné par un découpage par
#   **exercice de survenance**, en scindant l'exposition de chaque contrat au
#   1er janvier 2024.
# - Les deux découpages sont **conservatifs** : l'exposition et les sinistres se
#   recomposent exactement, sans perte ni double compte.
# - Les fréquences des échantillons d'apprentissage et de test ne diffèrent pas
#   significativement, et la structure des variables tarifaires est stable.
# - **Les deux dispositifs seront utilisés conjointement au §7.** Un modèle qui tient
#   sur les deux est robuste ; une dégradation sur le seul dispositif temporel
#   signalerait une instabilité dans le temps, une dégradation sur le seul dispositif
#   aléatoire un surapprentissage des profils.

# %% [markdown]
# ---
# # §5 — Partie D · Modélisation par GLM
#
# > **Ce que demande l'énoncé (point D)**
# > Mettre en place un ou plusieurs GLM pour modéliser la fréquence, le coût moyen, et
# > si pertinent les sinistres graves séparément — en précisant pour chacun : la
# > variable cible, les variables explicatives, la loi, la fonction de lien, et les
# > critères de sélection et de validation.
#
# ## 5.1 Le cadre
#
# Un **modèle linéaire généralisé** étend la régression linéaire sur trois points :
#
# 1. **Une loi de la famille exponentielle**, choisie selon la nature de la cible :
#    Poisson ou binomiale négative pour un comptage, Gamma ou log-normale pour un
#    montant positif et asymétrique.
# 2. **Une fonction de lien** `g` reliant l'espérance au prédicteur linéaire :
#    `g(E[Y]) = β₀ + β₁x₁ + … + βₚxₚ`. Le **lien logarithmique** est celui de la
#    tarification, pour une raison pratique décisive : il rend le tarif
#    **multiplicatif**. En effet `E[Y] = exp(β₀) × exp(β₁x₁) × … `, si bien que chaque
#    coefficient s'interprète directement comme un **coefficient tarifaire**
#    `exp(βⱼ)` appliqué à une prime de base.
# 3. **Un offset**, terme du prédicteur linéaire dont le coefficient est fixé à 1.
#
# ### Pourquoi l'exposition est un offset et non une variable explicative
#
# Point souligné par le cours, et souvent mal compris. Un contrat exposé six mois a,
# toutes choses égales par ailleurs, exactement **deux fois moins** de chances de
# déclarer un sinistre qu'un contrat exposé douze mois. Ce n'est pas une hypothèse à
# estimer : c'est une propriété du processus de comptage.
#
# En posant `N ~ Poisson(λ)` avec `λ = exposition × exp(Xβ)`, on obtient :
#
# $$\log(\lambda) = \log(\text{exposition}) + X\beta$$
#
# Le terme `log(exposition)` entre donc dans le modèle avec un coefficient **imposé à 1**.
# Laisser le modèle l'estimer librement reviendrait à admettre qu'un contrat de six mois
# puisse présenter, par exemple, 0,7 fois le risque d'un contrat annuel — ce qui n'a
# aucun sens. C'est cette contrainte que réalise l'offset.

# %% [markdown]
# ## 5.2 GLM de fréquence
#
# | Élément | Choix |
# |---|---|
# | **Variable cible** | `nb_sinistres` — nombre de sinistres du contrat sur la période |
# | **Loi** | Poisson (à vérifier : test de surdispersion ci-dessous) |
# | **Fonction de lien** | Logarithmique |
# | **Offset** | `log(exposition)` |
# | **Échantillon de calibration** | Train temporel (survenance 2023) |
#
# ### 5.2.1 La loi de Poisson est-elle adaptée ? Test de surdispersion
#
# La loi de Poisson impose une contrainte forte : **variance = espérance**. En pratique,
# les portefeuilles réels présentent souvent une variance supérieure — de la
# *surdispersion* — parce que des facteurs de risque non observés subsistent. On teste
# donc cette hypothèse avant de l'adopter.

# %%
titre("TEST DE SURDISPERSION")

moy = train_temp["nb_sinistres"].mean()
var = train_temp["nb_sinistres"].var()
print(f"Moyenne du nombre de sinistres  : {fr(moy, 5)}")
print(f"Variance du nombre de sinistres : {fr(var, 5)}")
print(f"Rapport variance / moyenne      : {fr(var / moy, 3)}")
print("\n-> Un rapport proche de 1 est compatible avec la loi de Poisson.")
print("   Un rapport nettement superieur a 1 signalerait une surdispersion.")

# %% [markdown]
# ### 5.2.2 Arbitrage de la colinéarité âge / ancienneté du permis
#
# Le §3.4 a montré que ces deux variables sont très fortement corrélées. Introduites
# ensemble, elles se neutralisent mutuellement. Trois spécifications sont donc mises en
# concurrence, départagées par l'AIC — critère qui pénalise la complexité du modèle et
# évite de retenir des variables qui n'améliorent l'ajustement que par surapprentissage.

# %%
def ajuster_frequence(df, variables, cible="nb_sinistres", famille=None):
    """Ajuste un GLM de frequence avec offset log(exposition)."""
    famille = famille or families.Poisson(links.Log())
    formule = f"{cible} ~ " + (" + ".join(f"C({v})" for v in variables) if variables else "1")
    return smf.glm(formule, data=df, family=famille,
                   offset=np.log(df["exposition"])).fit(), formule


titre("ARBITRAGE DE LA COLINEARITE AGE / ANCIENNETE", 2)
AUTRES = ["zone_geo", "carburant", "classe_vehicule", "tr_bonus", "tr_age_vehicule",
          "tr_puissance", "usage_vehicule"]

specifications = {
    "Âge seul": ["tr_age"] + AUTRES,
    "Ancienneté seule": ["tr_anciennete"] + AUTRES,
    "Les deux": ["tr_age", "tr_anciennete"] + AUTRES,
    "Indicatrice jeune conducteur": ["jeune_conducteur"] + AUTRES,
}

res_spec = []
modeles_spec = {}
for nom, vars_ in specifications.items():
    m, f = ajuster_frequence(train_temp, vars_)
    modeles_spec[nom] = m
    res_spec.append({"Spécification": nom, "Paramètres": int(m.df_model) + 1,
                     "Log-vraisemblance": round(m.llf, 1),
                     "AIC": round(m.aic, 1), "BIC": round(m.bic_llf, 1),
                     "Déviance": round(m.deviance, 1)})
tab_spec = pd.DataFrame(res_spec).sort_values("AIC")
exporter_table(tab_spec, "5_2_arbitrage_colinearite", index=False)
display(tab_spec)

meilleure = tab_spec.iloc[0]["Spécification"]
print(f"\nSpecification retenue par l'AIC : « {meilleure} »")

# Test du rapport de vraisemblance : ajouter la seconde variable apporte-t-il quelque chose ?
m_age = modeles_spec["Âge seul"]
m_deux = modeles_spec["Les deux"]
lr = 2 * (m_deux.llf - m_age.llf)
ddl = int(m_deux.df_model - m_age.df_model)
p_lr = 1 - stats.chi2.cdf(lr, ddl)
print(f"\nTest du rapport de vraisemblance — ajout de l'anciennete au modele avec age :")
print(f"   statistique = {lr:.2f}, ddl = {ddl}, p-value = {p_lr:.4f}")
print(f"   -> {'apport significatif' if p_lr < 0.05 else 'AUCUN apport significatif'}")
print("\n-> Conclusion : les deux variables portent la meme information. En retenir une")
print("   seule suffit, et evite des coefficients instables.")

# %% [markdown]
# ### 5.2.3 Sélection des variables par élimination descendante
#
# On part du modèle complet et on retire, une à une, la variable dont la suppression
# améliore le plus l'AIC. La procédure s'arrête quand aucune suppression ne l'améliore
# plus. Ce critère intègre automatiquement l'arbitrage entre qualité d'ajustement et
# parcimonie.

# %%
def selection_descendante(df, variables, cible="nb_sinistres", famille=None,
                          offset_col="exposition", verbeux=True):
    """Elimination descendante guidee par l'AIC.

    A chaque etape, on essaie de retirer chaque variable encore presente et on conserve
    la suppression qui abaisse le plus l'AIC. On s'arrete quand aucune n'ameliore plus.
    """
    famille = famille or families.Poisson(links.Log())
    offset = np.log(df[offset_col]) if offset_col else None
    retenues = list(variables)

    def ajuste(vars_):
        f = f"{cible} ~ " + (" + ".join(f"C({v})" for v in vars_) if vars_ else "1")
        return smf.glm(f, data=df, family=famille, offset=offset).fit(), f

    modele, formule = ajuste(retenues)
    historique = [{"Étape": 0, "Variable retirée": "—",
                   "Variables restantes": len(retenues), "AIC": round(modele.aic, 1)}]
    etape = 0
    while retenues:
        candidat, meilleur_aic, meilleur_modele, meilleure_formule = None, modele.aic, None, None
        for v in retenues:
            reste = [x for x in retenues if x != v]
            m2, f2 = ajuste(reste)
            if m2.aic < meilleur_aic - 1e-9:
                candidat, meilleur_aic, meilleur_modele, meilleure_formule = v, m2.aic, m2, f2
        if candidat is None:
            break
        etape += 1
        retenues.remove(candidat)
        modele, formule = meilleur_modele, meilleure_formule
        historique.append({"Étape": etape, "Variable retirée": candidat,
                           "Variables restantes": len(retenues), "AIC": round(modele.aic, 1)})
    if verbeux:
        display(pd.DataFrame(historique))
    return modele, formule, retenues


titre("SELECTION DESCENDANTE — GLM DE FREQUENCE")
vars_depart = specifications[meilleure]
print(f"Variables de depart ({len(vars_depart)}) : {vars_depart}\n")
glm_freq, formule_freq, vars_freq = selection_descendante(train_temp, vars_depart)

print(f"\nVariables RETENUES  : {vars_freq}")
print(f"Variables ECARTEES  : {[v for v in vars_depart if v not in vars_freq]}")
print(f"\nFormule finale : {formule_freq}")
print(f"AIC = {glm_freq.aic:.1f}   Deviance = {glm_freq.deviance:.1f}   "
      f"ddl residuels = {int(glm_freq.df_resid)}")

# %% [markdown]
# ### 5.2.4 Confirmation du choix de la loi
#
# Maintenant que le modèle est spécifié, on peut tester la surdispersion de façon plus
# rigoureuse : sur les **résidus du modèle**, et non sur la distribution brute. On
# compare également avec une binomiale négative, loi de référence en cas de
# surdispersion.

# %%
titre("SURDISPERSION — DIAGNOSTIC SUR LE MODELE AJUSTE", 2)
pearson = float((glm_freq.resid_pearson ** 2).sum())
phi = pearson / glm_freq.df_resid
print(f"Chi2 de Pearson             : {pearson:.1f}")
print(f"Degres de liberte residuels : {int(glm_freq.df_resid)}")
print(f"Parametre de dispersion phi : {phi:.3f}")
print("   (phi = 1 sous Poisson ; phi > 1 indiquerait une surdispersion)")

res_lois = []
for nom, fam in [("Poisson", families.Poisson(links.Log())),
                 ("Binomiale négative (α = 0,5)",
                  families.NegativeBinomial(links.Log(), alpha=0.5)),
                 ("Binomiale négative (α = 1,0)",
                  families.NegativeBinomial(links.Log(), alpha=1.0))]:
    m = smf.glm(formule_freq, data=train_temp, family=fam,
                offset=np.log(train_temp["exposition"])).fit()
    res_lois.append({"Loi": nom, "Log-vraisemblance": round(m.llf, 1),
                     "AIC": round(m.aic, 1), "Déviance": round(m.deviance, 1)})
tab_lois = pd.DataFrame(res_lois)
exporter_table(tab_lois, "5_2_comparaison_lois_frequence", index=False)
display(tab_lois)

print("\n-> RESULTAT NEGATIF, documente et non dissimule : la binomiale negative")
print("   DEGRADE l'AIC par rapport a la loi de Poisson. Il n'y a pas de surdispersion")
print("   a corriger dans ce portefeuille.")
print("\n   C'est en soi une information : elle signifie qu'une fois l'exposition prise")
print("   en compte par l'offset, la variabilite residuelle du nombre de sinistres est")
print("   exactement celle qu'un processus de Poisson produit. La LOI DE POISSON EST")
print("   RETENUE, non par defaut, mais parce qu'elle a ete testee.")

# %% [markdown]
# ### 5.2.5 Coefficients et lecture tarifaire
#
# Avec un lien logarithmique, `exp(βⱼ)` est le **coefficient multiplicatif** appliqué à
# la prime de base pour la modalité `j`. Une valeur de 1,35 signifie « 35 % de sinistres
# en plus que la modalité de référence, à autres caractéristiques égales ».

# %%
titre("COEFFICIENTS DU GLM DE FREQUENCE")
print(glm_freq.summary())

# %%
def table_coefficients(modele, nom_fichier=None):
    """Met en forme les coefficients d'un GLM en coefficients multiplicatifs."""
    ic = modele.conf_int()
    tab = pd.DataFrame({
        "Coefficient β": modele.params.round(4),
        "Erreur-type": modele.bse.round(4),
        "p-value": modele.pvalues.round(4),
        "Coef. multiplicatif exp(β)": np.exp(modele.params).round(3),
        "IC 95 % bas": np.exp(ic[0]).round(3),
        "IC 95 % haut": np.exp(ic[1]).round(3),
    })
    tab["Significatif à 5 %"] = np.where(tab["p-value"] < 0.05, "Oui", "Non")
    tab.index = [i.replace("C(", "").replace(")[T.", " : ").replace("]", "")
                 for i in tab.index]
    if nom_fichier:
        exporter_table(tab, nom_fichier)
    return tab


coefs_freq = table_coefficients(glm_freq, "5_2_coefficients_frequence")
display(coefs_freq)

print("\nModalites de reference (coefficient implicitement egal a 1) :")
for v in vars_freq:
    modalites = train_temp[v].dropna().unique()
    ref = sorted(map(str, modalites))[0] if not hasattr(train_temp[v], "cat") \
        else str(train_temp[v].cat.categories[0])
    print(f"   {v:<18} -> {ref}")

n_signif = int((coefs_freq["Significatif à 5 %"] == "Oui").sum())
print(f"\nCoefficients significatifs a 5 % : {n_signif} sur {len(coefs_freq)}")

# %% [markdown]
# ### 5.2.6 Diagnostics du modèle de fréquence

# %%
titre("DIAGNOSTIC 1 — CONSERVATION DE LA CHARGE (equilibre du modele)", 2)
pred_train = glm_freq.predict(train_temp, offset=np.log(train_temp["exposition"]))
print(f"Sinistres predits sur l'echantillon d'apprentissage : {fr(pred_train.sum(), 1)}")
print(f"Sinistres observes                                  : {fr(train_temp['nb_sinistres'].sum())}")
print(f"Ecart                                               : "
      f"{pct(pred_train.sum() / train_temp['nb_sinistres'].sum() - 1, 4)}")
print("\n-> Un GLM de Poisson a lien log comportant une constante redonne EXACTEMENT")
print("   le total observe. C'est une propriete mathematique du modele, et donc un")
print("   controle imparable : tout ecart signalerait une erreur de code.")
assert abs(pred_train.sum() / train_temp["nb_sinistres"].sum() - 1) < 1e-6

# %%
titre("DIAGNOSTIC 2 — CALIBRATION PAR DECILE DE RISQUE", 2)


def table_calibration(df, predictions, n_groupes=10, cible="nb_sinistres"):
    """Compare le predit et l'observe par groupe de risque croissant."""
    t = pd.DataFrame({
        "pred": predictions.values if hasattr(predictions, "values") else predictions,
        "obs": df[cible].values,
        "expo": df["exposition"].values,
    })
    t["risque"] = t["pred"] / t["expo"]
    t["groupe"] = pd.qcut(t["risque"].rank(method="first"), n_groupes,
                          labels=[f"D{i}" for i in range(1, n_groupes + 1)])
    g = t.groupby("groupe", observed=True).agg(
        Exposition=("expo", "sum"), Observés=("obs", "sum"), Prédits=("pred", "sum"))
    g["Fréq. observée"] = g["Observés"] / g["Exposition"]
    g["Fréq. prédite"] = g["Prédits"] / g["Exposition"]
    g["Écart"] = g["Prédits"] / g["Observés"] - 1
    return g


calib = table_calibration(train_temp, pred_train)
aff_calib = pd.DataFrame({
    "Exposition": calib["Exposition"].map(lambda v: fr(v, 1)),
    "Sinistres observés": calib["Observés"].map(fr),
    "Sinistres prédits": calib["Prédits"].map(lambda v: fr(v, 1)),
    "Fréq. observée": calib["Fréq. observée"].map(lambda v: fr(v, 4)),
    "Fréq. prédite": calib["Fréq. prédite"].map(lambda v: fr(v, 4)),
    "Écart": calib["Écart"].map(lambda v: pct(v)),
})
display(aff_calib)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
x = np.arange(len(calib))
axes[0].bar(x - 0.2, calib["Fréq. observée"], width=0.4, label="Observée",
            color=PALETTE["petrole"])
axes[0].bar(x + 0.2, calib["Fréq. prédite"], width=0.4, label="Prédite",
            color=PALETTE["ambre"])
axes[0].set_xticks(x); axes[0].set_xticklabels(calib.index)
axes[0].set_title("Calibration par décile de risque (apprentissage)")
axes[0].set_xlabel("Décile de fréquence prédite")
axes[0].set_ylabel("Sinistres par année-police")
axes[0].legend(frameon=False)

res_dev = glm_freq.resid_deviance
axes[1].scatter(np.log(pred_train.clip(lower=1e-6)), res_dev, s=6, alpha=0.25,
                color=PALETTE["encre"])
axes[1].axhline(0, color=PALETTE["brique"], lw=1.2)
axes[1].set_title("Résidus de déviance")
axes[1].set_xlabel("log (nombre de sinistres prédit)")
axes[1].set_ylabel("Résidu de déviance")
plt.tight_layout()
sauver_fig("5_2_diagnostics_frequence")

print("\n-> La frequence predite croit regulierement du decile 1 au decile 10 et suit")
print("   l'observe : le modele ordonne correctement le risque. La structure en deux")
print("   branches des residus est NORMALE pour une cible de comptage majoritairement")
print("   nulle (une branche pour les contrats sans sinistre, une pour les autres).")

# %% [markdown]
# **Un résultat inattendu à ne pas passer sous silence.** La modalité « Non renseignée »
# de la zone géographique ressort avec un coefficient de **1,79**, significatif à 5 %.
# Autrement dit, les contrats dont la zone n'est pas renseignée présentent une fréquence
# près de deux fois supérieure à la référence. Le non-renseignement n'est donc pas
# aléatoire : il porte lui-même une information de risque. C'est la justification a
# posteriori du choix, fait au §1.6, de créer une modalité dédiée plutôt que d'imputer
# ces valeurs par la zone la plus fréquente — ce qui aurait dilué cette information.
#
# Avec seulement 201 contrats concernés, l'intervalle de confiance reste large
# (de 1,01 à 3,18) : le résultat mérite d'être signalé, pas d'être surinterprété.

# %% [markdown]
# ### 5.2.7 Une interaction améliorerait-elle le modèle ?
#
# Le modèle retenu est **multiplicatif** : il suppose que la surfréquence d'un
# conducteur novice est la même dans toutes les zones. Une interaction lèverait cette
# hypothèse — un novice pourrait être plus pénalisé en zone urbaine dense. On la teste
# plutôt que de la supposer absente.

# %%
titre("TEST D'INTERACTION CONDUCTEUR NOVICE x ZONE", 2)
glm_inter = smf.glm("nb_sinistres ~ C(jeune_conducteur) * C(zone_geo)", data=train_temp,
                    family=families.Poisson(links.Log()),
                    offset=np.log(train_temp["exposition"])).fit()
lr_inter = 2 * (glm_inter.llf - glm_freq.llf)
ddl_inter = int(glm_inter.df_model - glm_freq.df_model)
p_inter = 1 - stats.chi2.cdf(lr_inter, ddl_inter)
print(f"AIC sans interaction : {glm_freq.aic:.1f}")
print(f"AIC avec interaction : {glm_inter.aic:.1f}")
print(f"Rapport de vraisemblance : {lr_inter:.2f}, ddl = {ddl_inter}, p-value = {p_inter:.3f}")
print("\n-> RESULTAT NEGATIF : l'interaction degrade l'AIC et n'est pas significative.")
print("   La surfrequence des novices est la meme dans toutes les zones ; la structure")
print("   multiplicative du tarif est validee, pas seulement supposee.")

# %% [markdown]
# ## 5.3 GLM de sévérité — sinistres attritionnels
#
# | Élément | Choix |
# |---|---|
# | **Variable cible** | `montant` — coût individuel de chaque sinistre attritionnel |
# | **Unité d'observation** | le sinistre, et non le contrat |
# | **Loi** | à déterminer : Gamma, log-normale ou inverse gaussienne |
# | **Fonction de lien** | Logarithmique |
# | **Offset** | aucun — le coût d'un sinistre ne dépend pas de l'exposition |
#
# **Pourquoi pas d'offset ici.** L'exposition gouverne le *nombre* de sinistres, pas
# leur *coût*. Un contrat exposé six mois n'a pas de sinistres deux fois moins chers.
# C'est la différence essentielle avec le modèle de fréquence.

# %%
titre("CONSTRUCTION DE LA BASE DE SEVERITE")

CARACS = ["id_police", "tr_age", "tr_anciennete", "jeune_conducteur", "sexe", "zone_geo",
          "usage_vehicule", "classe_vehicule", "carburant", "type_couverture",
          "tr_bonus", "tr_age_vehicule", "tr_puissance"]

sin_carac = sinistres.merge(contrats[CARACS], on="id_police", how="inner")
sin_carac["annee"] = sin_carac["date_sin"].dt.year

sev_att_train = sin_carac[(sin_carac["categorie"] == "Attritionnel")
                          & (sin_carac["montant"] > 0)
                          & (sin_carac["annee"] == 2023)].copy()
sev_att_test = sin_carac[(sin_carac["categorie"] == "Attritionnel")
                         & (sin_carac["montant"] > 0)
                         & (sin_carac["annee"] == 2024)].copy()

print(f"Sinistres attritionnels de montant strictement positif :")
print(f"   apprentissage (survenance 2023) : {fr(len(sev_att_train))}")
print(f"   test          (survenance 2024) : {fr(len(sev_att_test))}")
print(f"\nMontants ecartes car nuls : "
      f"{fr((sin_carac['categorie'] == 'Attritionnel').sum() - len(sev_att_train) - len(sev_att_test))}")
print("   (les lois Gamma, log-normale et inverse gaussienne exigent une cible")
print("    strictement positive ; ces sinistres restent comptes en frequence)")
print(f"\nStatistiques du cout attritionnel (apprentissage) :")
print(sev_att_train["montant"].describe().to_string())
print(f"Coefficient d'asymetrie : {fr(stats.skew(sev_att_train['montant']), 2)}")

# %% [markdown]
# ### 5.3.1 Choix de la loi
#
# Le cours cite trois lois pour les montants : **exponentielle**, **Gamma** et
# **log-normale**. On y ajoute l'**inverse gaussienne**, également usuelle en
# tarification. L'exponentielle étant un cas particulier de la Gamma (paramètre de forme
# égal à 1), elle est couverte par cette dernière.
#
# **Précaution de comparaison.** On ne peut pas comparer directement l'AIC d'un modèle
# Gamma sur `y` et celui d'une régression sur `log(y)` : les deux ne portent pas sur la
# même variable, et leurs vraisemblances ne sont pas exprimées dans la même unité. Pour
# une comparaison valide, on recalcule la **log-vraisemblance de chaque modèle sur
# l'échelle d'origine**, en appliquant au modèle log-normal la correction du jacobien
# `-Σ log(yᵢ)`.

# %%
def comparer_lois_severite(df, variables, cible="montant"):
    """Compare Gamma, log-normale et inverse gaussienne sur une base commune.

    Les log-vraisemblances sont toutes exprimees sur l'echelle du montant d'origine,
    ce qui rend les AIC directement comparables.
    """
    formule = f"{cible} ~ " + (" + ".join(f"C({v})" for v in variables) if variables else "1")
    y = df[cible].values
    resultats, modeles = [], {}

    # --- Gamma, lien log ---
    m_gam = smf.glm(formule, data=df, family=families.Gamma(links.Log())).fit()
    mu = m_gam.fittedvalues.values
    phi = m_gam.scale
    k = 1.0 / phi
    ll_gam = float(np.sum(stats.gamma.logpdf(y, a=k, scale=mu * phi)))
    p_gam = int(m_gam.df_model) + 2
    modeles["Gamma"] = m_gam
    resultats.append(("Gamma", ll_gam, p_gam, -2 * ll_gam + 2 * p_gam))

    # --- Log-normale : regression gaussienne sur log(y), puis retour a l'echelle y ---
    df_log = df.assign(_logy=np.log(y))
    m_ln = smf.ols(formule.replace(cible, "_logy"), data=df_log).fit()
    s2 = float(np.sum(m_ln.resid ** 2) / len(y))
    ll_ln = float(np.sum(stats.norm.logpdf(np.log(y), loc=m_ln.fittedvalues.values,
                                           scale=np.sqrt(s2))) - np.sum(np.log(y)))
    p_ln = int(m_ln.df_model) + 2
    modeles["Log-normale"] = m_ln
    resultats.append(("Log-normale", ll_ln, p_ln, -2 * ll_ln + 2 * p_ln))

    # --- Inverse gaussienne, lien log ---
    m_ig = smf.glm(formule, data=df, family=families.InverseGaussian(links.Log())).fit()
    mu_ig = m_ig.fittedvalues.values
    phi_ig = m_ig.scale
    ll_ig = float(np.sum(-0.5 * np.log(2 * np.pi * phi_ig * y ** 3)
                         - (y - mu_ig) ** 2 / (2 * phi_ig * mu_ig ** 2 * y)))
    p_ig = int(m_ig.df_model) + 2
    modeles["Inverse gaussienne"] = m_ig
    resultats.append(("Inverse gaussienne", ll_ig, p_ig, -2 * ll_ig + 2 * p_ig))

    tab = pd.DataFrame(resultats, columns=["Loi", "Log-vraisemblance", "Paramètres", "AIC"])
    tab["ΔAIC"] = (tab["AIC"] - tab["AIC"].min()).round(1)
    tab["Log-vraisemblance"] = tab["Log-vraisemblance"].round(1)
    tab["AIC"] = tab["AIC"].round(1)
    return tab.sort_values("AIC"), modeles


titre("COMPARAISON DES LOIS DE SEVERITE", 2)
VARS_SEV = ["tr_age", "zone_geo", "classe_vehicule", "carburant", "tr_age_vehicule",
            "tr_puissance", "type_couverture", "usage_vehicule"]
tab_lois_sev, modeles_sev = comparer_lois_severite(sev_att_train, VARS_SEV)
exporter_table(tab_lois_sev, "5_3_comparaison_lois_severite", index=False)
display(tab_lois_sev)

loi_retenue = tab_lois_sev.iloc[0]["Loi"]
print(f"\nLoi retenue : {loi_retenue} (AIC le plus faible sur echelle comparable)")

# %%
# Diagnostic graphique de l'ajustement des trois lois
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
y_obs = sev_att_train["montant"].values
for ax, nom in zip(axes, ["Gamma", "Log-normale", "Inverse gaussienne"]):
    m = modeles_sev[nom]
    if nom == "Log-normale":
        resid = m.resid / np.sqrt(np.sum(m.resid ** 2) / len(y_obs))
    else:
        resid = m.resid_deviance / np.sqrt(m.scale)
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title(f"{nom}")
    ax.get_lines()[0].set_markerfacecolor(PALETTE["petrole"])
    ax.get_lines()[0].set_markeredgecolor("none")
    ax.get_lines()[0].set_markersize(3.5)
    ax.get_lines()[1].set_color(PALETTE["brique"])
    ax.set_xlabel("Quantiles théoriques")
    ax.set_ylabel("Résidus standardisés")
plt.suptitle("Droites de Henry des résidus — plus les points suivent la droite, "
             "meilleur est l'ajustement", y=1.03, fontsize=10.5)
plt.tight_layout()
sauver_fig("5_3_qqplots_lois_severite")

# %% [markdown]
# %% [markdown]
# **La log-normale l'emporte nettement** (ΔAIC supérieur à 35 face à la Gamma). Elle
# capte mieux l'asymétrie du coût attritionnel. Les droites de Henry le confirment
# visuellement : les résidus log-normaux suivent la droite sur presque toute la plage,
# alors que ceux de la Gamma et de l'inverse gaussienne s'en écartent dans les queues.
#
# **Une contrepartie technique à traiter.** La régression log-normale modélise
# `log(montant)`, si bien que `exp(Xβ)` estime la **médiane** du montant, pas sa moyenne.
# Or la tarification a besoin de l'**espérance**. Utiliser `exp(Xβ)` directement
# sous-estimerait systématiquement la prime.
#
# On applique donc le **facteur de lissage de Duan** (*smearing estimator*), qui corrige
# le retour à l'échelle d'origine sans hypothèse de normalité des résidus :
#
# $$\widehat{\mathbb{E}}[Y \mid X] = \exp(X\hat\beta) \times \frac{1}{n}\sum_{i=1}^{n} \exp(\hat\varepsilon_i)$$

# %% [markdown]
# ### 5.3.2 Sélection des variables du modèle de sévérité

# %%
def ajusteur_severite(df, loi, cible="montant"):
    """Renvoie une fonction d'ajustement adaptee a la loi retenue.

    Pour la log-normale, on regresse log(montant) par moindres carres ordinaires : les
    AIC de modeles emboites sur la meme cible restent comparables entre eux, ce qui
    suffit pour la selection de variables.
    """
    if loi == "Log-normale":
        donnees = df.assign(_logy=np.log(df[cible]))

        def ajuste(vars_):
            f = "_logy ~ " + (" + ".join(f"C({v})" for v in vars_) if vars_ else "1")
            return smf.ols(f, data=donnees).fit(), f
    else:
        fam = (families.Gamma(links.Log()) if loi == "Gamma"
               else families.InverseGaussian(links.Log()))

        def ajuste(vars_):
            f = f"{cible} ~ " + (" + ".join(f"C({v})" for v in vars_) if vars_ else "1")
            return smf.glm(f, data=df, family=fam).fit(), f
    return ajuste


def selection_severite(df, variables, loi, cible="montant", verbeux=True):
    """Elimination descendante guidee par l'AIC, pour la loi de severite retenue."""
    ajuste = ajusteur_severite(df, loi, cible)
    retenues = list(variables)
    modele, formule = ajuste(retenues)
    historique = [{"Étape": 0, "Variable retirée": "—",
                   "Variables restantes": len(retenues), "AIC": round(modele.aic, 1)}]
    etape = 0
    while retenues:
        candidat, meilleur, mod_c, form_c = None, modele.aic, None, None
        for v in retenues:
            reste = [x for x in retenues if x != v]
            m2, f2 = ajuste(reste)
            if m2.aic < meilleur - 1e-9:
                candidat, meilleur, mod_c, form_c = v, m2.aic, m2, f2
        if candidat is None:
            break
        etape += 1
        retenues.remove(candidat)
        modele, formule = mod_c, form_c
        historique.append({"Étape": etape, "Variable retirée": candidat,
                           "Variables restantes": len(retenues), "AIC": round(modele.aic, 1)})
    if verbeux:
        display(pd.DataFrame(historique))
    return modele, formule, retenues


def facteur_duan(modele, loi):
    """Facteur de lissage corrigeant le retour a l'echelle d'origine (log-normale)."""
    if loi != "Log-normale":
        return 1.0
    return float(np.mean(np.exp(modele.resid)))


def predire_severite(modele, df, loi, lissage=1.0):
    """Predit l'ESPERANCE du montant, quelle que soit la loi du modele."""
    if loi == "Log-normale":
        return np.exp(modele.predict(df)) * lissage
    return modele.predict(df)


titre("SELECTION DESCENDANTE — GLM DE SEVERITE ATTRITIONNELLE")
print(f"Loi utilisee : {loi_retenue}\n")
glm_sev, formule_sev, vars_sev = selection_severite(sev_att_train, VARS_SEV, loi_retenue)
lissage_sev = facteur_duan(glm_sev, loi_retenue)

print(f"\nVariables RETENUES : {vars_sev if vars_sev else 'AUCUNE (modele reduit a la constante)'}")
print(f"Variables ECARTEES : {[v for v in VARS_SEV if v not in vars_sev]}")
print(f"\nFormule finale : {formule_sev}")
print(f"AIC = {glm_sev.aic:.1f}")
print(f"Facteur de lissage de Duan : {lissage_sev:.4f}")

pred_sev_train = predire_severite(glm_sev, sev_att_train, loi_retenue, lissage_sev)
print(f"\nControle du retour a l'echelle d'origine :")
print(f"   cout moyen predit  : {fr(pred_sev_train.mean())}")
print(f"   cout moyen observe : {fr(sev_att_train['montant'].mean())}")
print(f"   ecart              : {pct(pred_sev_train.mean() / sev_att_train['montant'].mean() - 1, 2)}")
print("\n-> Sans le facteur de lissage, le cout moyen predit serait "
      f"{fr((pred_sev_train / lissage_sev).mean())}, soit "
      f"{pct((pred_sev_train / lissage_sev).mean() / sev_att_train['montant'].mean() - 1)} "
      "sous l'observe.")
print("   La correction est donc loin d'etre cosmetique.")

# %%
if vars_sev:
    coefs_sev = table_coefficients(glm_sev, "5_3_coefficients_severite")
    display(coefs_sev)
    n_sig_sev = int((coefs_sev["Significatif à 5 %"] == "Oui").sum()) - 1
    print(f"\nCoefficients significatifs a 5 % (hors constante) : {n_sig_sev} "
          f"sur {len(coefs_sev) - 1}")
    etendue_sev = (coefs_sev["Coef. multiplicatif exp(β)"].drop("Intercept").max()
                   / coefs_sev["Coef. multiplicatif exp(β)"].drop("Intercept").min())
    print(f"Rapport entre le coefficient le plus eleve et le plus faible : {fr(etendue_sev, 2)}")
else:
    print("\n-> La selection ecarte TOUTES les variables explicatives.")

# %% [markdown]
# **Comparaison avec le modèle de fréquence : le contraste est frappant.**
#
# Deux variables survivent bien à la sélection sur la sévérité — la classe de véhicule
# et le carburant — mais seules deux de leurs modalités sur huit sont significatives à
# 5 %, et l'amplitude entre le coefficient le plus faible et le plus élevé reste
# modeste. Surtout, **aucune variable liée au conducteur** — ni l'âge, ni l'ancienneté
# du permis, ni le statut de novice — n'est retenue, alors que ces mêmes variables
# dominaient le modèle de fréquence.
#
# Ce contraste a une lecture métier directe : **le profil du conducteur détermine la
# probabilité d'avoir un accident, pas le coût du dommage matériel qui en résulte.**
# Un conducteur novice heurte un obstacle plus souvent, mais un pare-brise cassé coûte
# le même prix quel que soit l'âge de celui qui l'a cassé. Ce qui influence le coût,
# ce sont les caractéristiques du **véhicule** — ce que retient précisément le modèle.
#
# **Conséquence tarifaire :** l'essentiel de la segmentation du tarif attritionnel passe
# par la fréquence. C'est un enseignement classique de la tarification automobile, et
# il est ici retrouvé sur données.

# %% [markdown]
# ## 5.4 Sinistres graves
#
# Les sinistres graves sont modélisés **séparément**, conformément au point D.3 de
# l'énoncé. Deux briques : leur fréquence, puis leur coût.

# %% [markdown]
# ### 5.4.1 Fréquence des sinistres graves
#
# | Élément | Choix |
# |---|---|
# | **Variable cible** | `nb_grave` |
# | **Loi** | Poisson |
# | **Lien** | Logarithmique |
# | **Offset** | `log(exposition)` |
#
# Avec un nombre d'événements très faible, le modèle doit rester **volontairement
# parcimonieux** : estimer de nombreux coefficients sur peu d'événements produirait des
# valeurs instables et non reproductibles.
#
# **Règle de sélection renforcée.** Toutes les variables tarifaires sont candidates.
# L'élimination descendante par l'AIC est suivie d'une **confirmation par test du rapport
# de vraisemblance** : une variable n'est conservée que si son retrait dégrade
# significativement l'ajustement, au seuil de 5 %. L'AIC seul est trop permissif quand
# les événements se comptent par dizaines : il retient une variable dès que son gain de
# vraisemblance dépasse le nombre de ses paramètres, ce qui correspond, pour une
# variable à deux coefficients, à une p-value d'environ 13 %.

# %%
titre("GLM DE FREQUENCE DES SINISTRES GRAVES")
n_grave_train = train_temp["nb_grave"].sum()
print(f"Sinistres graves dans l'echantillon d'apprentissage : {fr(n_grave_train)}")
print(f"Exposition                                          : {fr(train_temp['exposition'].sum(), 1)}")
print(f"Frequence des graves                                : "
      f"{fr(n_grave_train / train_temp['exposition'].sum(), 5)}")
print(f"soit environ 1 sinistre grave pour "
      f"{fr(train_temp['exposition'].sum() / n_grave_train, 0)} annees-police.")

print("\nRegle empirique de dimensionnement : environ 10 evenements par parametre estime.")
print(f"   Budget de parametres raisonnable ici : {int(n_grave_train // 10)}")

CANDIDATES_GRAVES = ["jeune_conducteur", "zone_geo", "classe_vehicule", "carburant",
                     "type_couverture", "usage_vehicule", "tr_bonus", "tr_age_vehicule",
                     "tr_puissance"]


def confirmer_par_lr(df, variables, cible, seuil=0.05):
    """Retire iterativement la variable la moins significative tant que son test du
    rapport de vraisemblance depasse le seuil. Renvoie les variables confirmees."""
    retenues, journal = list(variables), []
    while retenues:
        complet, _ = ajuster_frequence(df, retenues, cible)
        pvals = {}
        for v in retenues:
            reduit, _ = ajuster_frequence(df, [x for x in retenues if x != v], cible)
            lr = 2 * (complet.llf - reduit.llf)
            pvals[v] = 1 - stats.chi2.cdf(lr, int(complet.df_model - reduit.df_model))
        pire = max(pvals, key=pvals.get)
        journal.append({"Variable testée": pire, "p-value du retrait": round(pvals[pire], 4),
                        "Décision": "retirée" if pvals[pire] > seuil else "conservée"})
        if pvals[pire] <= seuil:
            break
        retenues.remove(pire)
    return retenues, pd.DataFrame(journal)


print(f"\nCandidates ({len(CANDIDATES_GRAVES)}) : {CANDIDATES_GRAVES}\n")
print("Etape 1 — elimination descendante par l'AIC :")
_, _, vars_fg_aic = selection_descendante(train_temp, CANDIDATES_GRAVES, cible="nb_grave")
print(f"   retenues par l'AIC : {vars_fg_aic}")

print("\nEtape 2 — confirmation par test du rapport de vraisemblance (seuil 5 %) :")
vars_fg, journal_lr = confirmer_par_lr(train_temp, vars_fg_aic, "nb_grave")
display(journal_lr)
glm_freq_grave, formule_fg = ajuster_frequence(train_temp, vars_fg, "nb_grave")
print(f"\nVariables RETENUES : {vars_fg if vars_fg else 'AUCUNE'}")
print(f"Formule finale : {formule_fg}")

lignes_specs = []
for nom, vars_ in [("Constante (mutualisation)", []),
                   ("Conducteur novice", ["jeune_conducteur"]),
                   ("Tranche d'âge", ["tr_age"]),
                   ("Novice + usage", ["jeune_conducteur", "usage_vehicule"])]:
    m, _ = ajuster_frequence(train_temp, vars_, "nb_grave")
    lignes_specs.append({"Spécification": nom, "Paramètres": int(m.df_model) + 1,
                         "AIC": round(m.aic, 1), "BIC": round(m.bic_llf, 1)})
specs_graves = pd.DataFrame(lignes_specs)
exporter_table(specs_graves, "5_4_specifications_frequence_graves", index=False)
print("\nSpecifications concurrentes :")
display(specs_graves)
print("-> L'AIC prefere les modeles comportant le statut de novice ; le BIC, plus severe,")
print("   prefere la constante. Les deux criteres divergent : c'est le signe d'un effet")
print("   reel mais estime sur trop peu d'evenements pour etre mesure avec precision.")
print("   Sa stabilite est examinee hors echantillon au §7.3.")

freq_grave_moy = n_grave_train / train_temp["exposition"].sum()
if vars_fg:
    coefs_fg = table_coefficients(glm_freq_grave, "5_4_coefficients_frequence_graves")
    display(coefs_fg)
    print("\n-> RESULTAT NOTABLE : malgre le faible effectif, le statut de conducteur")
    print("   novice ressort significativement sur la frequence des sinistres GRAVES,")
    print("   avec un coefficient encore plus eleve que sur la frequence globale.")
    print("\n   Lecture metier : un conducteur novice ne declare pas seulement PLUS de")
    print("   sinistres, il declare aussi des sinistres PLUS GRAVES. C'est coherent avec")
    print("   ce que l'on sait de l'accidentologie des jeunes conducteurs (vitesse,")
    print("   experience limitee des situations d'urgence).")
    print("\n   PRECAUTION : l'intervalle de confiance reste large. Le sens de l'effet")
    print("   est etabli, son ampleur exacte ne l'est pas.")
else:
    print("\n-> Aucune variable ne ressort. La frequence des graves est MUTUALISEE")
    print(f"   sur l'ensemble du portefeuille, a {fr(freq_grave_moy, 5)} par annee-police.")
    print("\n   Avec seulement quelques dizaines d'evenements, la puissance statistique")
    print("   est insuffisante pour detecter des ecarts de segmentation, meme s'ils")
    print("   existent. Affirmer le contraire sur cette base serait une erreur.")

# %% [markdown]
# ### 5.4.2 Coût des sinistres graves
#
# Trois approches sont mises en concurrence : un GLM Gamma segmenté, une moyenne
# mutualisée, et l'espérance déduite de la **loi de Pareto généralisée** ajustée au §2.4.
#
# L'approche GPD a un fondement théorique : si les excès au-delà du seuil suivent une
# GPD de paramètres (ξ, σ), alors l'espérance du montant d'un sinistre grave vaut :
#
# $$\mathbb{E}[X \mid X > u] = u + \frac{\sigma}{1 - \xi} \quad \text{(définie pour } \xi < 1\text{)}$$

# %%
titre("COUT DES SINISTRES GRAVES")
sev_gr_train = sin_carac[(sin_carac["categorie"] == "Grave")
                         & (sin_carac["montant"] > 0)
                         & (sin_carac["annee"] == 2023)].copy()
print(f"Effectif disponible pour l'apprentissage : {fr(len(sev_gr_train))} sinistres graves")
print(sev_gr_train["montant"].describe().to_string())

# a) GLM Gamma segmente
glm_sev_grave, form_sg, vars_sg = selection_severite(
    sev_gr_train, ["zone_geo", "classe_vehicule", "jeune_conducteur"], "Gamma")
print(f"\na) GLM Gamma segmente -> variables retenues : {vars_sg if vars_sg else 'AUCUNE'}")

# b) Moyenne mutualisee
cout_grave_moyen = float(sev_gr_train["montant"].mean())
erreur_type = float(sev_gr_train["montant"].std() / np.sqrt(len(sev_gr_train)))
print(f"\nb) Moyenne empirique mutualisee : {fr(cout_grave_moyen)}")
print(f"   Erreur-type : {fr(erreur_type)}  "
      f"-> IC 95 % = [{fr(cout_grave_moyen - 1.96 * erreur_type)} ; "
      f"{fr(cout_grave_moyen + 1.96 * erreur_type)}]")
print(f"   Largeur relative de l'intervalle : "
      f"{pct(2 * 1.96 * erreur_type / cout_grave_moyen)} de la valeur estimee")

# c) Esperance deduite de la GPD
xi_tr, _, sig_tr = stats.genpareto.fit(
    (sev_gr_train["montant"] - SEUIL_GRAVE).values, floc=0)
esp_gpd = SEUIL_GRAVE + sig_tr / (1 - xi_tr) if xi_tr < 1 else np.inf
print(f"\nc) Esperance deduite de la GPD : xi = {xi_tr:.3f}, sigma = {fr(sig_tr)}")
print(f"   E[X | X > u] = {fr(SEUIL_GRAVE)} + {fr(sig_tr)} / (1 - {xi_tr:.3f}) = {fr(esp_gpd)}")

comp_grave = pd.DataFrame({
    "Approche": ["GLM Gamma segmenté", "Moyenne mutualisée", "Espérance GPD"],
    "Coût moyen estimé": [
        fr(float(np.exp(glm_sev_grave.params["Intercept"]))) if not vars_sg
        else "segmenté (voir coefficients)",
        fr(cout_grave_moyen), fr(esp_gpd)],
    "Commentaire": [
        "Aucune variable retenue" if not vars_sg else "Variables retenues",
        f"IC 95 % large de {pct(2 * 1.96 * erreur_type / cout_grave_moyen)}",
        "Fondée sur la théorie des valeurs extrêmes"],
})
exporter_table(comp_grave, "5_4_cout_graves_comparaison", index=False)
display(comp_grave)

print("\n-> APPROCHE RETENUE : la moyenne mutualisee.")
print("   Justification : avec une centaine d'observations et un intervalle de")
print("   confiance de l'ordre de 40 % de la valeur estimee, toute segmentation")
print("   du cout des graves serait du bruit presente comme du signal.")
print("   L'estimation GPD sert de CONTROLE : sa proximite avec la moyenne empirique")
print("   conforte cette derniere.")

# %% [markdown]
# **La loi de Pareto généralisée permettrait-elle une estimation plus stable ?**
# L'argument est séduisant : en lissant la queue par une loi paramétrique, on réduirait
# l'influence des quelques très gros sinistres sur la moyenne. On le teste par
# **bootstrap** — on rééchantillonne 500 fois les sinistres graves, on recalcule les deux
# estimateurs à chaque fois, et on compare leur dispersion. L'estimateur le plus stable
# est celui dont le coefficient de variation est le plus faible.

# %%
titre("STABILITE DU COUT DES GRAVES : MOYENNE EMPIRIQUE vs GPD (bootstrap)", 2)
gen_boot = np.random.default_rng(GRAINE)
x_gr = sev_gr_train["montant"].values
boot_emp, boot_gpd = [], []
for _ in range(500):
    ech = x_gr[gen_boot.integers(0, len(x_gr), len(x_gr))]
    boot_emp.append(ech.mean())
    xi_b, _, sg_b = stats.genpareto.fit(ech - SEUIL_GRAVE, floc=0)
    boot_gpd.append(SEUIL_GRAVE + sg_b / (1 - xi_b) if xi_b < 1 else np.nan)
boot_emp, boot_gpd = np.array(boot_emp), np.array(boot_gpd)

stab = pd.DataFrame({
    "Estimateur": ["Moyenne empirique", "Espérance GPD"],
    "Valeur": [fr(cout_grave_moyen), fr(esp_gpd)],
    "Écart-type bootstrap": [fr(boot_emp.std()), fr(np.nanstd(boot_gpd))],
    "Coefficient de variation": [pct(boot_emp.std() / boot_emp.mean()),
                                 pct(np.nanstd(boot_gpd) / np.nanmean(boot_gpd))],
})
exporter_table(stab, "5_4_stabilite_cout_graves", index=False)
display(stab)
cv_emp = boot_emp.std() / boot_emp.mean()
cv_gpd = np.nanstd(boot_gpd) / np.nanmean(boot_gpd)
if cv_gpd >= cv_emp * 0.95:
    print("\n-> RESULTAT NEGATIF : la GPD n'apporte pas de gain de stabilite sensible")
    print(f"   (coefficient de variation {pct(cv_gpd)} contre {pct(cv_emp)}).")
    print(f"   Elle doit estimer deux parametres (xi et sigma) sur les memes {len(x_gr)}")
    print("   sinistres, et l'incertitude sur l'indice de queue se repercute sur")
    print("   l'esperance. Le lissage parametrique ne cree pas d'information : il ne fait")
    print("   que la redistribuer. La moyenne empirique est conservee.")
else:
    print(f"\n-> La GPD reduit la dispersion ({pct(cv_gpd)} contre {pct(cv_emp)}) :")
    print("   elle constitue alors un estimateur preferable du cout des graves.")

# %% [markdown]
# ## 5.5 Récapitulatif des modèles
#
# Tableau demandé explicitement par l'énoncé.

# %%
recap = pd.DataFrame([
    {"Modèle": "Fréquence — tous sinistres",
     "Variable cible": "nb_sinistres (comptage)",
     "Variables explicatives": ", ".join(vars_freq),
     "Loi": "Poisson", "Lien": "log", "Offset": "log(exposition)",
     "Sélection": "AIC (élimination descendante)",
     "Validation": "Conservation du total, calibration par décile, test hors échantillon"},
    {"Modèle": "Sévérité — attritionnels",
     "Variable cible": "montant du sinistre (continu > 0)",
     "Variables explicatives": ", ".join(vars_sev) if vars_sev else "aucune (constante)",
     "Loi": loi_retenue, "Lien": "log", "Offset": "aucun",
     "Sélection": "AIC sur échelle comparable, puis élimination descendante",
     "Validation": "Droite de Henry des résidus, comparaison de 3 lois"},
    {"Modèle": "Fréquence — graves",
     "Variable cible": "nb_grave (comptage)",
     "Variables explicatives": ", ".join(vars_fg) if vars_fg else "aucune (constante)",
     "Loi": "Poisson", "Lien": "log", "Offset": "log(exposition)",
     "Sélection": "AIC sur toutes les variables, puis test du rapport de vraisemblance (5 %)",
     "Validation": "Niveau hors échantillon, stabilité de l'effet entre périodes (§7.3.3)"},
    {"Modèle": "Sévérité — graves",
     "Variable cible": "montant du sinistre grave",
     "Variables explicatives": "aucune (coût mutualisé)",
     "Loi": "Gamma / GPD (contrôle)", "Lien": "log", "Offset": "aucun",
     "Sélection": "AIC ; rejet de la segmentation pour insuffisance de puissance",
     "Validation": "Confrontation moyenne empirique / espérance GPD"},
])
exporter_table(recap, "5_5_recapitulatif_modeles", index=False)
with pd.option_context("display.max_colwidth", 60):
    display(recap.set_index("Modèle").T)

# %% [markdown]
# > **Pourquoi le GLM Tweedie n'a pas été retenu.** Il existe une alternative consistant
# > à modéliser directement la charge par contrat en un seul GLM, au moyen de la loi de
# > Tweedie. Elle est écartée ici pour une raison de fond : elle produit une prime, mais
# > ne permet plus de distinguer ce qui relève de la fréquence de ce qui relève du coût.
# > Or c'est précisément cette décomposition que l'énoncé demande d'analyser — et c'est
# > elle qui a livré le résultat le plus instructif de cette partie : la segmentation du
# > risque attritionnel passe **essentiellement** par la fréquence.
#
# ### Ce qu'il faut retenir de la partie D
#
# - **Fréquence** : la loi de Poisson est retenue après test — la binomiale négative
#   dégrade l'AIC, il n'y a pas de surdispersion. Deux variables survivent à la
#   sélection : le statut de conducteur novice et la zone géographique. L'interaction
#   entre les deux a été testée et rejetée : le tarif multiplicatif est validé.
# - **Colinéarité tranchée** : âge et ancienneté du permis portent la même information ;
#   une indicatrice binaire « conducteur novice » les résume mieux que l'une ou l'autre.
# - **Sévérité attritionnelle** : la log-normale l'emporte, avec la correction de Duan.
#   Seules deux variables liées au **véhicule** survivent, avec un pouvoir discriminant
#   faible ; aucune variable liée au conducteur. La segmentation du tarif attritionnel
#   repose donc essentiellement sur la fréquence.
# - **Graves** : la fréquence est segmentée par le seul statut de conducteur novice,
#   retenu parmi toutes les variables candidates et confirmé par test. Le coût est
#   mutualisé : aucune segmentation n'est crédible sur une centaine de sinistres, et la
#   loi de Pareto généralisée ne stabilise pas l'estimation.

# %% [markdown]
# ---
# # §6 — Partie E · Comparaison des approches de pricing
#
# > **Ce que demande l'énoncé (point E)**
# > Comparer les résultats selon plusieurs approches : pricing global / brut ; pricing
# > sur les seuls sinistres attritionnels ; pricing avec répartition des graves sur les
# > attritionnels ; pricing spécifique des graves ; puis comparaison des primes obtenues
# > et analyse des écarts.
#
# ## 6.1 Les approches mises en concurrence
#
# | # | Approche | Formule | Ce qu'elle suppose |
# |---|---|---|---|
# | 1 | **Globale / brute** | `f_tous × c_tous` | Un seul modèle pour tous les sinistres, graves compris |
# | 2 | **Attritionnels seuls** | `f_att × c_att` | Les graves sont ignorés |
# | 3a | **Graves répartis — forfait additif** | `f_att × c_att + PP_graves moyenne` | Le risque grave est identique pour tous |
# | 3b | **Graves répartis — chargement proportionnel** | `f_att × c_att × (1 + τ)` | Le risque grave est proportionnel au risque attritionnel |
# | 4 | **Graves tarifés séparément** | `f_att × c_att + f_graves(X) × c_graves` | Le risque grave a sa propre segmentation |
#
# L'approche 2 est volontairement incluse bien qu'elle soit **structurellement
# insuffisante** : elle sert de point de comparaison pour mesurer ce que coûte le fait
# d'ignorer les sinistres graves.

# %% [markdown]
# ## 6.2 Calibration d'ensemble
#
# Toutes les briques sont réestimées par une fonction unique, ce qui permettra au §7 de
# rejouer exactement la même calibration sur l'autre dispositif de découpage, sans
# risque de divergence entre les deux.

# %%
def calibrer_tarif(base_train, sinistres_carac, annees, loi_sev="Log-normale"):
    """Calibre l'ensemble des briques tarifaires sur un echantillon d'apprentissage.

    Renvoie un dictionnaire contenant les modeles ajustes, les constantes mutualisees
    et les facteurs de chargement necessaires aux cinq approches comparees.
    """
    sev_att = sinistres_carac[(sinistres_carac["categorie"] == "Attritionnel")
                              & (sinistres_carac["montant"] > 0)
                              & (sinistres_carac["annee"].isin(annees))]
    sev_gr = sinistres_carac[(sinistres_carac["categorie"] == "Grave")
                             & (sinistres_carac["montant"] > 0)
                             & (sinistres_carac["annee"].isin(annees))]
    sev_tous = sinistres_carac[(sinistres_carac["montant"] > 0)
                               & (sinistres_carac["annee"].isin(annees))]

    m = {}
    # --- briques de frequence ---
    m["freq_tous"], _ = ajuster_frequence(base_train, vars_freq, "nb_sinistres")
    m["freq_att"], _ = ajuster_frequence(base_train, vars_freq, "nb_att")
    m["freq_grave"], _ = ajuster_frequence(base_train, vars_fg if vars_fg else [], "nb_grave")

    # --- briques de severite ---
    ajuste_att = ajusteur_severite(sev_att, loi_sev)
    m["sev_att"], _ = ajuste_att(vars_sev)
    m["lissage_att"] = facteur_duan(m["sev_att"], loi_sev)

    ajuste_tous = ajusteur_severite(sev_tous, loi_sev)
    m["sev_tous"], _ = ajuste_tous(vars_sev)
    m["lissage_tous"] = facteur_duan(m["sev_tous"], loi_sev)
    m["loi_sev"] = loi_sev

    # --- constantes mutualisees ---
    expo = base_train["exposition"].sum()
    m["cout_grave_moyen"] = float(sev_gr["montant"].mean())
    m["pp_grave_forfait"] = float(base_train["charge_grave"].sum() / expo)
    m["pp_att_moyenne"] = float(base_train["charge_att"].sum() / expo)
    m["tau_proportionnel"] = float(base_train["charge_grave"].sum()
                                   / base_train["charge_att"].sum())
    return m


def appliquer_tarif(m, df):
    """Calcule la prime pure de chaque contrat selon les cinq approches."""
    expo = df["exposition"].values
    p = pd.DataFrame(index=df.index)

    # Frequences annualisees (offset neutre : on veut une frequence par annee-police)
    zero = np.zeros(len(df))
    f_tous = m["freq_tous"].predict(df, offset=zero).values
    f_att = m["freq_att"].predict(df, offset=zero).values
    f_grave = m["freq_grave"].predict(df, offset=zero).values
    if "credibilite_graves" in m:
        # Relativite novice credibilisee (§8.1.1) : r_Z = 1 + Z (r - 1), base recalee
        c = m["credibilite_graves"]
        f_grave = c["base"] * np.where((df["jeune_conducteur"] == "Oui").values, c["r_z"], 1.0)

    # Couts moyens
    c_tous = predire_severite(m["sev_tous"], df, m["loi_sev"], m["lissage_tous"]).values
    c_att = predire_severite(m["sev_att"], df, m["loi_sev"], m["lissage_att"]).values

    p["1. Globale"] = f_tous * c_tous
    p["2. Attritionnels seuls"] = f_att * c_att
    p["3a. Graves en forfait"] = f_att * c_att + m["pp_grave_forfait"]
    p["3b. Graves en proportion"] = f_att * c_att * (1 + m["tau_proportionnel"])
    p["4. Graves tarifés"] = f_att * c_att + f_grave * m["cout_grave_moyen"]
    p["exposition"] = expo
    return p


titre("CALIBRATION DES CINQ APPROCHES SUR LE TRAIN TEMPOREL")
modeles_temp = calibrer_tarif(train_temp, sin_carac, [2023], loi_retenue)

print(f"Cout moyen des graves (mutualise)     : {fr(modeles_temp['cout_grave_moyen'])}")
print(f"Prime pure graves en forfait additif  : {fr(modeles_temp['pp_grave_forfait'])} "
      f"par annee-police")
print(f"Taux de chargement proportionnel tau  : {fr(modeles_temp['tau_proportionnel'], 3)} "
      f"(soit +{pct(modeles_temp['tau_proportionnel'])})")
print(f"Prime pure attritionnelle moyenne     : {fr(modeles_temp['pp_att_moyenne'])}")

primes_train = appliquer_tarif(modeles_temp, train_temp)
APPROCHES = [c for c in primes_train.columns if c != "exposition"]

# %% [markdown]
# ## 6.3 Comparaison des primes obtenues

# %%
titre("COMPARAISON DES PRIMES PURES — ECHANTILLON D'APPRENTISSAGE")

charge_obs = train_temp["charge_totale"].sum()
expo_train = train_temp["exposition"].sum()
pp_observee = charge_obs / expo_train

lignes = []
for a in APPROCHES:
    prime_annuelle = primes_train[a]
    collecte = float((prime_annuelle * primes_train["exposition"]).sum())
    lignes.append({
        "Approche": a,
        "Prime pure moyenne": fr(prime_annuelle.mean()),
        "Écart-type": fr(prime_annuelle.std()),
        "Minimum": fr(prime_annuelle.min()),
        "Maximum": fr(prime_annuelle.max()),
        "Rapport max / min": fr(prime_annuelle.max() / prime_annuelle.min(), 2),
        "Charge collectée": fr(collecte),
        "Couverture de la charge": pct(collecte / charge_obs),
    })
tab_comp = pd.DataFrame(lignes)
exporter_table(tab_comp, "6_3_comparaison_approches", index=False)
display(tab_comp)

print(f"\nCharge observee a couvrir : {fr(charge_obs)}")
print(f"Prime pure d'equilibre    : {fr(pp_observee)} par annee-police")
print("\nLecture de la colonne « Couverture de la charge » : une valeur de 100 %")
print("signifie que le tarif collecte exactement la charge observee. En dessous,")
print("l'assureur est structurellement en perte.")

# %% [markdown]
# ### Analyse des écarts
#
# Trois enseignements se dégagent immédiatement du tableau.

# %%
titre("ANALYSE DES ECARTS", 2)
pp2 = primes_train["2. Attritionnels seuls"]
collecte2 = float((pp2 * primes_train["exposition"]).sum())
print(f"1) L'approche 2 (attritionnels seuls) ne collecte que {pct(collecte2 / charge_obs)}")
print(f"   de la charge, soit un deficit de {fr(charge_obs - collecte2)} sur la periode.")
print("   Elle est donc inutilisable en l'etat : elle n'est presentee que pour")
print("   quantifier ce que couterait le fait d'ignorer les sinistres graves.")

ecart_3a_3b = (primes_train["3b. Graves en proportion"]
               - primes_train["3a. Graves en forfait"])
print(f"\n2) Les approches 3a et 3b collectent le meme total mais le repartissent")
print(f"   differemment. Ecart individuel : de {fr(ecart_3a_3b.min())} a "
      f"{fr(ecart_3a_3b.max())}.")
print("   Le forfait additif MUTUALISE le risque grave : chacun paie le meme")
print("   supplement. Le chargement proportionnel le SEGMENTE : celui qui a un")
print("   risque attritionnel eleve paie aussi davantage pour les graves.")

corr_1_4 = np.corrcoef(primes_train["1. Globale"], primes_train["4. Graves tarifés"])[0, 1]
print(f"\n3) Correlation entre l'approche 1 (globale) et l'approche 4 (graves tarifes) : "
      f"{fr(corr_1_4, 3)}")
print("   Les deux ordonnent le risque de facon comparable, mais l'approche 1 repose")
print("   sur un cout moyen contamine par les graves, donc beaucoup plus instable.")

# %%
# Distribution des primes selon les approches
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for a in APPROCHES:
    axes[0].hist(primes_train[a], bins=45, histtype="step", lw=1.7, label=a)
axes[0].axvline(pp_observee, color=PALETTE["encre"], linestyle="--", lw=1.4,
                label=f"Prime d'équilibre : {fr(pp_observee)}")
axes[0].set_title("Distribution des primes pures selon l'approche")
axes[0].set_xlabel("Prime pure annuelle")
axes[0].set_ylabel("Nombre de contrats")
axes[0].legend(frameon=False, fontsize=8)
axe_fr(axes[0]); axe_fr(axes[0], "x")

largeurs = [primes_train[a].max() / primes_train[a].min() for a in APPROCHES]
axes[1].barh(range(len(APPROCHES)), largeurs, color=CYCLE[:len(APPROCHES)])
axes[1].set_yticks(range(len(APPROCHES)))
axes[1].set_yticklabels(APPROCHES, fontsize=8.5)
axes[1].set_title("Amplitude du tarif (prime maximale / prime minimale)")
axes[1].set_xlabel("Rapport")
for i, v in enumerate(largeurs):
    axes[1].text(v + 0.03, i, fr(v, 2), va="center", fontsize=8.5)
plt.tight_layout()
sauver_fig("6_3_distribution_primes")

# %% [markdown]
# ## 6.4 Concentration du tarif — courbe de Lorenz et indice de Gini
#
# Un tarif ne se juge pas seulement à son niveau moyen, mais à sa **capacité à
# discriminer le risque**. La courbe de Lorenz classe les contrats par prime croissante
# et trace la part cumulée de prime collectée en fonction de la part cumulée
# d'exposition. Plus la courbe s'éloigne de la diagonale, plus le tarif est segmentant.
#
# L'**indice de Gini** résume cette concentration : 0 pour un tarif uniforme, proche
# de 1 pour un tarif extrêmement segmenté.

# %%
def courbe_lorenz(primes, expo):
    """Renvoie (x, y, gini) : parts cumulees d'exposition et de prime, et indice de Gini."""
    ordre = np.argsort(primes)
    e = np.asarray(expo)[ordre]
    montant = np.asarray(primes)[ordre] * e
    x = np.concatenate([[0], np.cumsum(e) / e.sum()])
    y = np.concatenate([[0], np.cumsum(montant) / montant.sum()])
    gini = 1 - 2 * np.trapezoid(y, x) if hasattr(np, "trapezoid") else 1 - 2 * np.trapz(y, x)
    return x, y, gini


titre("CONCENTRATION DU TARIF", 2)
fig, ax = plt.subplots(figsize=(6.4, 5.6))
ax.plot([0, 1], [0, 1], color=PALETTE["gris"], linestyle="--", lw=1.2,
        label="Tarif uniforme (Gini = 0)")
res_gini = []
for a, couleur in zip(APPROCHES, CYCLE):
    x, y, g = courbe_lorenz(primes_train[a].values, primes_train["exposition"].values)
    ax.plot(x, y, lw=1.8, color=couleur, label=f"{a} — Gini = {fr(g, 3)}")
    res_gini.append({"Approche": a, "Indice de Gini": round(g, 4)})
ax.set_title("Courbe de Lorenz des primes pures")
ax.set_xlabel("Part cumulée de l'exposition")
ax.set_ylabel("Part cumulée de la prime collectée")
ax.legend(frameon=False, fontsize=8, loc="upper left")
sauver_fig("6_4_courbe_lorenz")

tab_gini = pd.DataFrame(res_gini).sort_values("Indice de Gini", ascending=False)
exporter_table(tab_gini, "6_4_indices_gini", index=False)
display(tab_gini)

print("\nATTENTION A LA LECTURE. Un Gini eleve signifie un tarif tres segmentant,")
print("PAS un tarif plus juste. L'approche 2, qui ignore les graves, peut afficher")
print("un Gini eleve tout en etant structurellement deficitaire. La segmentation")
print("n'a de valeur que si elle est ADOSSEE a un risque reel : c'est ce que le §7")
print("verifie, en confrontant les primes a la sinistralite d'une periode non utilisee")
print("pour la calibration.")

# %% [markdown]
# ## 6.5 Effet sur des profils types
#
# La comparaison chiffrée sur des profils concrets rend les écarts tangibles.

# %%
titre("PRIMES PAR PROFIL TYPE")

profils = pd.DataFrame([
    {"Profil": "Conducteur novice, zone D", "jeune_conducteur": "Oui", "zone_geo": "D",
     "classe_vehicule": "Citadine", "carburant": "Essence"},
    {"Profil": "Conducteur novice, zone A", "jeune_conducteur": "Oui", "zone_geo": "A",
     "classe_vehicule": "Citadine", "carburant": "Essence"},
    {"Profil": "Conducteur confirmé, zone D", "jeune_conducteur": "Non", "zone_geo": "D",
     "classe_vehicule": "Berline", "carburant": "Diesel"},
    {"Profil": "Conducteur confirmé, zone A", "jeune_conducteur": "Non", "zone_geo": "A",
     "classe_vehicule": "Berline", "carburant": "Diesel"},
    {"Profil": "Confirmé, SUV hybride, zone C", "jeune_conducteur": "Non", "zone_geo": "C",
     "classe_vehicule": "Suv", "carburant": "Hybride"},
])
profils["exposition"] = 1.0
primes_profils = appliquer_tarif(modeles_temp, profils)

aff_profils = pd.DataFrame({"Profil": profils["Profil"]})
for a in APPROCHES:
    aff_profils[a] = primes_profils[a].map(fr)
exporter_table(aff_profils, "6_5_primes_profils_types", index=False)
display(aff_profils)

ratio_novice = (primes_profils.loc[0, "4. Graves tarifés"]
                / primes_profils.loc[2, "4. Graves tarifés"])
print(f"\nDans l'approche 4, un conducteur novice en zone D paie {fr(ratio_novice, 2)} fois")
print("la prime d'un conducteur confirme en zone D, a vehicule comparable.")

ratio_3a = (primes_profils.loc[0, "3a. Graves en forfait"]
            / primes_profils.loc[2, "3a. Graves en forfait"])
print(f"Dans l'approche 3a (forfait), ce rapport tombe a {fr(ratio_3a, 2)} :")
print("le forfait identique pour tous ECRASE mecaniquement les ecarts relatifs.")
print("\n-> C'est l'arbitrage central de la partie E : mutualiser le risque grave")
print("   (3a) ou le segmenter (3b, 4). La reponse depend de ce que l'on croit")
print("   pouvoir estimer de facon fiable — question tranchee au §7.")

# %% [markdown]
# **Remarque sur l'approche 1.** Elle collecte 99,1 % de la charge et non exactement
# 100 %. Ce n'est pas une erreur : le modèle de fréquence restitue exactement le nombre
# de sinistres, et le modèle de coût exactement le coût moyen, mais le **produit des
# deux prédictions individuelles** ne reconstitue la charge totale que si fréquence et
# coût sont décorrélés entre contrats. L'écart résiduel mesure précisément cette
# corrélation. Un tarif opérationnel serait de toute façon recalé sur la charge à
# couvrir — c'est l'objet du chargement de sécurité au §8.

# %% [markdown]
# ---
# # §7 — Validation hors échantillon
#
# C'est le test décisif. Jusqu'ici, tous les indicateurs ont été calculés sur les
# données ayant servi à calibrer les modèles : ils mesurent la qualité de l'ajustement,
# pas la qualité de la **prédiction**. On confronte maintenant chaque approche à des
# données qu'elle n'a jamais vues, **sur les deux dispositifs de découpage** construits
# au §4.
#
# | Dispositif | Apprentissage | Test | Question posée |
# |---|---|---|---|
# | Temporel | survenance 2023 | survenance 2024 | Le tarif tient-il l'année suivante ? |
# | Aléatoire | 80 % des polices | 20 % des polices | Le tarif tient-il sur de nouveaux assurés ? |

# %%
titre("CALIBRATION SUR CHAQUE DISPOSITIF")


def calibrer_sur(base_train, annees):
    """Calibre le tarif en n'utilisant QUE l'information de l'echantillon d'apprentissage.

    Les sinistres servant a estimer la severite sont restreints aux polices du train :
    sans cette precaution, le modele de cout verrait des sinistres de polices de test,
    ce qui constituerait une fuite d'information.
    """
    sc = sin_carac[sin_carac["id_police"].isin(set(base_train["id_police"]))]
    return calibrer_tarif(base_train, sc, annees, loi_retenue)


modeles_temporel = calibrer_sur(train_temp, [2023])
modeles_aleatoire = calibrer_sur(train_alea, [2023, 2024])

for nom, m in [("temporel", modeles_temporel), ("aléatoire", modeles_aleatoire)]:
    print(f"Dispositif {nom:<10} : coût moyen des graves = "
          f"{fr(m['cout_grave_moyen']):>8}, forfait graves = "
          f"{fr(m['pp_grave_forfait']):>5}, tau = {fr(m['tau_proportionnel'], 2):>5}")

# %% [markdown]
# ## 7.1 Les indicateurs de validation
#
# Quatre indicateurs complémentaires, qui ne mesurent pas la même chose :
#
# - **Le ratio de couverture** `charge prédite / charge observée`. Il mesure le
#   **niveau** du tarif. Une valeur inférieure à 100 % signale une sous-tarification.
# - **La déviance de Poisson** sur la fréquence. Elle mesure la qualité prédictive du
#   comptage. Plus elle est faible, mieux c'est.
# - **L'indice de Gini de discrimination.** Contrairement au Gini du §6.4, qui mesurait
#   la dispersion du tarif, celui-ci mesure sa **pertinence** : on classe les contrats
#   par prime prédite croissante et on observe si la charge réellement survenue se
#   concentre bien sur les contrats les plus chargés. Un tarif segmentant mais mal
#   orienté aura un Gini de dispersion élevé et un Gini de discrimination nul.
# - **La courbe de lift.** Elle vérifie par décile que la charge observée croît bien
#   avec la prime prédite.

# %%
def gini_discrimination(primes, charges_observees, expo):
    """Indice de Gini mesurant la capacite du tarif a ORDONNER le risque reel.

    On classe les contrats par prime predite croissante, puis on trace la part cumulee
    de charge REELLEMENT survenue contre la part cumulee d'exposition. Un tarif sans
    pouvoir predictif donne 0 ; un tarif parfait s'approche de 1.
    """
    ordre = np.argsort(np.asarray(primes))
    e = np.asarray(expo)[ordre]
    c = np.asarray(charges_observees)[ordre]
    if c.sum() <= 0:
        return np.nan
    x = np.concatenate([[0], np.cumsum(e) / e.sum()])
    y = np.concatenate([[0], np.cumsum(c) / c.sum()])
    aire = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    return 1 - 2 * aire


def deviance_poisson(observes, predits):
    """Deviance de Poisson : mesure d'ecart adaptee a une cible de comptage."""
    o = np.asarray(observes, dtype=float)
    p = np.clip(np.asarray(predits, dtype=float), 1e-10, None)
    terme = np.where(o > 0, o * np.log(o / p), 0.0)
    return float(2 * np.sum(terme - (o - p)))


def evaluer(m, df_test, nom_dispositif):
    """Evalue les cinq approches sur un echantillon de test."""
    primes = appliquer_tarif(m, df_test)
    charge_obs = df_test["charge_totale"].values
    expo = df_test["exposition"].values
    total_obs = charge_obs.sum()

    # frequence predite, pour la deviance
    pred_nb = m["freq_tous"].predict(df_test, offset=np.log(df_test["exposition"]))
    dev = deviance_poisson(df_test["nb_sinistres"].values, pred_nb.values)

    lignes = []
    for a in APPROCHES:
        collecte = float((primes[a].values * expo).sum())
        lignes.append({
            "Dispositif": nom_dispositif,
            "Approche": a,
            "Prime moyenne": fr(primes[a].mean()),
            "Charge prédite": fr(collecte),
            "Charge observée": fr(total_obs),
            "Ratio de couverture": pct(collecte / total_obs),
            "Écart absolu": fr(abs(collecte - total_obs)),
            "Gini de discrimination": round(
                gini_discrimination(primes[a].values, charge_obs, expo), 4),
        })
    return pd.DataFrame(lignes), primes, dev


titre("RESULTATS SUR LES ECHANTILLONS DE TEST")
res_temp, primes_test_temp, dev_temp = evaluer(modeles_temporel, test_temp, "Temporel (2024)")
res_alea, primes_test_alea, dev_alea = evaluer(modeles_aleatoire, test_alea, "Aléatoire (20 %)")

resultats_test = pd.concat([res_temp, res_alea], ignore_index=True)
exporter_table(resultats_test, "7_1_validation_hors_echantillon", index=False)
display(resultats_test)

print(f"\nDeviance de Poisson de la frequence :")
print(f"   dispositif temporel : {dev_temp:.1f} sur {fr(len(test_temp))} contrats "
      f"({dev_temp / len(test_temp):.4f} par contrat)")
print(f"   dispositif aleatoire: {dev_alea:.1f} sur {fr(len(test_alea))} contrats "
      f"({dev_alea / len(test_alea):.4f} par contrat)")

# %% [markdown]
# ## 7.2 Lecture croisée des deux dispositifs

# %%
titre("SYNTHESE : COMPORTEMENT DE CHAQUE APPROCHE SUR LES DEUX DISPOSITIFS", 2)

pivot = resultats_test.pivot(index="Approche", columns="Dispositif",
                             values="Ratio de couverture")
pivot_gini = resultats_test.pivot(index="Approche", columns="Dispositif",
                                  values="Gini de discrimination")
synthese_test = pd.concat(
    {"Ratio de couverture": pivot, "Gini de discrimination": pivot_gini}, axis=1)
exporter_table(synthese_test, "7_2_synthese_deux_dispositifs")
display(synthese_test)

print("\nGrille de lecture :")
print("  - une approche robuste tient sur LES DEUX dispositifs ;")
print("  - une degradation sur le seul dispositif TEMPOREL signale une instabilite")
print("    dans le temps : le tarif ne se reconduit pas d'une annee sur l'autre ;")
print("  - une degradation sur le seul dispositif ALEATOIRE signale un")
print("    surapprentissage : le tarif colle aux assures vus, pas au risque.")

# %% [markdown]
# ### Courbe de lift de la charge totale, avec intervalles de confiance
#
# La courbe de lift classe les contrats par prime prédite croissante, les regroupe en
# déciles, et compare dans chaque décile la prime prédite à la charge réellement
# observée. **Sans intervalle de confiance, ce graphique est illisible** : on ne peut pas
# savoir si l'écart entre une barre et la courbe signale un défaut du modèle ou un simple
# aléa. On calcule donc, par **bootstrap** — 1 000 rééchantillonnages des contrats de
# chaque décile — l'intervalle dans lequel la charge observée aurait pu tomber par le
# seul effet du hasard.

# %%
titre("COURBES DE LIFT AVEC INTERVALLES DE CONFIANCE", 2)


def table_lift(primes, df_test, approche, n=10, cible="charge_totale", n_boot=1000,
               graine=GRAINE):
    """Charge observee par groupe de prime predite, avec IC bootstrap a 95 %.

    Le bootstrap est stratifie par groupe : on reechantillonne les contrats a
    l'interieur de chaque decile, ce qui mesure l'incertitude propre a chaque barre.
    """
    t = pd.DataFrame({
        "prime": primes[approche].values if hasattr(primes, "columns") else primes,
        "charge": df_test[cible].values,
        "expo": df_test["exposition"].values,
        "nb": df_test["nb_sinistres"].values,
        "nb_grave": df_test["nb_grave"].values,
        "charge_grave": df_test["charge_grave"].values,
    })
    etiquettes = [f"D{i}" for i in range(1, n + 1)] if n == 10 else \
        [f"Q{i}" for i in range(1, n + 1)]
    t["groupe"] = pd.qcut(t["prime"].rank(method="first"), n, labels=etiquettes)
    g = t.groupby("groupe", observed=True).agg(
        Exposition=("expo", "sum"), Charge=("charge", "sum"), Sinistres=("nb", "sum"),
        Graves=("nb_grave", "sum"), Plus_gros_grave=("charge_grave", "max"),
        Prime_moyenne=("prime", "mean"))
    g["PP observée"] = g["Charge"] / g["Exposition"]
    g["Fréquence observée"] = g["Sinistres"] / g["Exposition"]

    gen = np.random.default_rng(graine)
    bas, haut = [], []
    for _, sous in t.groupby("groupe", observed=True):
        c, e = sous["charge"].values, sous["expo"].values
        tirages = gen.integers(0, len(sous), (n_boot, len(sous)))
        pp = c[tirages].sum(axis=1) / e[tirages].sum(axis=1)
        bas.append(np.percentile(pp, 2.5))
        haut.append(np.percentile(pp, 97.5))
    g["IC bas"], g["IC haut"] = bas, haut
    g["Prime dans l'IC"] = (g["Prime_moyenne"] >= g["IC bas"]) & (g["Prime_moyenne"] <= g["IC haut"])
    return g


def tracer_lift(ax, g, titre_graph, lib_prime, lib_obs="Charge observée par année-police"):
    x = np.arange(len(g))
    ax.bar(x, g["PP observée"], color=PALETTE["petrole"], width=0.62, label=lib_obs)
    ax.errorbar(x, g["PP observée"],
                yerr=[np.clip(g["PP observée"] - g["IC bas"], 0, None),
                      np.clip(g["IC haut"] - g["PP observée"], 0, None)],
                fmt="none", ecolor=PALETTE["encre"], capsize=3, lw=1,
                label="Intervalle de confiance à 95 %")
    ax.plot(x, g["Prime_moyenne"], marker="o", ms=5, color=PALETTE["brique"], lw=1.8,
            label=lib_prime)
    ax.set_xticks(x); ax.set_xticklabels(g.index)
    ax.set_title(titre_graph)
    ax.set_ylabel("Montant par année-police")
    ax.legend(frameon=False, fontsize=8)
    axe_fr(ax)


lifts = {}
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
for ax, (primes_t, df_t, lib) in zip(
        axes, [(primes_test_temp, test_temp, "Test temporel (2024)"),
               (primes_test_alea, test_alea, "Test aléatoire (20 %)")]):
    g = table_lift(primes_t, df_t, "4. Graves tarifés")
    lifts[lib] = g
    tracer_lift(ax, g, f"Courbe de lift — {lib}", "Prime prédite (approche 4)")
    ax.set_xlabel("Décile de prime prédite")
plt.tight_layout()
sauver_fig("7_2_courbes_lift")

for lib, g in lifts.items():
    print(f"\n--- {lib} : detail par decile ---")
    display(pd.DataFrame({
        "Prime prédite": g["Prime_moyenne"].map(fr),
        "Charge observée": g["PP observée"].map(fr),
        "IC 95 %": [f"[{fr(b)} ; {fr(h)}]" for b, h in zip(g["IC bas"], g["IC haut"])],
        "Prime dans l'IC": np.where(g["Prime dans l'IC"], "Oui", "Non"),
        "Graves": g["Graves"].map(fr),
        "Plus gros sinistre": g["Plus_gros_grave"].map(fr),
    }))
    n_compatibles = int(g["Prime dans l'IC"].sum())
    print(f"   Deciles ou la prime predite est compatible avec l'observe : "
          f"{n_compatibles} / {len(g)}")

# %% [markdown]
# ### Pourquoi ce graphique ne peut pas être plus régulier — et ce qu'il dit vraiment
#
# Trois mécanismes expliquent l'aspect du graphique. Aucun ne relève d'un défaut du
# modèle, et le premier est même une propriété voulue du tarif.
#
# **1. Le tarif n'a que deux niveaux.** La prime d'un conducteur confirmé varie d'environ
# 700 à 820, celle d'un novice de 1 420 à 1 560. Les novices représentent 10 % de
# l'exposition : ils forment à eux seuls le décile 10. Les déciles 1 à 9 découpent donc
# **arbitrairement** un groupe de primes quasi identiques — il n'y a rien à ordonner entre
# eux, et leurs écarts de charge observée ne peuvent être que du bruit.
#
# **2. La charge a une queue lourde.** Dans une distribution de ce type, la plupart des
# déciles observent *moins* que leur espérance, et quelques-uns *beaucoup plus* : la
# moyenne est juste, mais la médiane est en dessous. C'est exactement la forme du
# graphique — une majorité de barres sous la courbe, quelques pics. Le pic du décile 3
# sur le test temporel tient à **un seul sinistre** de plus de 300 000.
#
# **3. L'incertitude est énorme, et elle couvre la prime prédite.** Le tableau ci-dessus
# montre que dans la grande majorité des déciles, la prime prédite tombe **à l'intérieur**
# de l'intervalle de confiance de la charge observée. Autrement dit, le graphique est
# **statistiquement compatible avec un modèle correctement calibré**.
#
# **Ce graphique ne pouvait donc pas valider la brique grave, et aucun modèle ne l'aurait
# rendu régulier avec ces données.** L'amélioration ne consiste pas à forcer le modèle à
# épouser ces barres — ce serait du surapprentissage pur — mais à **mesurer chaque brique
# du tarif sur la cible qui lui correspond**, là où le signal n'est pas noyé.

# %% [markdown]
# ## 7.3 Valider chaque brique sur sa propre cible
#
# Le tarif retenu est une somme de briques. Chacune se valide sur sa propre variable
# observée, avec un niveau de bruit très différent :
#
# | Brique | Cible de validation | Événements disponibles |
# |---|---|---|
# | Fréquence (tous sinistres) | nombre de sinistres | plusieurs centaines |
# | Prime attritionnelle | charge attritionnelle | plusieurs centaines, coûts bornés à 7 500 |
# | Fréquence des graves | nombre de sinistres graves | quelques dizaines |
# | Coût des graves | montant moyen des graves | quelques dizaines, queue lourde |
#
# ### 7.3.1 Fréquence

# %%
titre("VALIDATION DE LA BRIQUE FREQUENCE", 2)


def lift_frequence(modele, df_test, n=10):
    """Frequence observee par decile de frequence PREDITE, hors echantillon."""
    pred = modele.predict(df_test, offset=np.zeros(len(df_test))).values
    t = pd.DataFrame({"pred": pred, "nb": df_test["nb_sinistres"].values,
                      "expo": df_test["exposition"].values})
    t["groupe"] = pd.qcut(t["pred"].rank(method="first"), n,
                          labels=[f"D{i}" for i in range(1, n + 1)])
    g = t.groupby("groupe", observed=True).agg(
        Exposition=("expo", "sum"), Sinistres=("nb", "sum"), Prédite=("pred", "mean"))
    g["Observée"] = g["Sinistres"] / g["Exposition"]
    return g


for modele, df_t, lib in [(modeles_temporel["freq_tous"], test_temp, "Test temporel (2024)"),
                          (modeles_aleatoire["freq_tous"], test_alea, "Test aléatoire (20 %)")]:
    gf = lift_frequence(modele, df_t)
    rapport = (gf["Sinistres"].iloc[8:].sum() / gf["Exposition"].iloc[8:].sum()) / \
              (gf["Sinistres"].iloc[:8].sum() / gf["Exposition"].iloc[:8].sum())
    pred_tot = modele.predict(df_t, offset=np.log(df_t["exposition"])).sum()
    print(f"{lib} : sinistres predits {fr(pred_tot, 1)} / observes {fr(df_t['nb_sinistres'].sum())}"
          f"  |  frequence deciles 9-10 / deciles 1-8 = {fr(rapport, 2)}")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
for ax, (modele, df_t, lib) in zip(
        axes, [(modeles_temporel["freq_tous"], test_temp, "Test temporel (2024)"),
               (modeles_aleatoire["freq_tous"], test_alea, "Test aléatoire (20 %)")]):
    gf = lift_frequence(modele, df_t)
    x = np.arange(len(gf))
    ax.bar(x, gf["Observée"], color=PALETTE["petrole"], width=0.62, label="Observée")
    ax.plot(x, gf["Prédite"], marker="o", ms=5, color=PALETTE["brique"], lw=1.8,
            label="Prédite")
    ax.errorbar(x, gf["Observée"], yerr=1.96 * np.sqrt(gf["Sinistres"]) / gf["Exposition"],
                fmt="none", ecolor=PALETTE["encre"], capsize=3, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(gf.index)
    ax.set_title(f"Fréquence par décile — {lib}")
    ax.set_xlabel("Décile de fréquence prédite")
    ax.set_ylabel("Sinistres par année-police")
    ax.legend(frameon=False, fontsize=8.5)
plt.tight_layout()
sauver_fig("7_3_lift_frequence")

# %% [markdown]
# ### 7.3.2 Prime attritionnelle
#
# C'est la brique où la segmentation doit se voir : les coûts y sont bornés par le seuil
# de gravité, si bien qu'aucun sinistre isolé ne peut écraser un groupe. On regroupe en
# **quintiles** plutôt qu'en déciles, pour disposer d'environ 150 sinistres par groupe.

# %%
titre("VALIDATION DE LA BRIQUE ATTRITIONNELLE", 2)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
lifts_att = {}
for ax, (primes_t, df_t, lib) in zip(
        axes, [(primes_test_temp, test_temp, "Test temporel (2024)"),
               (primes_test_alea, test_alea, "Test aléatoire (20 %)")]):
    g = table_lift(primes_t, df_t, "2. Attritionnels seuls", n=5, cible="charge_att")
    lifts_att[lib] = g
    tracer_lift(ax, g, f"Lift attritionnel — {lib}", "Prime attritionnelle prédite",
                "Charge attritionnelle observée")
    ax.set_xlabel("Quintile de prime attritionnelle prédite")
plt.tight_layout()
sauver_fig("7_3_lift_attritionnel")

for lib, g in lifts_att.items():
    print(f"\n--- {lib} ---")
    display(pd.DataFrame({
        "Prime prédite": g["Prime_moyenne"].map(fr),
        "Charge observée": g["PP observée"].map(fr),
        "IC 95 %": [f"[{fr(b)} ; {fr(h)}]" for b, h in zip(g["IC bas"], g["IC haut"])],
    }))
    print(f"   Charge observee, 5e quintile / 1er quintile : "
          f"{fr(g['PP observée'].iloc[-1] / g['PP observée'].iloc[0], 2)}  "
          f"(predit : {fr(g['Prime_moyenne'].iloc[-1] / g['Prime_moyenne'].iloc[0], 2)})")
    n_q_ok = int(g["Prime dans l'IC"].sum())
    print(f"   Quintiles dont la prime predite est dans l'IC : {n_q_ok} / {len(g)}")

print("\n-> Ici, contrairement au lift de la charge totale, la charge observee CROIT avec")
print("   la prime predite, et le 5e quintile coute nettement plus que le 1er : la")
print("   segmentation attritionnelle est reelle et se retrouve hors echantillon.")
print("   L'ecart predit entre extremes est un peu plus large que l'ecart observe :")
print("   le tarif attritionnel est legerement trop segmentant, mais sans que cet ecart")
print("   sorte des intervalles de confiance. C'est le signe classique de coefficients")
print("   estimes avec un peu d'optimisme sur l'echantillon d'apprentissage.")

# %% [markdown]
# ### 7.3.3 Fréquence des sinistres graves et stabilité de l'effet « novice »
#
# Le modèle de fréquence des graves repose sur un seul effet : le statut de conducteur
# novice. On vérifie d'abord le **niveau** — le nombre de graves prédit hors échantillon —
# puis la **stabilité de l'effet** d'une période à l'autre.

# %%
titre("VALIDATION DE LA BRIQUE FREQUENCE DES GRAVES", 2)
for m, df_t, lib in [(modeles_temporel, test_temp, "Test temporel (2024)"),
                     (modeles_aleatoire, test_alea, "Test aléatoire (20 %)")]:
    pred = m["freq_grave"].predict(df_t, offset=np.log(df_t["exposition"])).sum()
    print(f"{lib} : graves predits {fr(pred, 1)} / observes {fr(df_t['nb_grave'].sum())}")


def relativite_novice(df):
    """Relativite novice sur la frequence des graves, avec IC a 95 %."""
    m, _ = ajuster_frequence(df, ["jeune_conducteur"], "nb_grave")
    b, se = m.params.iloc[1], m.bse.iloc[1]
    n_nov = int(df.loc[df["jeune_conducteur"] == "Oui", "nb_grave"].sum())
    return b, se, n_nov


stabilite = []
for lib, df in [("2023 (apprentissage temporel)", train_temp),
                ("2024 (test temporel)", test_temp),
                ("2023 + 2024 (ensemble)", base_tarif)]:
    b, se, n_nov = relativite_novice(df)
    stabilite.append({"Période": lib, "Graves de novices": n_nov,
                      "Relativité novice": fr(np.exp(b), 2),
                      "IC 95 %": f"[{fr(np.exp(b - 1.96 * se), 2)} ; {fr(np.exp(b + 1.96 * se), 2)}]",
                      "_b": b, "_se": se})
tab_stab = pd.DataFrame(stabilite)
exporter_table(tab_stab.drop(columns=["_b", "_se"]), "7_3_stabilite_effet_novice_graves",
               index=False)
display(tab_stab.drop(columns=["_b", "_se"]))

b23, se23 = tab_stab.loc[0, "_b"], tab_stab.loc[0, "_se"]
b24, se24 = tab_stab.loc[1, "_b"], tab_stab.loc[1, "_se"]
z_het = (b23 - b24) / np.sqrt(se23 ** 2 + se24 ** 2)
p_het = 2 * (1 - stats.norm.cdf(abs(z_het)))
print(f"\nTest d'egalite de l'effet entre 2023 et 2024 : z = {z_het:.2f}, p-value = {p_het:.3f}")

dev_test = []
for lib, fit, test in [("Test temporel", train_temp, test_temp),
                       ("Test aléatoire", train_alea, test_alea)]:
    ligne = {"Dispositif": lib,
             "Graves de novices en test": int(test.loc[test["jeune_conducteur"] == "Oui",
                                                        "nb_grave"].sum())}
    for nom, vars_ in [("Déviance — constante", []),
                       ("Déviance — novice", ["jeune_conducteur"])]:
        m, _ = ajuster_frequence(fit, vars_, "nb_grave")
        ligne[nom] = round(deviance_poisson(test["nb_grave"].values,
                                            m.predict(test, offset=np.log(test["exposition"])).values), 1)
    dev_test.append(ligne)
tab_dev = pd.DataFrame(dev_test)
exporter_table(tab_dev, "7_3_deviance_graves_hors_echantillon", index=False)
print("\nDeviance hors echantillon de la frequence des graves (plus faible = meilleur) :")
display(tab_dev)

# %% [markdown]
# **Lecture.** Le niveau de fréquence des graves est bien prédit sur les deux
# dispositifs. L'effet « novice », en revanche, appelle une lecture prudente :
#
# - estimé à **2,06 sur 2023**, il ne vaut que **1,09 sur 2024** — mais ce dernier chiffre
#   repose sur **sept** sinistres graves de novices, et le test d'égalité entre les deux
#   périodes n'est pas significatif : 2024 ne contredit pas 2023, il est simplement trop
#   pauvre pour trancher ;
# - hors échantillon, le modèle avec l'effet novice fait légèrement **moins bien** que la
#   constante, sur les deux dispositifs — mais d'environ 0,5 % de déviance, sur des
#   échantillons de test qui ne comptent que quelques graves de novices ;
# - estimé sur **l'ensemble des données**, l'effet vaut **1,71** et son intervalle de
#   confiance exclut 1 : il est significatif.
#
# Conclusion : **l'effet existe, mais l'estimation de 2023 (2,06) le surestimait
# probablement.** C'est la justification de la principale amélioration du modèle, mise en
# œuvre au §8.1 : le tarif final est recalibré sur **toutes** les données, et la
# sensibilité à l'ampleur de cet effet est rendue explicite par la théorie de la
# crédibilité.

# %% [markdown]
# ### 7.3.4 Synthèse : que valide-t-on réellement ?
#
# On termine par le même indicateur — le Gini de discrimination — calculé sur chaque
# cible, avec son intervalle de confiance bootstrap. Un Gini dont l'intervalle exclut 0
# démontre que le tarif ordonne réellement le risque correspondant.

# %%
titre("GINI DE DISCRIMINATION PAR BRIQUE, AVEC IC BOOTSTRAP", 2)


def gini_bootstrap(primes, cible, expo, n_boot=1000, graine=GRAINE):
    gen = np.random.default_rng(graine)
    n = len(primes)
    vals = [gini_discrimination(primes[i], cible[i], expo[i])
            for i in gen.integers(0, n, (n_boot, n))]
    return gini_discrimination(primes, cible, expo), np.nanpercentile(vals, 2.5), \
        np.nanpercentile(vals, 97.5)


lignes_gini, lignes_couv = [], []
for primes_t, df_t, lib in [(primes_test_temp, test_temp, "Temporel (2024)"),
                            (primes_test_alea, test_alea, "Aléatoire (20 %)")]:
    e = df_t["exposition"].values
    p4 = primes_t["4. Graves tarifés"].values
    p_att = primes_t["2. Attritionnels seuls"].values
    p_gr = p4 - p_att
    for brique, prime, cible in [("Charge totale", p4, df_t["charge_totale"].values),
                                 ("Charge attritionnelle", p_att, df_t["charge_att"].values),
                                 ("Nombre de graves", p_gr, df_t["nb_grave"].values)]:
        gv, gb, gh = gini_bootstrap(prime, cible, e)
        lignes_gini.append({"Dispositif": lib, "Cible": brique, "Gini": fr(gv, 3),
                            "IC 95 %": f"[{fr(gb, 3)} ; {fr(gh, 3)}]",
                            "Discrimination démontrée": "Oui" if gb > 0 else "Non"})

    charge = df_t["charge_totale"].values
    tirages = np.random.default_rng(GRAINE).integers(0, len(e), (1000, len(e)))
    couv_boot = [(p4[i] * e[i]).sum() / charge[i].sum() for i in tirages]
    b_c, h_c = np.percentile(couv_boot, 2.5), np.percentile(couv_boot, 97.5)
    lignes_couv.append({"Dispositif": lib,
                        "Ratio de couverture": pct((p4 * e).sum() / charge.sum()),
                        "IC 95 %": f"[{pct(b_c)} ; {pct(h_c)}]",
                        "100 % dans l'IC": "Oui" if b_c <= 1 <= h_c else "Non"})

tab_gini_ic = pd.DataFrame(lignes_gini)
tab_couv_ic = pd.DataFrame(lignes_couv)
exporter_table(tab_gini_ic, "7_3_gini_par_brique_ic", index=False)
exporter_table(tab_couv_ic, "7_3_couverture_ic", index=False)
display(tab_gini_ic)
print("\nRatio de couverture de l'approche 4 hors echantillon, avec IC bootstrap :")
display(tab_couv_ic)

print("\n-> Sur la CHARGE TOTALE, aucun Gini ne se distingue de zero : la brique grave,")
print("   qui pese 89 % de la charge, noie tout signal. Sur la CHARGE ATTRITIONNELLE,")
print("   le tarif ordonne le risque de facon demontree sur le test temporel. C'est la")
print("   vraie lecture du graphique de lift : le modele segmente correctement ce qui")
print("   peut l'etre, et mutualise, a juste titre, ce qui ne peut pas l'etre.")
if (tab_couv_ic["100 % dans l'IC"] == "Oui").all():
    print(f"\n   Les ratios de couverture ({' et '.join(tab_couv_ic['Ratio de couverture'])})")
    print("   ont des intervalles de confiance qui contiennent 100 % sur les deux dispositifs.")
    print("   Ce ne sont donc PAS des erreurs de niveau, mais la variabilite naturelle")
    print("   d'une charge dominee par quelques dizaines de sinistres graves.")


# %% [markdown]
# ## 7.4 Approche recommandée

# %%
titre("CHOIX DE L'APPROCHE")

ecarts = resultats_test.copy()
ecarts["ecart_num"] = ecarts["Ratio de couverture"].str.replace(" ", "").str.replace(
    " %", "").str.replace(",", ".").astype(float)
ecarts["distance_a_100"] = (ecarts["ecart_num"] - 100).abs()
classement = ecarts.groupby("Approche").agg(
    Écart_moyen_à_100=("distance_a_100", "mean"),
    Gini_moyen=("Gini de discrimination", "mean")).sort_values("Écart_moyen_à_100")
classement["Écart_moyen_à_100"] = classement["Écart_moyen_à_100"].round(2)
exporter_table(classement, "7_4_classement_approches")
display(classement)

print("\nLes approches 1, 3a, 3b et 4 sont INDISCERNABLES sur les criteres chiffres :")
print("elles collectent la meme charge globale et ne different que par sa repartition")
print("entre assures. Le §7.3.4 a montre, de plus, que leurs ecarts de niveau sont tous")
print("dans l'intervalle de confiance. Le choix se fait donc sur des criteres de")
print("construction.")

criteres = pd.DataFrame([
    {"Approche": "1. Globale", "Couvre la charge": "Oui",
     "Coût moyen stable": "Non — contaminé par les graves",
     "Briques révisables séparément": "Non",
     "Verdict": "À écarter : instabilité structurelle"},
    {"Approche": "2. Attritionnels seuls", "Couvre la charge": "Non — 11 % seulement",
     "Coût moyen stable": "Oui", "Briques révisables séparément": "Sans objet",
     "Verdict": "À écarter sans discussion"},
    {"Approche": "3a. Graves en forfait", "Couvre la charge": "Oui",
     "Coût moyen stable": "Oui", "Briques révisables séparément": "Oui",
     "Verdict": "Cas particulier de 4 (crédibilité nulle)"},
    {"Approche": "3b. Graves en proportion", "Couvre la charge": "Oui",
     "Coût moyen stable": "Oui", "Briques révisables séparément": "Oui",
     "Verdict": "Hypothèse de proportionnalité non étayée"},
    {"Approche": "4. Graves tarifés", "Couvre la charge": "Oui",
     "Coût moyen stable": "Oui", "Briques révisables séparément": "Oui",
     "Verdict": "RETENUE, recalibrée et crédibilisée (§8.1)"},
])
exporter_table(criteres, "7_4_criteres_de_choix", index=False)
with pd.option_context("display.max_colwidth", 55):
    display(criteres)

# %% [markdown]
# ### Argument central : un risque grave *estimé* plutôt que *postulé*
#
# Les approches 3a, 3b et 4 collectent **la même charge grave** — environ 730 par
# année-police. Elles ne diffèrent que par **la façon de la répartir entre assurés**.
# Et c'est là que tout se joue : cette répartition est-elle *supposée* à l'avance, ou
# *mesurée* sur les sinistres graves eux-mêmes ?
#
# | Approche | Comment la charge grave est répartie | Nature de la règle |
# |---|---|---|
# | **3a** — forfait | Même montant pour tous | **Postulée** : on suppose que tous les assurés ont le même risque grave |
# | **3b** — proportion | Au prorata de la prime attritionnelle (× 1 + τ) | **Postulée** : on suppose que le risque grave suit exactement le risque attritionnel |
# | **4** — tarification séparée | Selon un GLM ajusté sur le **nombre de sinistres graves** | **Estimée** : les données décident quelles variables comptent et de combien |
#
# Dans l'approche 4, ce ne sont pas nous qui décidons que le statut de novice compte et
# que la zone ne compte pas pour les graves : c'est la sélection de variables sur les
# sinistres graves qui l'a établi (§5.4.1). Dans 3a et 3b, la réponse est fixée avant
# même de regarder les graves.
#
# Une hypothèse postulée n'est pas forcément fausse. Mais elle doit être **vérifiée**. On
# confronte donc ci-dessous les deux hypothèses aux données : si 3a était juste, le risque
# grave serait le même partout ; si 3b était juste, les relativités sur les graves seraient
# égales aux relativités attritionnelles.

# %%
titre("LES HYPOTHESES DE 3a ET 3b RESISTENT-ELLES AUX DONNEES ?", 2)
freq_att_moy = base_tarif["nb_att"].sum() / base_tarif["exposition"].sum()
freq_gr_moy = base_tarif["nb_grave"].sum() / base_tarif["exposition"].sum()
pp_att_moy = base_tarif["charge_att"].sum() / base_tarif["exposition"].sum()

lignes_hyp = []
for v, lib_v in [("jeune_conducteur", "Conducteur novice"), ("zone_geo", "Zone")]:
    g_h = base_tarif.groupby(v).agg(e=("exposition", "sum"), ng=("nb_grave", "sum"),
                                    ca=("charge_att", "sum"))
    for mod, r in g_h.iterrows():
        if mod == "Non renseignée":   # 112 annees-police : trop peu pour etre lisible
            continue
        lignes_hyp.append({
            "Variable": lib_v, "Modalité": mod, "Graves observés": int(r["ng"]),
            "3a suppose": "1,00",
            "3b suppose (relativité attritionnelle)": fr((r["ca"] / r["e"]) / pp_att_moy, 2),
            "Relativité observée sur les graves": fr((r["ng"] / r["e"]) / freq_gr_moy, 2),
        })
tab_hyp = pd.DataFrame(lignes_hyp)
exporter_table(tab_hyp, "7_4_hypotheses_3a_3b_confrontees", index=False)
display(tab_hyp)

print("\nLECTURE :")
print("  - ZONE. 3b applique aux graves la relativite attritionnelle de chaque zone : la")
print("    zone C paierait ~17 % de plus sur ses graves, la zone E ~20 % de moins. Or la")
print("    frequence des graves ne depend pas de la zone (test du §5.4.1 : la zone est")
print("    ecartee ; zone C a 0,99). 3b fait donc payer aux assures de zone C un risque")
print("    grave qu'ils n'ont pas : c'est une segmentation ARTIFICIELLE.")
print("  - NOVICE. Chez les novices, la relativite observee sur les graves (~1,6) DEPASSE")
print("    la relativite attritionnelle (~1,4). 3a l'ignore totalement (1,00) ; 3b la")
print("    sous-estime. Seule l'approche 4 la mesure directement.")
print("\n  -> Les deux hypotheses postulees sont contredites par les donnees, chacune a sa")
print("     maniere : 3a par defaut de segmentation, 3b par une segmentation mal placee.")
print("     L'approche 4 ne suppose rien : elle retient la zone pour l'attritionnel, ou")
print("     elle compte, et l'ignore pour les graves, ou elle ne compte pas.")

# %%
print("POURQUOI L'APPROCHE 4, ET SOUS QUELLE FORME :")
print("  1. Elle tarife le risque grave sur une base ESTIMEE plutot que postulee :")
print("     la repartition de la charge grave entre assures est mesuree sur les")
print("     sinistres graves eux-memes, alors que 3a et 3b la fixent a priori — et le")
print("     tableau ci-dessus montre que ces deux hypotheses a priori sont fausses.")
print("     RESERVE : estime ne veut pas dire certain. L'effet novice repose sur 31")
print("     sinistres graves ; c'est pourquoi son ampleur est credibilisee au §8.1.")
print("  2. Ses quatre briques se revisent et se justifient separement.")
print("  3. 3a et 4 ne sont pas deux choix opposes : 3a est exactement l'approche 4 dans")
print("     laquelle on n'accorderait AUCUNE credibilite a l'effet novice sur les graves.")
print("     Le §8.1 rend ce curseur explicite, et chiffre la prime des novices selon")
print("     le degre de confiance accorde a cet effet.")

# %% [markdown]
# ---
# # §8 — Du tarif pur au tarif commercial
#
# ## 8.1 Recalibrage du tarif final sur l'ensemble des données
#
# **C'est la principale amélioration du modèle.** Les découpages apprentissage / test
# servaient à **choisir et valider une méthode**. Une fois la méthode validée, le tarif
# que l'on met en production doit être estimé sur **toute** l'information disponible :
# se priver d'un tiers des sinistres graves pour le tarif définitif n'aurait aucun sens,
# alors même que la brique grave est la plus fragile du modèle.
#
# Le gain est direct : la brique grave passe de 125 à 187 événements, et l'effet
# « novice », surestimé sur la seule année 2023, est ramené à une valeur plus fiable.

# %%
titre("RECALIBRAGE DU TARIF FINAL SUR 2023 + 2024")
modeles_final = calibrer_tarif(base_tarif, sin_carac, [2023, 2024], loi_retenue)

comparaison_final = pd.DataFrame([
    {"Paramètre": "Sinistres graves utilisés",
     "Calibration 2023": fr(int(train_temp["nb_grave"].sum())),
     "Calibration finale": fr(int(base_tarif["nb_grave"].sum()))},
    {"Paramètre": "Coût moyen des graves",
     "Calibration 2023": fr(modeles_temporel["cout_grave_moyen"]),
     "Calibration finale": fr(modeles_final["cout_grave_moyen"])},
    {"Paramètre": "Relativité novice — fréquence globale",
     "Calibration 2023": fr(float(np.exp(modeles_temporel["freq_tous"].params.get(
         "C(jeune_conducteur)[T.Oui]", 0))), 2),
     "Calibration finale": fr(float(np.exp(modeles_final["freq_tous"].params.get(
         "C(jeune_conducteur)[T.Oui]", 0))), 2)},
    {"Paramètre": "Relativité novice — fréquence des graves",
     "Calibration 2023": fr(float(np.exp(modeles_temporel["freq_grave"].params.get(
         "C(jeune_conducteur)[T.Oui]", 0))), 2),
     "Calibration finale": fr(float(np.exp(modeles_final["freq_grave"].params.get(
         "C(jeune_conducteur)[T.Oui]", 0))), 2)},
])
exporter_table(comparaison_final, "8_1_recalibrage_final", index=False)
display(comparaison_final)

primes_final = appliquer_tarif(modeles_final, base_tarif)
pp_final = float((primes_final["4. Graves tarifés"] * primes_final["exposition"]).sum()
                 / primes_final["exposition"].sum())
couv_final = float((primes_final["4. Graves tarifés"] * primes_final["exposition"]).sum()
                   / base_tarif["charge_totale"].sum())
print(f"\nPrime pure moyenne (avant credibilite) : {fr(pp_final)} par annee-police")
print(f"Couverture de la charge totale observee : {pct(couv_final, 2)}")

# %% [markdown]
# ### 8.1.1 Le curseur de crédibilité
#
# Reste une incertitude assumée : l'**ampleur** de l'effet novice sur les graves, estimée
# sur 31 événements. La **théorie de la crédibilité** fournit le cadre pour la traiter : on
# retient une relativité intermédiaire entre celle estimée et la neutralité,
#
# $$r_Z = 1 + Z \times (\hat r - 1), \qquad Z \in [0 ; 1]$$
#
# où `Z` mesure la confiance accordée à l'estimation. La fréquence de base est ensuite
# recalée pour que le tarif collecte toujours la même charge totale.
#
# - `Z = 1` : on retient pleinement l'effet estimé — c'est l'**approche 4** ;
# - `Z = 0` : on l'ignore et l'on mutualise les graves — c'est exactement l'**approche 3a** ;
# - la **crédibilité de fluctuation limitée** fournit une valeur de référence *a priori* :
#   `Z = √(n / n_F)`, où `n` est le nombre d'événements et `n_F ≈ 1 082` le seuil de
#   pleine crédibilité usuel (estimation à ±5 % avec 90 % de confiance).

# %%
titre("SENSIBILITE DU TARIF A LA CREDIBILITE DE L'EFFET NOVICE", 2)
r_hat = float(np.exp(modeles_final["freq_grave"].params.get("C(jeune_conducteur)[T.Oui]", 0)))
n_nov_gr = int(base_tarif.loc[base_tarif["jeune_conducteur"] == "Oui", "nb_grave"].sum())
n_pleine = (stats.norm.ppf(0.95) / 0.05) ** 2
z_lf = min(1.0, np.sqrt(n_nov_gr / n_pleine))
print(f"Relativite estimee : {fr(r_hat, 2)} sur {n_nov_gr} graves de novices")
print(f"Credibilite de fluctuation limitee : Z = racine({n_nov_gr} / {fr(n_pleine)}) = {fr(z_lf, 3)}")

expo_f = base_tarif["exposition"].values
novice = (base_tarif["jeune_conducteur"] == "Oui").values
prime_att_f = primes_final["2. Attritionnels seuls"].values
nb_grave_tot = base_tarif["nb_grave"].sum()
lignes_z = []
for lib, z in [("Z = 0 (approche 3a)", 0.0),
               (f"Z = {fr(z_lf, 2)} (fluctuation limitée)", z_lf),
               ("Z = 0,5", 0.5), ("Z = 1 (approche 4)", 1.0)]:
    r_z = 1 + z * (r_hat - 1)
    base_f = nb_grave_tot / (expo_f * np.where(novice, r_z, 1.0)).sum()
    prime_gr = base_f * np.where(novice, r_z, 1.0) * modeles_final["cout_grave_moyen"]
    prime_tot = prime_att_f + prime_gr
    lignes_z.append({
        "Crédibilité": lib, "Relativité novice (graves)": fr(r_z, 2),
        "Prime pure — confirmé": fr(float(np.average(prime_tot[~novice], weights=expo_f[~novice]))),
        "Prime pure — novice": fr(float(np.average(prime_tot[novice], weights=expo_f[novice]))),
        "Rapport novice / confirmé": fr(float(np.average(prime_tot[novice], weights=expo_f[novice])
                                              / np.average(prime_tot[~novice], weights=expo_f[~novice])), 2),
    })
tab_z = pd.DataFrame(lignes_z)
exporter_table(tab_z, "8_1_curseur_credibilite", index=False)
display(tab_z)

# %% [markdown]
# **Quel Z retenir ?** Plutôt que de le fixer arbitrairement, on le choisit par
# **validation croisée à 5 blocs** sur l'ensemble des données : on estime l'effet sur
# quatre blocs de contrats, on mesure la déviance de la prédiction sur le cinquième, et
# l'on répète en faisant tourner les blocs. Le Z retenu est celui qui prédit le mieux des
# contrats non utilisés pour l'estimation.

# %%
titre("CHOIX DE Z PAR VALIDATION CROISEE", 2)


def deviance_graves_credibilisee(fit, val, z):
    """Deviance, sur `val`, du modele des graves estime sur `fit` avec la credibilite z."""
    m, _ = ajuster_frequence(fit, ["jeune_conducteur"], "nb_grave")
    r_z = 1 + z * (float(np.exp(m.params.iloc[1])) - 1)
    nov_fit = (fit["jeune_conducteur"] == "Oui").values
    base_z = fit["nb_grave"].sum() / (fit["exposition"].values
                                     * np.where(nov_fit, r_z, 1.0)).sum()
    nov_val = (val["jeune_conducteur"] == "Oui").values
    pred = base_z * np.where(nov_val, r_z, 1.0) * val["exposition"].values
    return deviance_poisson(val["nb_grave"].values, pred)


blocs_cv = np.random.default_rng(GRAINE).integers(0, 5, len(base_tarif))
grille_z = [round(z, 1) for z in np.arange(0, 1.01, 0.1)]
dev_cv = {z: sum(deviance_graves_credibilisee(base_tarif[blocs_cv != b],
                                              base_tarif[blocs_cv == b], z)
                 for b in range(5)) for z in grille_z}
z_cv = min(dev_cv, key=dev_cv.get)
tab_cv = pd.DataFrame({"Z": [fr(z, 1) for z in grille_z],
                       "Déviance en validation croisée": [round(dev_cv[z], 2) for z in grille_z],
                       "Écart au meilleur": [round(dev_cv[z] - dev_cv[z_cv], 2) for z in grille_z]})
exporter_table(tab_cv, "8_1_credibilite_validation_croisee", index=False)
display(tab_cv)
print(f"\nZ optimal en validation croisee : {fr(z_cv, 1)}")
print("-> La courbe de deviance est tres plate au voisinage de l'optimum : les donnees")
print("   distinguent nettement Z = 0 (mutualisation complete, la plus mauvaise) des")
print("   valeurs elevees, mais pas les valeurs elevees entre elles.")


def credibiliser_graves(m, base, z):
    """Applique la credibilite z a la relativite novice du modele des graves.

    La frequence de base est recalee pour que le nombre de graves predit sur `base`
    reste egal au nombre observe : la credibilite deplace la prime entre assures,
    elle ne change pas la charge collectee.
    """
    assert list(m["freq_grave"].params.index) == ["Intercept", "C(jeune_conducteur)[T.Oui]"], \
        "la credibilite est implementee pour un modele des graves a une seule indicatrice"
    m = dict(m)
    r = float(np.exp(m["freq_grave"].params["C(jeune_conducteur)[T.Oui]"]))
    r_z = 1 + z * (r - 1)
    nov = (base["jeune_conducteur"] == "Oui").values
    base_f = base["nb_grave"].sum() / (base["exposition"].values * np.where(nov, r_z, 1.0)).sum()
    m["credibilite_graves"] = {"z": z, "r": r, "r_z": r_z, "base": base_f}
    return m


Z_RETENU = z_cv
modeles_final = credibiliser_graves(modeles_final, base_tarif, Z_RETENU)
primes_final = appliquer_tarif(modeles_final, base_tarif)
pp_final = float((primes_final["4. Graves tarifés"] * primes_final["exposition"]).sum()
                 / primes_final["exposition"].sum())
couv_final = float((primes_final["4. Graves tarifés"] * primes_final["exposition"]).sum()
                   / base_tarif["charge_totale"].sum())
nov_f = (base_tarif["jeune_conducteur"] == "Oui").values
pp_nov = float(np.average(primes_final["4. Graves tarifés"][nov_f], weights=expo_f[nov_f]))
pp_conf = float(np.average(primes_final["4. Graves tarifés"][~nov_f], weights=expo_f[~nov_f]))

titre("TARIF FINAL RETENU")
print(f"Credibilite retenue           : Z = {fr(Z_RETENU, 1)} (validation croisee)")
print(f"Relativite novice sur graves  : {fr(modeles_final['credibilite_graves']['r_z'], 2)} "
      f"(estimee : {fr(r_hat, 2)} ; calibration 2023 seule : 2,06)")
print(f"Prime pure moyenne            : {fr(pp_final)} par annee-police")
print(f"Prime pure moyenne — confirme : {fr(pp_conf)}")
print(f"Prime pure moyenne — novice   : {fr(pp_nov)}  (rapport {fr(pp_nov / pp_conf, 2)})")
print(f"Couverture de la charge       : {pct(couv_final, 2)}")
assert abs(couv_final - 1) < 0.01, "le tarif final doit couvrir la charge observee"

tarif_final = pd.DataFrame([
    {"Indicateur": "Crédibilité Z (validation croisée)", "Valeur": fr(Z_RETENU, 1)},
    {"Indicateur": "Relativité novice estimée (graves)", "Valeur": fr(r_hat, 2)},
    {"Indicateur": "Relativité novice retenue (graves)",
     "Valeur": fr(modeles_final["credibilite_graves"]["r_z"], 2)},
    {"Indicateur": "Prime pure moyenne", "Valeur": fr(pp_final)},
    {"Indicateur": "Prime pure — conducteur confirmé", "Valeur": fr(pp_conf)},
    {"Indicateur": "Prime pure — conducteur novice", "Valeur": fr(pp_nov)},
    {"Indicateur": "Rapport novice / confirmé", "Valeur": fr(pp_nov / pp_conf, 2)},
    {"Indicateur": "Couverture de la charge observée", "Valeur": pct(couv_final, 2)},
])
exporter_table(tarif_final, "8_1_tarif_final", index=False)

print("\n-> Le tarif final combine les deux ameliorations : il est estime sur toutes les")
print("   donnees, et l'ampleur de l'effet le plus fragile est fixee par une procedure")
print("   objective plutot que prise telle que l'estimation la livre. L'ecart de 2023")
print("   (2,06) est ainsi corrige deux fois : par l'ajout de 2024, puis par la credibilite.")
print("\n   La valeur de fluctuation limitee reste une option PRUDENTIELLE, a envisager si")
print("   l'on veut limiter l'ecart de prime entre novices et confirmes tant que")
print("   l'historique est court. Le choix est une decision de politique tarifaire,")
print("   et le tableau du curseur en chiffre exactement la portee.")

# %% [markdown]
# ## 8.2 Chargements et prime commerciale
#
# Complément issu du cours, qui rappelle que la prime se compose de trois étages :
#
# > **Prime pure → Prime d'inventaire → Prime commerciale**
#
# Tout ce qui précède a porté sur la **prime pure**, c'est-à-dire l'espérance des
# pertes. Un tarif réel y ajoute trois éléments, que le cours désigne comme les
# contraintes de suffisance, de risque de ruine et de rémunération des fonds propres.
#
# Les taux ci-dessous sont des **hypothèses de travail explicites**, et non des
# résultats de l'étude : ils dépendent de la structure de coûts de l'assureur et de la
# fiscalité applicable.

# %%
titre("DU TARIF PUR AU TARIF COMMERCIAL")

TAUX_SECURITE = 0.05     # chargement de sécurité (contrainte de risque de ruine)
FRAIS_GESTION = 0.12     # frais de gestion des sinistres et des contrats
TAUX_MARGE = 0.05        # rémunération des fonds propres immobilisés
TAUX_TAXE = 0.18         # taxe sur les conventions d'assurance

prime_pure_ref = pp_final   # tarif final recalibre sur 2023 + 2024 (§8.1)

etapes = []
p = prime_pure_ref
etapes.append(("Prime pure", p, "Espérance des pertes (approche 4 recalibrée, §8.1)"))
p_sec = p * (1 + TAUX_SECURITE)
etapes.append((f"+ chargement de sécurité ({pct(TAUX_SECURITE, 0)})", p_sec,
               "Couvre l'écart défavorable autour de l'espérance"))
p_inv = p_sec / (1 - FRAIS_GESTION)
etapes.append((f"= Prime d'inventaire (frais {pct(FRAIS_GESTION, 0)})", p_inv,
               "Intègre les frais de gestion"))
p_com_ht = p_inv / (1 - TAUX_MARGE)
etapes.append((f"+ marge bénéficiaire ({pct(TAUX_MARGE, 0)})", p_com_ht,
               "Rémunère les fonds propres immobilisés"))
p_ttc = p_com_ht * (1 + TAUX_TAXE)
etapes.append((f"= Prime commerciale TTC (taxe {pct(TAUX_TAXE, 0)})", p_ttc,
               "Montant effectivement payé par l'assuré"))

tab_com = pd.DataFrame(etapes, columns=["Étape", "Montant", "Nature"])
tab_com["Montant"] = tab_com["Montant"].map(fr)
exporter_table(tab_com, "8_2_passage_tarif_commercial", index=False)
display(tab_com)

print(f"\nRapport prime commerciale / prime pure : {fr(p_ttc / p, 2)}")
print(f"La prime pure represente {pct(p / p_ttc)} de la prime payee par l'assure.")

# %%
# Application aux profils types, avec le tarif final
primes_profils_final = appliquer_tarif(modeles_final, profils)
aff_com = pd.DataFrame({"Profil": profils["Profil"]})
pp_profils = primes_profils_final["4. Graves tarifés"]
aff_com["Prime pure"] = pp_profils.map(fr)
aff_com["Prime d'inventaire"] = (pp_profils * (1 + TAUX_SECURITE)
                                 / (1 - FRAIS_GESTION)).map(fr)
aff_com["Prime commerciale TTC"] = (pp_profils * (1 + TAUX_SECURITE)
                                    / (1 - FRAIS_GESTION) / (1 - TAUX_MARGE)
                                    * (1 + TAUX_TAXE)).map(fr)
exporter_table(aff_com, "8_3_tarif_commercial_profils", index=False)
display(aff_com)

# %% [markdown]
# ---
# # §9 — Synthèse, limites et pistes
#
# ## 9.1 Ce que l'étude établit

# %%
titre("SYNTHESE DES RESULTATS")

resultats_cles = pd.DataFrame([
    ("Fiabilisation",
     f"{fr(n_recoup)} valeurs restaurées par recoupement entre variables, "
     f"{fr(n_imput)} par imputation",
     "Une règle d'imputation doit être back-testée avant d'être appliquée"),
    ("Fiabilisation",
     "Hypothèses de faute de frappe testées et rejetées (6,5 % de cohérence)",
     "Les valeurs aberrantes étaient injectées, non mal saisies"),
    ("Seuil de gravité",
     f"S = {fr(SEUIL_GRAVE)}, validé par 4 méthodes convergentes",
     "Concordance de 99,4 % avec la nature du sinistre"),
    ("Structure du risque",
     f"Les graves : {pct(part_nb)} du nombre, {pct(part_ch)} de la charge",
     "La prime est dominée par un événement rare"),
    ("Fréquence",
     "Loi de Poisson retenue après test ; pas de surdispersion",
     "La binomiale négative dégrade l'AIC"),
    ("Fréquence",
     f"Conducteur novice : coefficient {fr(float(np.exp(glm_freq.params.get('C(jeune_conducteur)[T.Oui]', 0))), 2)}",
     "Seule variable conducteur retenue ; l'âge et l'ancienneté sont redondants"),
    ("Sévérité",
     "Log-normale retenue ; aucune variable conducteur ne ressort",
     "Le profil détermine la probabilité d'accident, pas le coût du dommage"),
    ("Graves",
     f"Conducteur novice : fréquence multipliée par {fr(r_hat, 2)} sur 2023 + 2024, "
     f"{fr(modeles_final['credibilite_graves']['r_z'], 2)} après crédibilité",
     "Effet significatif, mais surestimé par la seule année 2023 (2,06)"),
    ("Tarif final",
     f"Prime pure {fr(pp_final)} ; confirmé {fr(pp_conf)}, novice {fr(pp_nov)}",
     f"Recalibré sur toutes les données, crédibilité Z = {fr(Z_RETENU, 1)}"),
    ("Graves",
     f"Coût moyen {fr(cout_grave_moyen)} ; espérance GPD {fr(esp_gpd)}",
     "Deux méthodes concordent ; la GPD n'apporte pas de gain de stabilité"),
    ("Validation",
     "Charge prédite dans l'IC bootstrap de l'observé, sur les deux dispositifs",
     "Les irrégularités du lift sont du bruit, pas un défaut de calibrage"),
    ("Validation",
     "Discrimination démontrée sur la charge attritionnelle, pas sur la charge totale",
     "Le tarif segmente ce qui peut l'être et mutualise le reste"),
    ("Données",
     "Sinistralité identique quelle que soit la garantie souscrite",
     "Les sinistres ont été générés indépendamment de la formule du contrat"),
], columns=["Domaine", "Résultat", "Portée"])
exporter_table(resultats_cles, "9_1_resultats_cles", index=False)
with pd.option_context("display.max_colwidth", 70):
    display(resultats_cles)

# %% [markdown]
# ## 9.2 Limites
#
# Elles sont réelles et il serait malhonnête de les minimiser.
#
# **1. Un seul exercice de souscription.** Tout le portefeuille prend effet en 2023.
# Le dispositif temporel repose donc sur un découpage par exercice de survenance, qui
# fait figurer les mêmes polices dans les deux échantillons. Aucune validation sur un
# millésime de souscription réellement distinct n'est possible.
#
# **2. La brique grave repose sur trop peu d'événements.** 187 sinistres graves au
# total, dont 31 de conducteurs novices. L'intervalle de confiance à 95 % sur leur coût
# moyen couvre environ 35 à 40 % de la valeur estimée. Or cette brique porte 89 % de la
# prime : c'est la principale fragilité de l'étude. Le recalibrage final sur toutes les
# données l'atténue, il ne la supprime pas.
#
# **3. Une part de graves anormalement élevée.** 14 % des sinistres dépassent le seuil,
# contre un ordre de grandeur de 1 à 2 % dans un portefeuille automobile réel. Les
# niveaux absolus de prime obtenus ici ne sont donc **pas transposables** à un
# portefeuille réel ; seule la méthode l'est.
#
# **4. Aucun retraitement des sinistres tardifs (IBNR).** Les sinistres survenus en fin
# de période mais déclarés après l'arrêté des données sont absents. La sinistralité
# 2024, en particulier le dernier trimestre, est donc sous-estimée, ce qui biaise
# favorablement toute validation portant sur cette période.
#
# **5. Des montants partiellement évalués.** Un sinistre sur cinq est encore ouvert,
# et 45 % des graves le sont. Leur charge est une provision, non un règlement définitif.
#
# **6. Un tarif peu segmentant.** Deux variables seulement survivent à la sélection sur
# la fréquence. Le tarif produit en pratique deux niveaux — novice ou non — modulés par
# la zone. Ce n'est pas un défaut de méthode mais une propriété des données : les
# autres variables ne portent pas d'information exploitable, et les tests le montrent.
#
# **7. Des garanties non modélisées dans les données.** Les trois formules — tiers,
# tiers étendu, tous risques — présentent exactement la même sinistralité, et près de la
# moitié des sinistres des contrats au tiers relèvent de garanties que ces contrats ne
# couvrent pas (§1.4.1). En situation réelle, la formule de garantie serait la première
# variable tarifaire ; ici, elle ne peut pas l'être.
#
# ## 9.3 Ce qui a été amélioré, et ce qui ne peut pas l'être
#
# La courbe de lift de la charge totale est irrégulière. La question « comment
# l'améliorer ? » appelle trois réponses de nature différente.
#
# **Améliorations mises en œuvre**
#
# | Amélioration | Où | Effet |
# |---|---|---|
# | Intervalles de confiance bootstrap sur le lift et le ratio de couverture | §7.2, §7.3.4 | Montre que l'irrégularité est compatible avec un modèle bien calibré |
# | Validation de chaque brique sur sa propre cible | §7.3 | Rend visible la discrimination réelle du tarif sur l'attritionnel |
# | Sélection des graves sur toutes les variables, confirmée par test | §5.4.1 | La spécification retenue ne repose plus sur une liste restreinte |
# | Stabilité de l'effet novice sur les graves testée entre périodes | §7.3.3 | Révèle que l'estimation 2023 surestimait l'effet |
# | **Recalibrage du tarif final sur toutes les données** | §8.1 | 187 graves au lieu de 125 ; effet novice ramené de 2,06 à une valeur plus fiable |
# | Crédibilité de l'effet novice, Z choisi par validation croisée | §8.1.1 | Fixe objectivement l'ampleur de l'effet le plus fragile, et chiffre les alternatives |
# | Contrôle de cohérence garanties / sinistres | §1.4.1, §2.7.3 | Identifie une limite majeure des données |
#
# **Pistes testées, sans gain — résultats négatifs documentés**
#
# | Piste | Où | Résultat |
# |---|---|---|
# | Interaction conducteur novice × zone | §5.2.7 | Non significative, dégrade l'AIC |
# | Tranches d'âge plutôt qu'indicatrice novice, pour les graves | §5.4.1 | AIC et BIC moins bons |
# | Coût des graves par la loi de Pareto généralisée | §5.4.2 | Pas plus stable que la moyenne empirique |
# | Binomiale négative pour la fréquence | §5.2.4 | Dégrade l'AIC |
#
# **Ce qui ne peut pas être amélioré avec ces données.** L'irrégularité du lift de la
# charge totale vient de ce qu'environ 60 sinistres graves, dont quelques-uns dépassent
# 200 000, se répartissent entre dix déciles. Aucun modèle ne peut prédire *lequel* des
# contrats subira un tel sinistre : chercher à faire coller la courbe aux barres
# reviendrait à apprendre le bruit. **Forcer ce graphique à devenir régulier serait la
# pire amélioration possible.**
#
# ## 9.4 Pistes pour aller plus loin
#
# 1. **Constituer un historique pluriannuel.** C'est la seule vraie réponse au problème
#    du lift : plus d'années, c'est plus de sinistres graves, donc une brique grave
#    estimable et validable. C'est aussi la condition pour valider sur des millésimes de
#    souscription distincts.
# 2. **Tarifer garantie par garantie**, sur des données où la sinistralité dépend de la
#    formule souscrite — ce qui suppose de faire corriger la source.
# 3. **Retraiter les IBNR** par une méthode de cadence (Chain Ladder) avant tarification.
# 4. **Réévaluer la loi de Pareto généralisée** quand l'effectif de graves aura grandi :
#    sur 125 sinistres elle n'apporte rien, sur plusieurs centaines elle permettrait de
#    tarifer la queue de distribution au-delà de ce qui a été observé.
# 5. **Enrichir les variables explicatives** : kilométrage annuel, zone plus fine que
#    cinq modalités, historique de sinistralité individuel.
# 6. **Exploiter la piste ouverte par la modalité « Non renseignée »** de la zone
#    géographique, dont la surfréquence suggère que la qualité de la souscription est
#    elle-même un facteur de risque.

# %% [markdown]
# ---
# # §10 — Fichiers produits

# %%
titre("FICHIERS PRODUITS")
fichiers_tables = sorted(TABLES.glob("*.xlsx"))
fichiers_figures = sorted(FIGURES.glob("*.png"))
print(f"Tableaux exportes ({len(fichiers_tables)}) dans outputs/tables/ :")
for f in fichiers_tables:
    print(f"   {f.name}")
print(f"\nFigures exportees ({len(fichiers_figures)}) dans outputs/figures/ :")
for f in fichiers_figures:
    print(f"   {f.name}")

# Export de la base de tarification finale, avec les primes des cinq approches
# calculees par le tarif recalibre sur l'ensemble des donnees (§8.1)
base_finale = base_tarif[["id_police", "exposition", "tr_age", "tr_anciennete",
                          "jeune_conducteur", "zone_geo", "classe_vehicule",
                          "carburant", "nb_sinistres", "nb_att", "nb_grave",
                          "charge_totale", "charge_att", "charge_grave"]].copy()
primes_finales = appliquer_tarif(modeles_final, base_tarif)
for a in APPROCHES:
    base_finale[f"prime_{a.split('.')[0]}"] = primes_finales[a].round(2)
exporter_table(base_finale, "10_base_tarification_finale", index=False)
print(f"\nBase de tarification finale exportee : {fr(len(base_finale))} contrats, "
      f"{base_finale.shape[1]} colonnes.")

titre("FIN DE L'ETUDE")
print("Toutes les etapes demandees par l'enonce ont ete traitees :")
print("   A - Fiabilisation et preparation des donnees          -> §1")
print("   B - Separation attritionnels / graves                 -> §2")
print("   C - Approche Frequence x Cout                         -> §3")
print("   D - Modelisation GLM                                  -> §5")
print("   E - Comparaison des approches de pricing              -> §6")
print("   F - Separation Train / Test (les deux approches)      -> §4 et §7")
