# -*- coding: utf-8 -*-
"""Convertit pricing_auto.py en notebook Jupyter, puis l'execute.

Le livrable principal est ecrit sous forme de script Python decoupe en cellules par
des marqueurs `# %%` (cellule de code) et `# %% [markdown]` (cellule de texte). Ce
format presente deux avantages : le script reste executable tel quel — c'est le
« script Python complet » demande par l'enonce — et il se convertit en notebook sans
perte, ce qui evite de maintenir deux versions du meme travail.

Usage :
    python construire_notebook.py            # convertit et execute
    python construire_notebook.py --sans-execution
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

DOSSIER = Path(__file__).parent
SOURCE = DOSSIER / "pricing_auto.py"
SORTIE = DOSSIER / "pricing_auto.ipynb"


def decouper_en_cellules(texte):
    """Decoupe le script en cellules d'apres les marqueurs `# %%`."""
    cellules = []
    courante, type_courant = [], "code"

    for ligne in texte.split("\n"):
        depouillee = ligne.strip()
        if depouillee.startswith("# %%"):
            if any(l.strip() for l in courante):
                cellules.append((type_courant, courante))
            courante = []
            type_courant = "markdown" if "[markdown]" in depouillee else "code"
        else:
            courante.append(ligne)

    if any(l.strip() for l in courante):
        cellules.append((type_courant, courante))
    return cellules


def nettoyer_markdown(lignes):
    """Retire le prefixe de commentaire des lignes d'une cellule markdown."""
    sorties = []
    for l in lignes:
        if l.startswith("# "):
            sorties.append(l[2:])
        elif l.strip() == "#":
            sorties.append("")
        else:
            sorties.append(l)
    return "\n".join(sorties).strip("\n")


def construire():
    texte = SOURCE.read_text(encoding="utf-8")
    notebook = new_notebook()

    for type_cellule, lignes in decouper_en_cellules(texte):
        if type_cellule == "markdown":
            contenu = nettoyer_markdown(lignes)
            if contenu:
                notebook.cells.append(new_markdown_cell(contenu))
        else:
            contenu = "\n".join(lignes).strip("\n")
            # On ne conserve pas la ligne d'encodage, inutile dans un notebook
            contenu = "\n".join(l for l in contenu.split("\n")
                                if not l.startswith("# -*- coding"))
            if contenu.strip():
                notebook.cells.append(new_code_cell(contenu))

    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
        "title": "Tarification automobile — de la fiabilisation au tarif technique",
    }
    return notebook


def executer(notebook):
    """Execute toutes les cellules et incorpore les sorties dans le notebook."""
    from nbclient import NotebookClient
    client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(DOSSIER)}})
    client.execute()
    return notebook


if __name__ == "__main__":
    nb = construire()
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    print(f"Notebook construit : {n_code} cellules de code, {n_md} cellules de texte.")

    if "--sans-execution" not in sys.argv:
        print("Execution en cours (quelques minutes)...")
        nb = executer(nb)
        print("Execution terminee.")

    nbformat.write(nb, SORTIE)
    print(f"Ecrit : {SORTIE}")
