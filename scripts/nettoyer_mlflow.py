"""Archive les anciens runs MLflow pour ne garder que la série la plus récente.

Pourquoi : chaque réexécution du notebook ajoute un run par modèle. Après plusieurs
passages, l'interface MLflow devient illisible et le comptage des runs ne veut plus
rien dire. Ce script archive les runs antérieurs à une date donnée.

La suppression MLflow est DOUCE : les runs passent en lifecycle_stage="deleted",
disparaissent de l'affichage par défaut, mais restent dans la base. L'option
--restaurer les fait revenir.

Usage :
    uv run python scripts/nettoyer_mlflow.py                    # simulation (ne touche à rien)
    uv run python scripts/nettoyer_mlflow.py --appliquer        # archive pour de bon
    uv run python scripts/nettoyer_mlflow.py --restaurer        # annule l'archivage
"""

import argparse
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.entities import ViewType

EXPERIENCE = "scoring_credit"
DATE_PIVOT_PAR_DEFAUT = "2026-09-16"

# La base est à la racine du projet, le script dans scripts/ : on remonte d'un cran.
# On construit un chemin absolu pour que le script marche depuis n'importe quel dossier.
RACINE = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{RACINE / 'mlflow.db'}")


def recuperer_runs(client, experience_id, archives=False):
    """Renvoie les runs de l'expérience, triés du plus ancien au plus récent.

    archives=False -> les runs visibles ; archives=True -> les runs déjà archivés.
    """
    vue = ViewType.DELETED_ONLY if archives else ViewType.ACTIVE_ONLY
    runs = client.search_runs(
        experiment_ids=[experience_id],
        run_view_type=vue,
        max_results=5000,
    )
    return sorted(runs, key=lambda r: r.info.start_time)


def nom_du_run(run):
    """Le nom lisible du run, avec repli sur le tag brut si besoin."""
    return run.info.run_name or run.data.tags.get("mlflow.runName", "(sans nom)")


def date_du_run(run):
    """Convertit l'horodatage MLflow (millisecondes) en datetime local."""
    return datetime.fromtimestamp(run.info.start_time / 1000)


def main():
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument(
        "--avant",
        default=DATE_PIVOT_PAR_DEFAUT,
        help=f"archiver les runs démarrés avant cette date, format AAAA-MM-JJ "
             f"(défaut : {DATE_PIVOT_PAR_DEFAUT})",
    )
    parseur.add_argument("--appliquer", action="store_true",
                         help="exécuter réellement l'archivage")
    parseur.add_argument("--restaurer", action="store_true",
                         help="restaurer tous les runs archivés")
    args = parseur.parse_args()

    client = mlflow.MlflowClient()
    experience = client.get_experiment_by_name(EXPERIENCE)
    if experience is None:
        print(f"Expérience « {EXPERIENCE} » introuvable.")
        print(f"Base consultée : {mlflow.get_tracking_uri()}")
        return

    # --- mode restauration -------------------------------------------------
    if args.restaurer:
        archives = recuperer_runs(client, experience.experiment_id, archives=True)
        if not archives:
            print("Aucun run archivé à restaurer.")
            return
        for run in archives:
            client.restore_run(run.info.run_id)
        print(f"{len(archives)} runs restaurés.")
        return

    # --- tri des runs ------------------------------------------------------
    pivot = datetime.strptime(args.avant, "%Y-%m-%d")
    runs = recuperer_runs(client, experience.experiment_id)

    a_archiver = [r for r in runs if date_du_run(r) < pivot]
    a_garder = [r for r in runs if date_du_run(r) >= pivot]

    print(f"Base       : {mlflow.get_tracking_uri()}")
    print(f"Expérience : {EXPERIENCE}")
    print(f"Date pivot : {args.avant}\n")
    print(f"{len(runs)} runs visibles → {len(a_archiver)} à archiver, {len(a_garder)} conservés\n")

    if a_garder:
        print("CONSERVÉS")
        for run in a_garder:
            print(f"  {date_du_run(run):%Y-%m-%d %H:%M}  {nom_du_run(run)}")
        print()

    if not a_archiver:
        print("Rien à archiver.")
        return

    print("À ARCHIVER")
    for run in a_archiver:
        print(f"  {date_du_run(run):%Y-%m-%d %H:%M}  {nom_du_run(run)}")
    print()

    if not args.appliquer:
        print("Simulation : aucun run n'a été touché.")
        print("Relance avec --appliquer pour archiver réellement.")
        return

    for run in a_archiver:
        client.delete_run(run.info.run_id)
    print(f"{len(a_archiver)} runs archivés.")
    print("Pour annuler : uv run python scripts/nettoyer_mlflow.py --restaurer")


if __name__ == "__main__":
    main()
