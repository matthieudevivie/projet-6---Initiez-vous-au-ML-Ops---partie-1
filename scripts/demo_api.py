"""
Démo de soutenance — interroger le modèle servi par MLflow.

Prérequis : le serveur de serving tourne dans un autre terminal (terminal 2) :

    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    uv run mlflow models serve -m "models:/scoring_credit_lgbm@champion" -p 5001 --env-manager local

Usage, depuis la racine du projet :

    uv run python scripts/demo_api.py

Ce script ne crée AUCUN run MLflow : il se contente de LIRE le seuil dans le registry.
Il ne pollue donc pas l'interface, contrairement à une réexécution du notebook.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Chemins construits à partir de l'emplacement du script : ça marche d'où qu'on le lance
RACINE = Path(__file__).resolve().parent.parent
PARQUET = RACINE / "data" / "df_prepared.parquet"
FICHIER_CLIENTS = RACINE / "data" / "demo_3_clients.json"
URL_API = "http://127.0.0.1:5001/invocations"


def preparer_clients():
    """Extrait, une fois pour toutes, les 3 premiers clients du jeu de test.

    Reprend à l'identique les cellules 2 et 3 de entrainement_modeles.ipynb :
    mêmes filets de sécurité, même découpage (random_state=42), donc mêmes clients.
    Le résultat est mis en cache dans data/demo_3_clients.json.
    """
    from sklearn.model_selection import train_test_split

    print("Préparation unique : extraction des 3 clients de démonstration (~20 s)…")
    df = pd.read_parquet(PARQUET)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    X = X.replace([np.inf, -np.inf], np.nan)                                # filet 1
    obj_cols = X.select_dtypes("object").columns                             # filet 2
    X[obj_cols] = X[obj_cols].apply(pd.to_numeric, errors="coerce")
    X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", str(c)) for c in X.columns]  # filet 3

    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clients = X_test.iloc[:3]

    # Même construction que dans le notebook : les NaN deviennent des null JSON
    data = [[None if pd.isna(v) else v for v in row] for row in clients.values.tolist()]
    contenu = {
        "ids": df.loc[clients.index, "SK_ID_CURR"].tolist(),   # pour l'affichage seulement
        "payload": {"dataframe_split": {"columns": clients.columns.tolist(), "data": data}},
    }
    FICHIER_CLIENTS.write_text(json.dumps(contenu))
    print(f"→ enregistré dans {FICHIER_CLIENTS.relative_to(RACINE)}\n")


def lire_seuil():
    """Lit le seuil métier dans MLflow, sur la version qui porte l'alias champion.

    Le seuil n'est pas codé en dur : il suit la version déployée.
    Lecture seule — aucun run créé, aucune modification de la base.
    """
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{RACINE / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias("scoring_credit_lgbm", "champion")
    seuil = client.get_run(version.run_id).data.metrics["seuil_final"]
    return version.version, seuil


def main():
    if not FICHIER_CLIENTS.exists():
        preparer_clients()
    contenu = json.loads(FICHIER_CLIENTS.read_text())

    version, seuil = lire_seuil()
    print(f"Modèle interrogé : scoring_credit_lgbm@champion (version {version})")
    print(f"Seuil métier lu dans MLflow : {seuil:.2f}\n")

    try:
        reponse = requests.post(URL_API, json=contenu["payload"], timeout=30)
    except requests.ConnectionError:
        raise SystemExit(
            "Le serveur ne répond pas sur le port 5001.\n"
            "→ Lance le terminal 2 et attends la ligne « Uvicorn running on http://127.0.0.1:5001 »."
        )
    if not reponse.ok:
        raise SystemExit(f"L'API a refusé la requête ({reponse.status_code}) :\n{reponse.text[:500]}")

    # L'API renvoie, pour chaque client, [proba de remboursement, proba de défaut]
    probas = [p[1] for p in reponse.json()["predictions"]]

    for id_client, p in zip(contenu["ids"], probas):
        decision = "REFUSÉ" if p >= seuil else "ACCORDÉ"
        print(f"  client {id_client} : probabilité de défaut {p:.3f}  →  {decision}")


if __name__ == "__main__":
    main()
