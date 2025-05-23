# script.py

import pandas as pd

#  Charger les prévisions depuis le fichier CSV
def load_forecast_data(csv_path="forecast_all_skills.csv"):
    """
    Charge le fichier CSV contenant les prévisions Prophet.
    Retourne un DataFrame Pandas.
    """
    df = pd.read_csv(csv_path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

# Récupérer les prévisions pour une compétence donnée
def get_forecast_for_skill(df, skill_name):
    """
    Retourne uniquement les lignes correspondant à une compétence donnée.
    """
    df_skill = df[df["Skill"].str.lower() == skill_name.lower()].copy()
    return df_skill.sort_values("ds")

# Lister toutes les compétences disponibles
def get_available_skills(df):
    """
    Retourne la liste triée des compétences dans le DataFrame.
    """
    return sorted(df["Skill"].unique())
