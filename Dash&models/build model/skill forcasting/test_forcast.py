import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# 🔹 Chargement des modèles et prévisions
@st.cache_resource
def load_models():
    with open("prophet_models.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_forecasts():
    df = pd.read_csv("forecast_all_skills.csv")
    df["ds"] = pd.to_datetime(df["ds"])
    return df

models = load_models()
df_forecasts = load_forecasts()

# 🔹 Interface Streamlit
st.title("📊 Prévision de la demande de compétences")

# Liste des compétences disponibles
all_skills = sorted(df_forecasts["Skill"].unique())

# 🔍 Sélection dynamique
selected_skill = st.selectbox("Choisissez une compétence :", all_skills, placeholder="ex: python")

if selected_skill:
    # 📆 Définir les bornes temporelles
    today = pd.to_datetime(datetime.today().date())
    start_date = today - pd.DateOffset(months=6)
    end_date = today + pd.DateOffset(months=12)

    # 🔍 Filtrer la compétence sélectionnée
    df_skill = df_forecasts[df_forecasts["Skill"] == selected_skill]
    df_skill = df_skill[(df_skill["ds"] >= start_date) & (df_skill["ds"] <= end_date)]

    # ➗ Séparer historique et prévision
    df_hist = df_skill[df_skill["ds"] <= today]
    df_pred = df_skill[df_skill["ds"] > today]

    # 📈 Tracer le graphe
    fig, ax = plt.subplots(figsize=(14, 6))

    # Historique : ligne bleue
    ax.plot(df_hist["ds"], df_hist["yhat"], label="Historique (yhat)", color="blue")

    # Prévision : ligne orange pointillée (sans intervalle)
    ax.plot(df_pred["ds"], df_pred["yhat"], label="Prévision (yhat)", color="orange", linestyle="--")

    # Ligne verticale pour aujourd’hui
    ax.axvline(today, color='red', linestyle=':', label="Aujourd’hui")

    # Axe des dates
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Titres et style
    ax.set_title(f"Prévision de la demande pour : {selected_skill}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Nombre d'occurrences")
    ax.legend()
    ax.grid(True)

    # Affichage dans Streamlit
    st.pyplot(fig)
