import streamlit as st
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pickle
import warnings

# ======================
# ✅ SETUP
# ======================
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Prediction Tools")

# Sidebar
st.sidebar.header("🔧 Select One Prediction Tool")
selected_tool = st.sidebar.radio(
    "Choose the type of prediction you want to perform:",
    ["📈 Skill Forecast", "🧠 Skill Recommendation", "💰 Salary Estimation"]
)

st.title("🧮 Prediction Center")

# ======================
# 📁 PATHS
# ======================
RECOMMENDER_MODEL_PATH = "D:\cycle_ing\\2eme anne bdia\S4\web scrapping\\final\projectfinal\Dash&models\\build model\skill recomendation\dl model\skill_recommender.h5"
RECOMMENDER_MLB_PATH = "D:\cycle_ing\\2eme anne bdia\S4\web scrapping\\final\projectfinal\Dash&models\\build model\skill recomendation\dl model\skill_label_binarizer (1).pkl"

FORECAST_MODEL_PATH = "D:\cycle_ing\\2eme anne bdia\S4\web scrapping\\final\projectfinal\Dash&models\\build model\skill forcasting\prophet_models.pkl"
FORECAST_CSV_PATH = "D:\cycle_ing\\2eme anne bdia\S4\web scrapping\\final\projectfinal\Dash&models\\build model\skill forcasting\\forecast_all_skills.csv"

SALARY_MODEL_PATH = "D:\cycle_ing\\2eme anne bdia\S4\web scrapping\\final\projectfinal\Dash&models\\build model\salary estimation\\final_deep_learning_model.h5"
SALARY_SCALER_PATH = "D:\cycle_ing\\2eme anne bdia\S4\web scrapping\\final\projectfinal\Dash&models\\build model\salary estimation\\feature_scaler (1).pkl"

# ======================
# 📦 LOAD MODELS
# ======================
@st.cache_resource
def load_recommender():
    return load_model(RECOMMENDER_MODEL_PATH), joblib.load(RECOMMENDER_MLB_PATH)

@st.cache_resource
def load_forecast_models():
    with open(FORECAST_MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_forecast_data():
    df = pd.read_csv(FORECAST_CSV_PATH)
    df["ds"] = pd.to_datetime(df["ds"])
    return df

@st.cache_resource
def load_salary_model():
    model = tf.keras.models.load_model(SALARY_MODEL_PATH)
    scaler = joblib.load(SALARY_SCALER_PATH)
    return model, scaler

# Load once
forecast_models = load_forecast_models()
df_forecasts = load_forecast_data()
recommender_model, mlb = load_recommender()
salary_model, scaler = load_salary_model()

# ======================
# 📈 SKILL FORECAST
# ======================
if selected_tool == "📈 Skill Forecast":
    st.header("📈 Skill Demand Forecast")
    selected_skill = st.selectbox("🔍 Select a skill to forecast:", sorted(df_forecasts["Skill"].unique()))

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


# ======================
# 🧠 SKILL RECOMMENDATION
# ======================
elif selected_tool == "🧠 Skill Recommendation":
    st.header("🧠 Skill Recommendation")
    all_skills = sorted(mlb.classes_)

    def recommend_skills(input_skills, top_k=5):
        input_vec = mlb.transform([input_skills])
        preds = recommender_model.predict(input_vec)[0]
        preds[input_vec[0] == 1] = 0
        top_indices = preds.argsort()[-top_k:][::-1]
        return mlb.classes_[top_indices].tolist()

    selected_skills = st.multiselect("✅ Select your known skills", options=all_skills)
    top_k = st.slider("🔝 Number of skills to recommend", 3, 20, 5)

    if st.button("🔍 Recommend", key="reco"):
        if not selected_skills:
            st.warning("Please select at least one skill.")
        else:
            suggestions = recommend_skills(selected_skills, top_k)
            st.success("💡 Recommended skills:")
            for skill in suggestions:
                st.markdown(f"- {skill}")

# ======================
# 💰 SALARY ESTIMATION
# ======================
elif selected_tool == "💰 Salary Estimation":
    st.header("💰 Salary Estimation")
    scaler_features = scaler.feature_names_in_
    job_titles = sorted([f.replace("jobtitle_", "") for f in scaler_features if f.startswith("jobtitle_")])
    skills_list = sorted([f.replace("skill_", "") for f in scaler_features if f.startswith("skill_")])

    selected_job = st.selectbox("🧠 Select job title", job_titles)
    selected_salary_skills = st.multiselect("🛠️ Select skills", options=skills_list)

    if st.button("🔍 Estimate Salary", key="salary"):
        if not selected_salary_skills:
            st.warning("Please select at least one skill.")
        else:
            X_input = pd.DataFrame(0, index=[0], columns=scaler_features)
            job_col = f"jobtitle_{selected_job}"
            if job_col in X_input.columns:
                X_input[job_col] = 1
            for skill in selected_salary_skills:
                skill_col = f"skill_{skill}"
                if skill_col in X_input.columns:
                    X_input[skill_col] = 1
            X_scaled = scaler.transform(X_input)
            predicted_salary = salary_model.predict(X_scaled)[0][0]
            st.success(f"💵 Estimated Salary: **{predicted_salary:,.2f} €**")
