# 📊 Skill Radar: Trends in Data Science & AI Jobs

**Skill Radar** is an intelligent dashboard and prediction platform that provides insights into the evolving job market in Data Science and AI. It leverages real job data to offer forecasts, skill recommendations, and salary estimations.

---

## 🚀 Features

- **Dashboard Visualizations**

  - Top Job Titles
  - Job Offers by Country and Over Time
  - Top Hiring Companies
  - Most In-Demand Skills
  - Skill Distribution by Job Title

- **Prediction Tools**
  - 📈 Forecast skill demand with Prophet
  - 🧠 Recommend relevant skills using a Deep Learning model
  - 💰 Estimate salaries based on job title and skills with a regression model

---

## 🧱 Project Structure

```
.
├── api/
│ ├── IndeedApiScraping.py
│ ├── LinkedinApiScraping.py
│ └── scheduler.py
│
├── Dash&models/
│ ├── build model/
│ │ ├── salary estimation/
│ │ │ ├── feature_scaler (1).pkl
│ │ │ ├── final_deep_learning_model.h5
│ │ │ └── salary-estimation (1).ipynb
│ │ ├── skill forcasting/
│ │ │ ├── forecast_all_skills.csv
│ │ │ ├── model_skills_forcast.ipynb
│ │ │ ├── prophet_models.pkl
│ │ │ ├── script_skill.py
│ │ │ └── test_forcast.py
│ │ └── skill recomendation/
│ │ ├── dl model/
│ │ │ ├── skill_label_binarizer (1).pkl
│ │ │ ├── skill_recommender.h5
│ │ │ └── SkillRecomendationDL.ipynb
│ │ ├── LR&KNN/
│ │ └── job_data_cleaned_final.csv
│ ├── pages/
│ │ └── Predictions.py
│ └── dashboard.py
│
├── DataCleaning&Preprocessing/
│ ├── Data-Science and AI Jobs - Indeed/
│ ├── Data-Science Job Postings & Skills/
│ ├── Data-Science Jobs & Salaries – Indeed/
│ ├── Data-Science, Data-Analyst & ML Jobs – Indeed/
│ ├── ML Engineer Jobs – Indeed/
│ ├── skill job dataset relationnel/
│ ├── data_preparation.ipynb
│ ├── scrapped_jobs_api.csv
│ └── SkillExtraction.py


```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend Models**: TensorFlow / Scikit-learn / Prophet
- **Data Sources**:

  - ✅ Public Datasets from Kaggle:
    - [Data Jobs and Skills](https://www.kaggle.com/datasets/tanvirachowdhury/data-jobs-and-skills)
    - [Data Science and AI Jobs – Indeed](https://www.kaggle.com/datasets/srivnaman/data-science-and-ai-jobsindeed)
    - [Data Science, Data Analyst, and ML Jobs – Indeed](https://www.kaggle.com/datasets/mdwaquarazam/data-science-dataanalyst-and-ml-job-indeed)
    - [Data Science Jobs and Salaries – Indeed](https://www.kaggle.com/datasets/ritiksharma07/data-science-jobs-andsalaries-indeed)
    - [Data Science, Data Analyst & ML Jobs from Indeed](https://www.kaggle.com/datasets/arnabk123/data-science-data-analyst-and-ml-jobs-from-indeed)
    - [Data Science Job Postings and Skills](https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills/data)
  - 🌐 Scraped job postings from:
    - **Indeed** (via Apify API)
    - **LinkedIn** (via Apify API)

- **Languages & Tools**: Python, Jupyter Notebooks, MongoDB, Apify, Transformers (Hugging Face)

---

## 👨‍💻 Author

Created with 💡 by BOUSKINE OTHMANE , Yassine Boulaalam , Taha Bouhafa

---

## 📄 License

This project is licensed under the MIT License.
