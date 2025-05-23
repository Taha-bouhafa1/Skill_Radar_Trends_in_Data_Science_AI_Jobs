import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
from collections import Counter
# --------------------------
# MongoDB Configuration
# --------------------------
MONGO_URI = "mongodb://localhost:27017"   # Change if needed
DB_NAME = "job_database"            # Replace with your DB name
COLLECTION_NAME = "job_offers"            # Replace with your collection name

# --------------------------
# Improved Blue Color Palette (High Contrast)
# --------------------------
CUSTOM_PALETTE = [
    "#caf0f8", "#90e0ef", "#00b4d8",
    "#0077b6", "#023e8a", "#03045e"
]

# --------------------------
# MongoDB Aggregation
# --------------------------
@st.cache_data
def get_job_count_by_country():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    pipeline = [
        {"$match": {"Location": {"$ne": None, "$ne": ""}}},
        {"$group": {"_id": "$Location", "Job Count": {"$sum": 1}}},
        {"$sort": {"Job Count": -1}}
    ]

    results = list(collection.aggregate(pipeline))
    df = pd.DataFrame(results)
    df.rename(columns={"_id": "Country"}, inplace=True)
    df["Country"] = df["Country"].str.strip().str.title()

    return df

# --------------------------
# Streamlit App
# --------------------------

st.set_page_config(page_title="Skill Radar: Trends in Data Science & AI Jobs", layout="wide")
# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    """
    This dashboard helps you:

📊 Explore job trends by title, company, country, and more

🔍 Forecast demand for tech skills over time

🧠 Recommend relevant skills based on your current profile

💰 Estimate salaries by job title and skillset


Ideal for job seekers, data professionals, and recruiters tracking the evolution of AI and data roles.
    """
)

# Main content
st.title("Skill Radar: Trends in Data Science & AI Jobs")


with st.spinner("🔄 Loading data from MongoDB..."):
    df = get_job_count_by_country()

# --------------------------
# Choropleth Heatmap
# --------------------------
col1, col2 = st.columns([1.5,1.5])
col3, col4 = st.columns(2)
col5, col6 = st.columns([2,1])
with col4:
    st.subheader("🏢 Top Companies by Job Offers")
    # --------------------------
    # Get Top Companies by Offer Count
    # --------------------------
    @st.cache_data
    def get_top_companies(limit=20):
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        pipeline = [
            {"$match": {"Company": {"$ne": None, "$ne": ""}}},
            {"$group": {"_id": "$Company", "Job Count": {"$sum": 1}}},
            {"$sort": {"Job Count": -1}},
            {"$limit": limit}
        ]

        result = list(collection.aggregate(pipeline))
        df = pd.DataFrame(result)
        df.rename(columns={"_id": "Company"}, inplace=True)
        return df

    # --------------------------
    # Section: Top Companies
    # --------------------------

    df_companies = get_top_companies(limit=20)

    fig_companies = px.bar(
        df_companies,
        x="Job Count",
        y="Company",
        orientation='h',
        title="Top Companies by Number of Job Offers",
        color_discrete_sequence=["#03045e"]  # darkest blue in your palette
    )

    fig_companies.update_layout(
        xaxis_title="Job Count",
        yaxis_title="Company",
        height=600,
        margin={"t":40, "r":0, "l":0, "b":0},
        yaxis=dict(autorange="reversed")  # largest at top
    )

    st.plotly_chart(fig_companies, use_container_width=True)


with col2:
    st.header("🌍 Job Offers by Country")
    fig = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="Job Count",
    color_continuous_scale=CUSTOM_PALETTE,
    title="Number of Job Offers per Country"
)

    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="rgba(0,0,0,0)",  # transparent borders
        showframe=False,
        showcoastlines=False
    )

    fig.update_layout(
        height=600,
        margin={"r":0, "t":40, "l":0, "b":0},
        coloraxis_colorbar=dict(title="Job Count")
    )

    st.plotly_chart(fig, use_container_width=True)


with col3:
    st.header("📈 Job Offers Over Time")
    # --------------------------
    # Get Job Count by Month (Python version)
    # --------------------------
    @st.cache_data
    def get_job_count_by_month():
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        # Only fetch the Date field
        cursor = collection.find({"Date": {"$ne": None}}, {"Date": 1})
        dates = list(cursor)
        df = pd.DataFrame(dates)

        # Convert string to datetime using the correct format
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
        df = df[df['Date'].notnull()]  # Remove invalid dates

        # Group by Month-Year
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        monthly = df.groupby('Month').size().reset_index(name='Job Count')
        return monthly.sort_values('Month')

    # --------------------------
    # Section: Job Count by Month
    # --------------------------

    df_monthly = get_job_count_by_month()

    fig_months = px.line(
        df_monthly,
        x="Month",
        y="Job Count",
        title="Number of Job Offers per Month",
        markers=True,
        color_discrete_sequence=["#0077b6"]
    )

    fig_months.update_layout(
        xaxis_title="Month",
        yaxis_title="Job Count",
        height=500,
        margin={"t":40, "r":0, "l":0, "b":0}
    )

    st.plotly_chart(fig_months, use_container_width=True)

with col1:
    st.header("• Top Job Titles ")
    # --------------------------
    # Get Top Job Titles for Treemap
    # --------------------------
    # Get Top Job Titles (Python cleaning)
    # --------------------------
    @st.cache_data
    def get_top_job_titles(limit=20):
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        # Only fetch the Job Title field
        cursor = collection.find({}, {"Job Title": 1})
        data = list(cursor)
        df = pd.DataFrame(data)

        # Drop missing or invalid entries
        df = df[df["Job Title"].notnull()]
        df["Job Title"] = df["Job Title"].astype(str).str.strip().str.lower()
        df = df[df["Job Title"] != ""]  # Remove empty strings

        # Count frequencies
        top_jobs = df["Job Title"].value_counts().head(limit).reset_index()
        top_jobs.columns = ["Job Title", "Job Count"]
        top_jobs["Job Title"] = top_jobs["Job Title"].str.title()  # Display formatting

        return top_jobs

    # --------------------------
    # Section: Job Title Treemap
    # --------------------------

    df_titles = get_top_job_titles(limit=20)

    fig_tree = px.treemap(
        df_titles,
        path=[px.Constant("All Jobs"), "Job Title"],
        values="Job Count",
        color="Job Count",
        color_continuous_scale=[
            "#caf0f8", "#90e0ef", "#00b4d8", "#0077b6", "#023e8a", "#03045e"
        ],
        title="Top Job Titles by Number of Offers"
    )

    fig_tree.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        height=600
    )

    st.plotly_chart(fig_tree, use_container_width=True)



with col5:
    # --------------------------
    # Get Top Skills from Skills Field
    # --------------------------
    @st.cache_data
    def get_top_skills(limit=20):
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        # Only fetch Skills field
        cursor = collection.find({"Skills": {"$ne": None}}, {"Skills": 1})
        data = list(cursor)
        df = pd.DataFrame(data)

        # Normalize and split skills
        all_skills = []
        for skills_str in df['Skills'].dropna():
            skills = [s.strip().lower() for s in str(skills_str).split(",") if s.strip()]
            all_skills.extend(skills)

        # Count frequency
        skills_series = pd.Series(all_skills)
        top_skills = skills_series.value_counts().head(limit).reset_index()
        top_skills.columns = ['Skill', 'Count']

        return top_skills

    # --------------------------
    # Section: Top Skills
    # --------------------------
    st.header("🛠️ Top In-Demand Skills")

    df_skills = get_top_skills(limit=20)

    fig_skills = px.bar(
        df_skills,
        x="Count",
        y="Skill",
        orientation='h',
        title="Top Requested Skills",
        color_discrete_sequence=["#0077b6"]
    )

    fig_skills.update_layout(
        xaxis_title="Count",
        yaxis_title="Skill",
        height=600,
        margin={"t":40, "r":0, "l":0, "b":0},
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig_skills, use_container_width=True)
with col6:
    # --------------------------
    # Normalize Job Titles
    # --------------------------
    def normalize_title(title):
        title = title.lower().strip()

        if "senior" in title and "data analyst" in title:
            return "Senior Data Analyst"
        elif "lead" in title and "data analyst" in title:
            return "Lead Data Analyst"
        elif "data analyst" in title:
            return "Data Analyst"

        elif "senior" in title and "data engineer" in title:
            return "Senior Data Engineer"
        elif "lead" in title and "data engineer" in title:
            return "Lead Data Engineer"
        elif "data engineer" in title:
            return "Data Engineer"

        elif "machine learning" in title or "ml engineer" in title:
            return "Machine Learning Engineer"
        elif "data scientist" in title and "senior" in title:
            return "Senior Data Scientist"
        elif "data scientist" in title:
            return "Data Scientist"

        elif "business analyst" in title:
            return "Business Analyst"
        elif "cloud engineer" in title:
            return "Cloud Engineer"
        elif "software engineer" in title:
            return "Software Engineer"
        elif "database administrator" in title:
            return "Database Administrator"
        else:
            return "Other"

    # --------------------------
    # Get Skills by Normalized Job Title
    # --------------------------
    @st.cache_data
    def get_job_title_skills():
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]

        # Fetch only job title and skills
        cursor = collection.find(
            {"Job Title": {"$ne": None}, "Skills": {"$ne": None}},
            {"Job Title": 1, "Skills": 1}
        )
        data = list(cursor)
        df = pd.DataFrame(data)

        # Clean and normalize Job Titles
        df = df[df["Job Title"].notnull()]
        df["Job Title"] = df["Job Title"].astype(str).str.strip()
        df = df[df["Job Title"] != ""]

        df["Normalized Title"] = df["Job Title"].apply(normalize_title)

        # Clean and split Skills
        df["Skills"] = df["Skills"].astype(str).apply(
            lambda x: [s.strip().lower() for s in x.split(",") if s.strip()]
        )

        return df

    # --------------------------
    # Section: Pie Chart of Skills by Job Title
    # --------------------------
    st.header("• Skill Distribution per Job Title")

    df_skills_by_title = get_job_title_skills()

    # Dropdown for normalized job title
    # Exclude 'Other' from dropdown
    top_titles = (
        df_skills_by_title[df_skills_by_title["Normalized Title"] != "Other"]
        ["Normalized Title"]
        .value_counts()
        .head(20)
        .index
        .tolist()
    )
    selected_title = st.selectbox("Select a Job Title", top_titles)

    # Filter by normalized title
    filtered_df = df_skills_by_title[
        df_skills_by_title["Normalized Title"] == selected_title
    ]

    # Flatten all skills
    all_skills = [skill for skills_list in filtered_df["Skills"] for skill in skills_list]
    skill_counts = Counter(all_skills)

    # Create skill DataFrame
    skill_df = pd.DataFrame(skill_counts.items(), columns=["Skill", "Count"])
    skill_df = skill_df.sort_values("Count", ascending=False).head(15)

    # Plot Pie Chart
    fig_pie = px.pie(
        skill_df,
        names="Skill",
        values="Count",
        title=f"Skill Distribution for {selected_title}",
        color_discrete_sequence=[
            "#03045e", "#023e8a", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8"
        ]
    )

    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)
