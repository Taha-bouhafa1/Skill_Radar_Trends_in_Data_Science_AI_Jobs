from apify_client import ApifyClient
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# -------------------- CONFIGURATION --------------------
APIFY_API_TOKEN = os.getenv("APIFY_TOKEN")
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "job_database"
COLLECTION_NAME = "job_offers"

# -------------------- INITIALIZATION --------------------
print("🚀 Loading model...")


tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert_knowledge_extraction")
model = AutoModelForTokenClassification.from_pretrained("jjzha/jobbert_knowledge_extraction")

skill_ner = pipeline(
    task="token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

print("✅ Model loaded.")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
collection.create_index("Skills")  # Optional performance index
client = ApifyClient(APIFY_API_TOKEN)

# -------------------- HELPERS --------------------
def extract_country(location):
    US_STATES = {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY",
                 "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND",
                 "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"}
    US_FULL_NAMES = {"california", "texas", "new york", "florida", "georgia", "illinois", "washington", "ohio", "colorado"}

    if not location:
        return "Unknown"
    parts = location.split(",")
    last_part = parts[-1].strip()
    if last_part.upper() in US_STATES or last_part.lower() in US_FULL_NAMES:
        return "United States"
    elif len(parts) == 1:
        return "Unknown"
    else:
        return last_part

def normalize_salary(s):
    if not s or pd.isna(s):
        return None

    s_clean = s.replace(',', '').replace('CA$', '').replace('USD', '').replace('$', '').strip().lower()

    # Match range or single value with valid time unit
    pattern = r'(\d+\.?\d*)\s*(?:-|to)?\s*(\d+\.?\d*)?\s*/?\s*(per\s+)?(year|yr|month|mo|week|day|hour|hr|annum)?'

    match = re.search(pattern, s_clean)
    if not match:
        return None

    try:
        min_val = float(match.group(1))
        max_val = float(match.group(2)) if match.group(2) else min_val
        period = match.group(4)

        # 💡 Reject nonsensical values like $56 per year
        if period in ["year", "yr", "annum"] and min_val < 1000:
            return None
        if period in ["month", "mo"] and min_val < 100:
            return None
        if period in ["week"] and min_val < 25:
            return None
        if period in ["hour", "hr"] and min_val < 5:
            return None

        avg_val = (min_val + max_val) / 2

        # Normalize to annual
        if period in ["hour", "hr"]:
            return round(avg_val * 40 * 52)
        elif period == "week":
            return round(avg_val * 52)
        elif period in ["month", "mo"]:
            return round(avg_val * 12)
        elif period in ["day"]:
            return round(avg_val * 5 * 52)
        else:  # year or None defaults to year
            return round(avg_val)

    except Exception as e:
        print(f"❌ Error normalizing salary '{s}': {e}")
        return None


def clean_skills(skills):
    cleaned = []
    for skill in skills:
        skill = re.sub(r"[#()\[\]{}«»”“’‘•—]", "", skill)
        skill = re.sub(r"\s+", " ", skill).strip()
        skill = re.sub(r"\s?\/\s?", "/", skill)
        skill = re.sub(r"[^\w\s\-/+\.]", "", skill)
        if skill and len(skill) > 2 and skill.lower() not in ["##i", ".", "’s"]:
            cleaned.append(skill)
    return list(dict.fromkeys(cleaned))


def extract_skills(text):
    if not text:
        return []

    try:
        # 🧠 Properly truncate before passing to the pipeline
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        truncated_text = tokenizer.decode(encoding["input_ids"][0], skip_special_tokens=True)

        # ✅ Run the NER pipeline on the truncated string
        results = skill_ner(truncated_text)
    except Exception as e:
        print(f"❌ Error during skill extraction: {e}")
        return []

    # 🧹 Post-process results
    raw_skills = []
    current_skill = []

    for r in results:
        label = r["entity_group"]
        token = r["word"]
        if label == "B":
            if current_skill:
                raw_skills.append(" ".join(current_skill))
            current_skill = [token]
        elif label == "I":
            current_skill.append(token)

    if current_skill:
        raw_skills.append(" ".join(current_skill))

    return clean_skills(raw_skills)


# -------------------- APIFY ACTOR CONFIG --------------------
run_input = {
    "title": "AI engineer, Data scientist, Data engineer, Data analyst, ML engineer",
    "location": "",
    "companyName": [],
    "companyId": [],
    "publishedAt": "r86400", 
    "rows": 50,
    "proxy": {}
}

# -------------------- SCRAPE, PROCESS & SAVE --------------------
run = client.actor("BHzefUZlZRKWxkTck").call(run_input=run_input)
for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    desc = item.get("description", "").strip()
    desc = re.sub(r'\s+', ' ', desc)  # remove extra spaces/newlines

    skills = extract_skills(desc)
    location = extract_country(item.get("location", ""))
    raw_salary = item.get("salary", "").strip()

    # Only normalize if the original salary field has a value
    salary = normalize_salary(raw_salary) if raw_salary else None


    job = {
        "Job Title": item.get("title"),
        "Description": desc,
        "Location": location,
        "Date": item.get("publishedAt"),
        "Company": item.get("companyName"),
        "Salary": salary,
        "URL": item.get("jobUrl"),
        "Skills": ", ".join(skills) if skills else None
    }

    collection.insert_one(job)
    print(f"✅ Inserted: {job['Job Title']} — {job['Company']}")
