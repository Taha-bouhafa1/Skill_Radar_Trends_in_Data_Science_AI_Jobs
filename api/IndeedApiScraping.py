from apify_client import ApifyClient
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()


# -------------------- CONFIGURATION --------------------
APIFY_API_TOKEN = os.getenv("APIFY_TOKEN")
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "job_scraper_db"
COLLECTION_NAME = "indeed_cleaned_jobs"

# -------------------- INITIALIZATION --------------------
print("🚀 Loading model...")
tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert_knowledge_extraction")
model = AutoModelForTokenClassification.from_pretrained("jjzha/jobbert_knowledge_extraction")
skill_ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
print("✅ Model loaded.")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
collection.create_index("Skills")

client = ApifyClient(APIFY_API_TOKEN)

# -------------------- HELPERS --------------------
def extract_country(location):
    return location.split(",")[-1].strip() if location else "Unknown"

def normalize_salary(s: str | None) -> int | None:
    """
    Convert any hourly / daily / weekly / monthly / yearly pay string
    into an **annual** integer salary (USD-equivalent). Returns None
    if nothing parsable is found.
    """
    if not s or pd.isna(s):
        return None

    # Clean up currency symbols, commas, upper/lower case
    s_clean = (
        re.sub(r"[\$£€]", "", s)      # remove $, £, €
          .replace(",", "")           # 80,000  -> 80000
          .lower()
          .strip()
    )

    # ------------------------------------------------------------------
    #  1️⃣  Grab the min / max amount and (optional) unit.
    #      ────────────────────────────────────────────────────────────
    #   ▸ works with “up to …”, “from …”,   en-dash / em-dash,
    #     “to”, “–”, “—”, “per”, “a”, “an”, etc.
    # ------------------------------------------------------------------
    pattern = (
        r"(?:up to|from)?\s*"                   # prefix      ── optional
        r"(\d+(?:\.\d+)?)"                      # min amount  ── group 1
        r"(?:\s*[-–—to]+\s*(\d+(?:\.\d+)?))?"   # max amount  ── group 2
        r"\s*(?:per|a|an)?\s*"                  # divider     ── optional
        r"(hourly|hour|hr|"                    # unit        ── group 3
        r"daily|day|weekly|week|monthly|month|mo|"
        r"yearly|year|yr|annually|annum)?"
    )

    m = re.search(pattern, s_clean)
    if not m:
        return None

    min_val = float(m.group(1))
    max_val = float(m.group(2)) if m.group(2) else min_val
    avg_val = (min_val + max_val) / 2
    unit    = (m.group(3) or "").lower()

    # ------------------------------------------------------------------
    #  2️⃣  Fallback: if the unit group is empty, sniff it in the text.
    # ------------------------------------------------------------------
    if not unit:
        for key in ("hour", "hr", "day", "week", "month", "mo", "year", "yr"):
            if key in s_clean:
                unit = key
                break

    # ------------------------------------------------------------------
    #  3️⃣  Annualise
    # ------------------------------------------------------------------
    match unit:
        case "hour" | "hr" | "hourly":
            annual = avg_val * 40 * 52        # 40 h / week
        case "day" | "daily":
            annual = avg_val * 5  * 52        # 5 d / week
        case "week" | "weekly":
            annual = avg_val * 52
        case "month" | "mo" | "monthly":
            annual = avg_val * 12
        case _:                               # year / yearly / missing → assume annual
            annual = avg_val

    return round(annual)



def clean_skills(skills):
    cleaned = []
    for skill in skills:
        skill = re.sub(r"[^\w\s\-/+\.]", "", skill).strip()
        if len(skill) > 2 and skill.lower() not in ["##i", ".", "’s"]:
            cleaned.append(skill)
    return list(dict.fromkeys(cleaned))

def extract_skills(text):
    if not text:
        return []
    try:
        encoding = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        truncated = tokenizer.decode(encoding["input_ids"][0], skip_special_tokens=True)
        results = skill_ner(truncated)
    except Exception as e:
        print("❌ Skill extraction failed:", e)
        return []
    raw, current = [], []
    for r in results:
        if r["entity_group"] == "B":
            if current: raw.append(" ".join(current))
            current = [r["word"]]
        elif r["entity_group"] == "I":
            current.append(r["word"])
    if current: raw.append(" ".join(current))
    return clean_skills(raw)

# -------------------- SCRAPING --------------------
run_input = {
    "position": "AI engineer, Data scientist, Data engineer, Data analyst, ML engineer",
    "country": "US",
    "location": "",
    "maxItems": 10,
    "parseCompanyDetails": False,
    "saveOnlyUniqueItems": True,
    "followApplyRedirects": False,
}

run = client.actor("hMvNSpz3JnHgl5jkh").call(run_input=run_input)

for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    desc = re.sub(r"\s+", " ", item.get("description", ""))
    skills = extract_skills(desc)
    raw_salary = item.get("salary", "")
    salary = normalize_salary(raw_salary)
    from datetime import datetime

    # Normalize date format to dd-mm-yyyy
    raw_date = item.get("postingDateParsed") or item.get("postedAt")
    try:
        parsed_date = datetime.fromisoformat(raw_date) if raw_date else None
        formatted_date = parsed_date.strftime("%d-%m-%Y") if parsed_date else None
    except Exception as e:
        print(f"❌ Failed to parse date '{raw_date}': {e}")
        formatted_date = None

    job = {
        "Job Title": item.get("positionName"),
        "Description": desc,
        "Location": item.get("location"),
        "Country": extract_country(item.get("location", "")),
        "Company": item.get("company"),
        "Date": formatted_date,
        "Salary": salary,
        "URL": item.get("url"),
        "Skills": ", ".join(skills) if skills else None
    }
    collection.insert_one(job)
    print(f"✅ Inserted: {job['Job Title']} — {job['Company']} — Salary: {salary}")
