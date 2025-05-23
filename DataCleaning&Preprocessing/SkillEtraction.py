from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pymongo import MongoClient
from tqdm import tqdm
import re
import traceback

# ------------------------------ CONFIG ------------------------------
DB_NAME = "job_database"                  # ← Change if needed
COLLECTION_NAME = "job_offers"           # ← Change if needed
MONGO_URI = "mongodb://localhost:27017/"

# ------------------------------ Load Model ------------------------------
print("🚀 Loading model...")
tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert_knowledge_extraction")
model = AutoModelForTokenClassification.from_pretrained("jjzha/jobbert_knowledge_extraction")
skill_ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
print("✅ Model loaded.")

# ------------------------------ Connect MongoDB ------------------------------
print("🔌 Connecting to MongoDB...")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
collection.create_index("Skills")  # Optional index
print(f"✅ Connected to collection '{COLLECTION_NAME}' in DB '{DB_NAME}'.")

# ------------------------------ Clean Extracted Skills ------------------------------
def clean_skills(skills):
    cleaned = []
    for skill in skills:
        skill = re.sub(r"[#()\[\]{}«»”“’‘•—]", "", skill)
        skill = re.sub(r"\s+", " ", skill).strip()
        skill = re.sub(r"\s?\/\s?", "/", skill)
        skill = re.sub(r"[^\w\s\-/+\.]", "", skill)
        if skill and len(skill) > 2 and skill.lower() not in {"##i", ".", "’s"}:
            cleaned.append(skill)
    return list(dict.fromkeys(cleaned))  # Remove duplicates, preserve order

# ------------------------------ Extract from Full Description ------------------------------
def extract_skills_full_text(text, max_tokens=512):
    if not text:
        return []

    try:
        # Tokenize without truncation
        encoding = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encoding["input_ids"][0]

        # Chunk input_ids
        chunks = [input_ids[i:i + max_tokens] for i in range(0, len(input_ids), max_tokens)]

        all_skills = []
        for chunk in chunks:
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            results = skill_ner(chunk_text)

            current_skill = []
            raw_skills = []

            for r in results:
                if r["entity_group"] == "B":
                    if current_skill:
                        raw_skills.append(" ".join(current_skill))
                    current_skill = [r["word"]]
                elif r["entity_group"] == "I":
                    current_skill.append(r["word"])

            if current_skill:
                raw_skills.append(" ".join(current_skill))

            all_skills.extend(raw_skills)

        return clean_skills(all_skills)

    except Exception as e:
        print(f"❌ Error extracting skills: {e}")
        return []

# ------------------------------ Mongo Query ------------------------------
query = {
    "$or": [
        {"Skills": {"$exists": False}},
        {"Skills": None},
        {"Skills": ""}
    ],
    "Description": {"$exists": True}
}

total = collection.count_documents(query)
print(f"📊 Found {total} documents missing skills.")

cursor = collection.find(query, no_cursor_timeout=True)

# ------------------------------ Process and Update ------------------------------
updated_count = 0

for doc in tqdm(cursor, total=total, desc="⏳ Processing"):
    try:
        description = doc.get("Description", "")
        skills = extract_skills_full_text(description)

        if skills:
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"Skills": ", ".join(skills)}}
            )
            updated_count += 1
            print(f"✅ Updated _id: {doc['_id']} with {len(skills)} skills.")
        else:
            print(f"⚠️ No skills found for _id: {doc['_id']}")

    except Exception as e:
        print(f"❌ Error processing _id: {doc.get('_id')}")
        traceback.print_exc()
        continue

cursor.close()
print(f"\n🎉 Done. Total documents updated: {updated_count} / {total}")
