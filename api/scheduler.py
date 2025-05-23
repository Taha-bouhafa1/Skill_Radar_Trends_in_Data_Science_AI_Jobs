import schedule
import subprocess
import time
from datetime import datetime

# Paths to your scripts
SCRIPT_1 = "d:/cycle_ing/2eme anne bdia/S4/web scrapping/final/api/LinkedinApiScraping.py"
SCRIPT_2 = "d:/cycle_ing/2eme anne bdia/S4/web scrapping/final/api/IndeedApiScraping.py"

def run_script(script_path):
    print(f"[{datetime.now()}] ▶️ Running: {script_path}")
    try:
        subprocess.run(["E:/anaconda/python.exe", script_path], check=True)
        print(f"[{datetime.now()}] ✅ Finished: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] ❌ Error running {script_path}:\n{e}")

def job():
    run_script(SCRIPT_1)
    run_script(SCRIPT_2)

# Schedule the job once a day at a specific time (24h format)
schedule.every().day.at("18:45").do(job)

print("📅 Scheduler started. Waiting for next run...")
while True:
    schedule.run_pending()
    time.sleep(60)
