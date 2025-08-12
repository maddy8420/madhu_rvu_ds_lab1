import requests
import pandas as pd

url = "https://remoteok.io/api"
headers = {"User-Agent": "Mozilla/5.0"}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print("Error fetching data:", e)
    exit()

jobs = response.json()
jobs = jobs[1:]

job_list = []
for job in jobs:
    job_list.append({
        "Company Name": job.get("company"),
        "Job Role": job.get("position"),
        "Location": job.get("location"),
        "Features/Tags": ", ".join(job.get("tags", []))
    })

df = pd.DataFrame(job_list)
output_file = "remoteok_jobs.csv"
df.to_csv(output_file, index=False)

print(f"✅ Successfully extracted {len(df)} job listings.")
print(f"📄 Data saved to '{output_file}'.")
print("\nSource: RemoteOK (https://remoteok.com/)")
