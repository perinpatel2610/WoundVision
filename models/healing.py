#This file was to generate the csv file of the metadata of the patients!
# NO NEED TO RUN THIS


import pandas as pd
import numpy as np

# === CONFIGURATION ===
CSV_PATH = r'models\image_metadata_with_random_info.csv'
OUTPUT_CSV = 'image_metadata_with_synthetic_healing.csv'

# === Load Metadata ===
df = pd.read_csv(CSV_PATH)

# === Drop rows with missing values ===
df = df.dropna(subset=['age', 'area_pixels', 'diabetic_status'])

# === Convert wound area from pixels to approx cm² ===
df['wound_area_cm2'] = df['area_pixels'] / 100.0  # 10,000 pixels ≈ 1 cm²

# === Generate Synthetic Healing Time ===
synthetic_healing = []

for _, row in df.iterrows():
    age = row['age']
    area_cm2 = row['wound_area_cm2']
    diabetic = row['diabetic_status']

    # Rule-based simulation
    base_days = 40
    age_penalty = 0.2 * age
    area_penalty = 1.5 * area_cm2
    diabetic_penalty = 50 if diabetic == 1 else 10
    noise = np.random.normal(loc=0, scale=1.5)  # small realistic noise

    healing_time = base_days + age_penalty + area_penalty + diabetic_penalty + noise
    healing_time = max(1, int(round(healing_time)))  # round to whole days, minimum 1

    synthetic_healing.append(healing_time)

# === Save new synthetic healing time column ===
df['synthetic_healing_time'] = synthetic_healing
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Synthetic healing time added (using wound area in cm²). Saved to {OUTPUT_CSV}")
