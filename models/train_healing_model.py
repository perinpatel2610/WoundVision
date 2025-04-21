import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# === CONFIGURATION ===
CSV_PATH = r'models/image_metadata_with_synthetic_healing.csv'
MODEL_SAVE_PATH = 'healing_model.pkl'

# === Load and clean data ===
df = pd.read_csv(CSV_PATH)

df = df.dropna(subset=['age', 'area_pixels', 'diabetic_status', 'synthetic_healing_time'])

# Convert diabetic status to numeric
df['diabetic_status'] = df['diabetic_status'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

# Convert pixel area to mm² if needed (already in mm²? skip this)
if 'wound_area_mm2' not in df.columns:
    df['wound_area_mm2'] = df['area_pixels'] / 100  # adjust if necessary

# === Select only 3 features ===
X = df[['age', 'wound_area_mm2', 'diabetic_status']]
y = df['synthetic_healing_time']

# === Train/val split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
val_preds = model.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
print(f"✅ Validation MAE: {mae:.2f} days")

# === Save model ===
joblib.dump(model, MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
