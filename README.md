# 🩺 WoundVision - AI-Powered Wound Assessment System

**WoundVision** is an AI-driven web application designed to assist clinicians in wound analysis. It performs **wound segmentation**, **wound type classification**, and **healing time prediction**, integrating deep learning models with patient metadata and explainable AI (XAI).

---

## 📌 Features

- 🔍 **Wound Type Classification**
  Predicts the type of wound (e.g., Burn, Bruise, Diabetic Ulcer) using a pre-trained CNN.

- 🧠 **Wound Segmentation**
  Uses a hybrid **ResNet152 + U-Net** model to segment wound boundaries in medical images.

- ⏱️ **Healing Time Prediction**
  Combines image features and metadata (age, diabetic status, wound area) using **XGBoost** to estimate healing duration.

- 💬 **Explainability with SHAP**
  Generates human-readable explanations of model predictions using SHAP values.

- 🌐 **Flask Web Application**
  User-friendly web interface to upload images, view results, and download reports.

---

## 🗂️ Project Structure

```
WoundVision/
├── app.py                         # Flask app entry point
├── templates/
│   └── results.html               # Output UI template
│   └── index.html
├── static/
│   └── outputs/                   # Segmentation result images
│   └── uploads/                   # Images that user will input
├── models/
│   ├── Segmentation_model
│        └──train_model.py         # To train segmentation model
│        └──model.py
│   ├── healing.py                 # To train healing time prediction model
│   ├── segmentation_model.py      # ResNet152 + U-Net definition
│   ├── train_healing_model.py     # Training code for healing prediction
│   ├── healing_model.pkl          # Trained XGBoost model
│   ├── classifier.pkl             # Keras model for wound type classification
│   ├── classify_image.py          # To access the classification model
│   └── prediction.py              # To access prediction model
├── explainability/
│   └── xai_tools.py               # SHAP-based explanation logic
├── utils/
│   └── preprocessing.py           # Image preprocessing and metadata formatting
├── data/
│   └── matched_metadata.csv       # Combined image & metadata table
├── train_images/, test_images/    # Wound image datasets
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/WoundVision.git
cd WoundVision
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

> ⚠️ Note: SHAP version should be **≤ 0.40.0** if using `shap.Explainer(model)` directly.

---

## 🚀 Usage

NOTE: Activate the virtual environment

1. **Start the Flask app:**
```bash
python app.py
```

2. **Open in browser:**
```
http://localhost:5000
```

3. **Upload an image**, provide metadata, and view:
   - Predicted wound type
   - Segmented wound area
   - Healing time estimate
   - SHAP explanation report

---

## 🧠 Models Used

| Task | Model |
|------|-------|
| Wound Type Classification | ResNet / MobileNet (Keras) |
| Wound Segmentation | ResNet152 + U-Net |
| Healing Time Prediction | XGBoost Regressor |
| Explainability | SHAP TreeExplainer |

---

## 📊 Example Output

- Segmented wound image overlay
- Prediction: “Diabetic Ulcer”
- Healing Time Estimate: **21.3 days**
- SHAP Explanation: “Wound area and age increased healing time significantly.”

---

## 🤝 Contributors

- **Rudri Jardosh, Perin Patel, Maitri Patel, Fenil Patel** – Master of Applied Computing
- Faculty advisor - Dr. ANK Zaman

---

## 💬 Acknowledgements

Thanks to datasets from Kaggle and medical research publications that helped support the development of this project.
