import pathlib
import platform
from fastai.vision.all import load_learner, PILImage

# ✅ Fix for loading FastAI model trained on Unix (e.g., Kaggle)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# ✅ Load the classifier model once (make sure the .pkl file is in the models/ folder)
model_path = 'models/classifier.pkl'  # Use relative path inside the Flask app
learn = load_learner(model_path)

# ✅ Function to classify wound type from an image path
def classify_wound(image_path):
    img = PILImage.create(image_path)
    pred, pred_idx, probs = learn.predict(img)
    print(pred)
    return str(pred)