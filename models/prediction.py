import joblib
import numpy as np
import os

# Load trained model (3 features: age, area_mm2, diabetic_status)
model_path = os.path.join(os.path.dirname(__file__), 'healing_model.pkl')
model = joblib.load(model_path)

def predict_healing_time(metadata: np.ndarray) -> float:
    """
    Predict healing time based on metadata.
    Args:
        metadata (np.ndarray): Input features (shape: [1, 3])
    Returns:
        float: Predicted healing time (in days)
    """
    # Safety check
    if metadata.shape[1] != 3:
        raise ValueError(f"Expected 3 features (age, area_mm2, diabetic_status), but got {metadata.shape[1]}.")

    prediction = model.predict(metadata)[0]
    return prediction
