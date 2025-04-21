import joblib
import numpy as np
import os
import shap

# === Load the trained model ===
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'healing_model.pkl')
model = joblib.load(model_path)

# === Initialize SHAP Explainer ===
explainer = shap.Explainer(model)

# === Main XAI Function ===
def generate_explanations(metadata: np.ndarray, segmentation_results: np.ndarray) -> str:
    """
    Generate explanations for the prediction using SHAP.

    Args:
        metadata (np.ndarray): Metadata input for the model (shape: [1, 3]).
        segmentation_results (np.ndarray): Segmentation output from the model (not used here).

    Returns:
        str: Human-friendly explanation including patient inputs, SHAP feature impact, and model reasoning.
    """
    # Ensure correct shape
    metadata = np.array(metadata).reshape(1, -1)

    # Extract actual input values
    actual_age = metadata[0, 0]
    actual_area = metadata[0, 1]
    actual_diabetic = int(metadata[0, 2])

    # Run SHAP
    shap_values = explainer(metadata)
    shap_scores = shap_values.values[0]  # shape: (3,)
    age_score, area_score, diabetic_score = shap_scores

    # Get baseline and predicted value
    base_value = explainer.expected_value
    predicted_value = base_value + np.sum(shap_scores)

    # Begin explanation report
    report = ""
    # report += f"📊 Predicted Healing Duration: {predicted_value:.1f} days\n\n"

    report += f"Patient Inputs:\n"
    report += f"- Age: {actual_age:.1f} years\n"
    report += f"- Wound Area: {actual_area:.2f} mm²\n"
    report += f"- Diabetic Status: {'Yes' if actual_diabetic else 'No'}\n\n"

    report += "Feature Impact on Prediction:\n\n"

    # --- Age Explanation ---
    if age_score < -4.5:
        report += (
            f"🧓 Age significantly reduced healing time (SHAP impact: {age_score:+.2f}).\n"
            f"   ↳ The patient's age appears to favor faster healing in this case, possibly due to other compensating health indicators or dataset bias.\n\n"
        )
    elif age_score > 2:
        report += (
            f"🧓 Age significantly increased healing time (SHAP impact: {age_score:+.2f}).\n"
            f"   ↳ Older age is generally associated with slower wound healing due to reduced cellular regeneration, which aligns with the model's interpretation.\n\n"
        )
    else:
        report += (
            f"🧓 Age had minimal impact on healing time (SHAP impact: {age_score:+.2f}).\n"
            f"   ↳ The model considers age to have limited influence in this prediction. Other factors likely played a larger role.\n\n"
        )

    # --- Wound Area Explanation ---
    if area_score > 2:
        report += (
            f"📏 Wound area slightly delayed healing (SHAP impact: {area_score:+.2f}).\n"
            f"   ↳ Larger wound areas often require more time to close, and the model detected a delay effect from the wound size.\n\n"
        )
    elif area_score < -2:
        report += (
            f"📏 Wound area helped accelerate healing (SHAP impact: {area_score:+.2f}).\n"
            f"   ↳ A smaller wound area may signal faster recovery, and the model reflected that in this case.\n\n"
        )
    else:
        report += (
            f"📏 Wound area had little effect on healing time (SHAP impact: {area_score:+.2f}).\n"
            f"   ↳ The wound size did not strongly influence the outcome according to the model.\n\n"
        )

    # --- Diabetic Status Explanation ---
    if actual_diabetic == 1:
        report += (
            f"💉 Diabetes increased healing time by design (+30 penalty applied). SHAP impact: {diabetic_score:+.2f}\n"
            f"   ↳ Diabetic wounds generally heal more slowly due to impaired circulation and immune response. The model accounts for this via a fixed penalty mechanism.\n\n"
        )
    else:
        report += (
            f"💉 Non-diabetic patient received a mild delay (+10 penalty applied). SHAP impact: {diabetic_score:+.2f}\n"
            f"   ↳ Even in non-diabetic cases, the model applies a slight adjustment to balance other healing variabilities.\n\n"
        )

    return report
