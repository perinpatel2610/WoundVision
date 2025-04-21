from flask import Flask, request, render_template

from explainability.xai_tools import generate_explanations
from models.classify_image import classify_wound
from models.prediction import predict_healing_time
from models.segmentation_model import segment_image, compute_wound_area
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads/'
OUTPUT_FOLDER = './static/outputs/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    original_image = cv2.imread(file_path)
    if original_image is None:
        return "Invalid image file", 400

    try:
        segmentation_mask = segment_image(original_image)
    except Exception as e:
        return f"Error during segmentation: {str(e)}", 500

    output_path = os.path.join(OUTPUT_FOLDER, f"{file.filename}_segmented.png")
    cv2.imwrite(output_path, (segmentation_mask * 255).astype(np.uint8))

    # Compute wound area
    area_pixels, area_mm2 = compute_wound_area(segmentation_mask)

    try:
        age = int(request.form.get('age'))
        diabetes_status = int(request.form.get('diabetes_status'))

        # location = request.form.get('location')
    except ValueError:
        return "Invalid input for age or diabetes status", 400

    # Your model expects [age, wound_area, diabetic_status]
    print(age,area_mm2,diabetes_status)
    metadata = np.array([[age, area_mm2, diabetes_status]])

    try:
        wound_type = classify_wound(file_path)
    except Exception as e:
        return f"Error during classification: {str(e)}",500
    try:
        prediction = predict_healing_time(metadata)
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

    print("---------------------------prediction",prediction)
    if prediction < 30:
        readable_time = f"{prediction:.1f} days"
    elif prediction < 365:
        readable_time = f"{(prediction / 30):.1f} months"
    elif prediction < 365 * 10:
        readable_time = f"{(prediction / 365):.1f} years"
    else:
        readable_time = "More than 10 years"

    try:
        explanation = generate_explanations(metadata, segmentation_mask)
    except Exception as e:
        return f"Error during explainability: {str(e)}", 500

    return render_template(
        'results.html',
        prediction=readable_time,
        explanation=explanation,
        wound_type=wound_type,
        image_path=f"/static/outputs/{file.filename}_segmented.png",
        area_pixels=area_pixels,
        area_mm2=round(area_mm2, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)
