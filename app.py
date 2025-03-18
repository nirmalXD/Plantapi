import torch
from torchvision import transforms
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io

import pandas as pd

# Load disease information CSV once when the server starts
file_path = "disease_info.csv"  # Update this with the correct path
disease_info_df = pd.read_csv(file_path, encoding="latin1")  # Use appropriate encoding
disease_info_df["disease_name"] = disease_info_df["disease_name"].str.strip().str.lower()  # Normalize for matching


# Import model from models folder
from models.model import ResNet9

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Define number of classes (adjust based on your dataset)
num_diseases = 38  # Update according to the number of plant diseases

# Class names for diseases (replace these with actual names from your dataset)
class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", "Corn_(maize)___Common_rust", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Load model
model = ResNet9(in_channels=3, num_diseases=num_diseases)
model.load_state_dict(torch.load("plant-disease-model.pth", map_location=torch.device("cpu")))
model.eval()  # Set model to evaluation mode

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Ensure the correct input size
    transforms.ToTensor(),
])

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/c1")
# def c1():
#     return render_template("c1.html")

# @app.route("/predi")
# def predi():
#     return render_template("predi.html")

# @app.route("/plantindex")
# def plantindex():
#     return render_template("plantindex.html")


@app.route("/")
def home():
    return "Plant Disease Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty file uploaded"}), 400 
    
    # Validate image type
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
        confidence, prediction_index = torch.max(probabilities, dim=1)  # Get the highest confidence score

    confidence_percentage = round(confidence.item() * 100, 2)  # Convert to percentage

    if confidence_percentage < 81:  
        return jsonify({
            "prediction": "Unknown class",
            "message": "Please upload a different image with a clearer view of the plant."
        })

    predicted_class = class_names[prediction_index.item()]
    
    # Retrieve disease information from dataset using index number
    disease_info = disease_info_df.iloc[prediction_index.item()]
    
    description = disease_info['description'] if 'description' in disease_info else "No additional information available."
    steps = disease_info['Possible Steps'] if 'Possible Steps' in disease_info else "No treatment steps available."
    image_url = disease_info['image_url'] if 'image_url' in disease_info else ""
    
    return jsonify({
        "prediction": predicted_class,
        "confidence": f"{confidence_percentage}%",  # Format as percentage
        "description": description,
        "possible_steps": steps,
        # "image_url": image_url
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
