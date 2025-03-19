import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import pandas as pd

# Load disease information CSV
file_path = "disease_info.csv"  # Update with the correct path
disease_info_df = pd.read_csv(file_path, encoding="latin1")
disease_info_df["disease_name"] = disease_info_df["disease_name"].str.strip().str.lower()

# Import model
from models.model import ResNet9

# Initialize FastAPI
app = FastAPI()

# Define number of classes
num_diseases = 38

# Class names for diseases
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
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.get("/")
def home():
    return {"message": "Plant Disease Prediction API is Running"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Validate image type
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
    
    img = transform(img).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction_index = torch.max(probabilities, dim=1)

    confidence_percentage = round(confidence.item() * 100, 2)

    if confidence_percentage < 81:
        return JSONResponse(
            content={"prediction": "Unknown class", "message": "Upload a clearer plant image."},
            status_code=200
        )
    
    predicted_class = class_names[prediction_index.item()]
    disease_info = disease_info_df.iloc[prediction_index.item()]
    
    return {
        "prediction": predicted_class,
        "confidence": f"{confidence_percentage}%",
        "description": disease_info.get("description", "No additional information available."),
        "possible_steps": disease_info.get("Possible Steps", "No treatment steps available."),
    }

# Run with: uvicorn filename:app --host 0.0.0.0 --port 4000 --reload
