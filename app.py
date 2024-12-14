from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from approach.ResEmoteNet import ResEmoteNet
import os

app = Flask(__name__)

# Emotions labels
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA if available
model = ResEmoteNet().to(device)
model_path = './models/rafdb_model_revised.pth'  # Path for deployed model
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(image):
    """Detect emotion scores from the given image."""
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    rounded_scores = [round(float(score), 2) for score in scores]  # Convert to Python float
    return rounded_scores

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "No image file in the request"}), 400
    
    file = request.files['image']
    
    try:
        # Read the image file as a PIL Image
        img = Image.open(file.stream)
        img = img.convert('RGB')
        img = np.array(img)
        
        # Detect faces and get emotion scores
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(24, 24))
        
        result = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Crop the face region
            crop_img = img[y: y + h, x: x + w]
            pil_crop_img = Image.fromarray(crop_img)
            
            # Get emotion scores for the cropped face
            rounded_scores = detect_emotion(pil_crop_img)
            
            # Create a dictionary for emotion scores
            emotion_data = {emotion: rounded_scores[i] for i, emotion in enumerate(emotions)}
            result.append(emotion_data)
        
        return jsonify({"emotion_results": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
