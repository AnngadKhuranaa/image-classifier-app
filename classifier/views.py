from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model once when server starts
model = MobileNetV2(weights='imagenet')

def index(request):
    return render(request, 'classifier/index.html')

def predict_image(request):
    if request.method == "POST" and request.FILES.get("file"):
        uploaded_file = request.FILES["file"]

        # Save file
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_url = fs.url(filename)

        # Load and preprocess image
        img_path = fs.path(filename)
        img = image.load_img(img_path, target_size=(224, 224))  # MobileNetV2 expects 224x224
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make prediction
        preds = model.predict(x)
        decoded = decode_predictions(preds, top=1)[0][0]  # top prediction
        predicted_label = decoded[1]   # e.g., "tabby_cat"
        confidence = round(decoded[2] * 100, 2)  # confidence %

        return render(request, "classifier/result.html", {
            "file_url": file_url,
            "predicted_label": predicted_label,
            "confidence": confidence
        })

    return render(request, 'classifier/index.html')
