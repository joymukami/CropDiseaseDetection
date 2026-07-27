from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import uuid
 
app = Flask(__name__)
 
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Load your trained crop health model
model = load_model('model/crop_health_model.keras')
 
# Load general-purpose ImageNet model for the "is this a crop?" filter
imagenet_model = MobileNetV2(weights='imagenet')
 
CLASS_NAMES = {0: 'healthy', 1: 'unhealthy'}
 
# Specific crop/fruit/vegetable keywords - narrow on purpose to avoid coincidental matches
CROP_RELATED_KEYWORDS = [
    'banana', 'corn', 'cauliflower', 'broccoli', 'mushroom', 'cucumber',
    'bell_pepper', 'zucchini', 'artichoke', 'pineapple', 'strawberry',
    'orange', 'lemon', 'fig', 'pomegranate', 'apple', 'cabbage',
    'squash', 'gourd', 'cardoon', 'rapeseed', 'daisy', 'acorn',
    'custard_apple', 'jackfruit'
]
 
 
def preprocess_image(img):
    img = img.convert('RGB').resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr
 
 
def is_likely_crop_image(arr, min_confidence=0.05, uncertainty_threshold=0.20):
    """
    Decide whether an uploaded image is likely to be a crop/plant/fruit photo,
    using the general-purpose ImageNet model as a filter before the crop-health model runs.
 
    Logic:
    1. If any of ImageNet's top-5 guesses matches a known crop/fruit/vegetable keyword
       with reasonable confidence -> accept.
    2. If ImageNet's top guess overall is low-confidence (it doesn't clearly recognize
       the image as anything specific) -> lean toward accepting, since this is common
       for unusual crop/leaf close-ups that ImageNet was never trained to classify.
    3. Otherwise, ImageNet is confidently guessing something specific and unrelated
       (e.g. "web_site", "envelope") -> reject.
    """
    preds = imagenet_model.predict(arr, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]
 
    # Step 1: keyword match
    for _, label, confidence in decoded:
        label_lower = label.lower()
        if confidence >= min_confidence and any(keyword in label_lower for keyword in CROP_RELATED_KEYWORDS):
            return True, label, float(confidence)
 
    top_label, top_confidence = decoded[0][1], float(decoded[0][2])
 
    # Step 2: genuine uncertainty -> allow through
    if top_confidence < uncertainty_threshold:
        return True, f"uncertain, allowed through ({top_label})", top_confidence
 
    # Step 3: confident about something unrelated -> reject
    return False, top_label, top_confidence
 
 
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    image_url = None
 
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            img = Image.open(file.stream)
 
            # Save the uploaded image so we can display it back
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.convert('RGB').save(filepath)
            image_url = f"/{filepath.replace(os.sep, '/')}"
 
            arr = preprocess_image(img)
            is_crop, guessed_label, guess_conf = is_likely_crop_image(arr)
 
            if not is_crop:
                error = f"This doesn't look like a crop image. (Detected: {guessed_label}, {guess_conf:.1%})"
            else:
                prob = model.predict(arr, verbose=0)[0][0]
                pred_class = int(prob > 0.5)
                label = CLASS_NAMES[pred_class]
                confidence = prob if pred_class == 1 else 1 - prob
                result = {'label': label.upper(), 'confidence': f"{confidence:.2%}"}
        else:
            error = "Please upload an image."
 
    return render_template('index.html', result=result, error=error, image_url=image_url)
 
 
if __name__ == '__main__':
    app.run(debug=True)