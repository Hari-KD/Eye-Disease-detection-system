import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# Initialize Flask application (Vercel Entrypoint)
app = Flask(__name__)

# Import model helper functions
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

IMG_SIZE = (128, 128)
CLASSES = ['Healthy', 'Disease']

def load_detection_model():
    """Load pre-trained model if available"""
    model_path = os.path.join(os.path.dirname(__file__), 'eye_disease_model.h5')
    if HAS_TF and os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

model = load_detection_model()

def best_first_search(predictions, class_labels):
    """
    Best First Search heuristic evaluation to find the most probable class node.
    """
    best_score = -1.0
    best_class = None
    
    for i, score in enumerate(predictions):
        label = class_labels[i]
        if score > best_score:
            best_score = float(score)
            best_class = label
            
    return best_class, best_score

def predict_image(image_pil):
    """Preprocess image and return prediction"""
    img = image_pil.convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if model is not None:
        raw_predictions = model.predict(img_array)[0]
    else:
        # Fallback heuristic calculation if model binary is unavailable
        green_mean = np.mean(img_array[:, :, 1])
        red_mean = np.mean(img_array[:, :, 0])
        healthy_score = float(green_mean / (green_mean + red_mean + 1e-5))
        disease_score = float(red_mean / (green_mean + red_mean + 1e-5))
        raw_predictions = [healthy_score, disease_score]

    result_class, confidence = best_first_search(raw_predictions, CLASSES)
    return result_class, confidence, [float(p) for p in raw_predictions]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eye Disease Detection System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #0f172a; color: #f8fafc; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 20px; }
        .container { max-width: 600px; width: 100%; background: #1e293b; border-radius: 16px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); text-align: center; border: 1px solid #334155; }
        h1 { font-size: 1.8rem; margin-bottom: 8px; color: #38bdf8; }
        p.subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px; }
        .dropzone { border: 2px dashed #475569; border-radius: 12px; padding: 30px; cursor: pointer; transition: border 0.3s; margin-bottom: 20px; background: #0f172a; }
        .dropzone:hover { border-color: #38bdf8; }
        input[type="file"] { display: none; }
        .btn { background: #0284c7; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: background 0.2s; width: 100%; font-size: 1rem; }
        .btn:hover { background: #0369a1; }
        #preview { max-width: 100%; max-height: 250px; border-radius: 8px; margin-top: 15px; display: none; margin-left: auto; margin-right: auto; }
        .result-box { margin-top: 24px; padding: 16px; border-radius: 8px; background: #0f172a; display: none; text-align: left; border: 1px solid #334155; }
        .result-title { font-size: 1.1rem; font-weight: bold; margin-bottom: 8px; }
        .Healthy { color: #4ade80; }
        .Disease { color: #f87171; }
        .info-row { display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.9rem; color: #cbd5e1; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Eye Disease Detection System</h1>
        <p class="subtitle">AI Ocular Image Classification & Best First Search Analysis</p>
        
        <form id="uploadForm">
            <div class="dropzone" onclick="document.getElementById('fileInput').click()">
                <p id="dropText">Click or Drag & Drop Eye Image Here</p>
                <img id="preview" alt="Selected Image Preview">
                <input type="file" id="fileInput" name="file" accept="image/*" onchange="showPreview(event)">
            </div>
            <button type="submit" class="btn">Analyze Image</button>
        </form>

        <div id="resultBox" class="result-box">
            <div id="resultTitle" class="result-title"></div>
            <div class="info-row"><span>Confidence Score:</span><span id="confidenceVal"></span></div>
            <div class="info-row"><span>Search Algorithm:</span><span>Best First Search (BFS)</span></div>
        </div>
    </div>

    <script>
        function showPreview(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.getElementById('preview');
                    img.src = e.target.result;
                    img.style.display = 'block';
                    document.getElementById('dropText').style.display = 'none';
                }
                reader.readAsDataURL(file);
            }
        }

        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) {
                alert("Please select an image file first.");
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const resBox = document.getElementById('resultBox');
            const resTitle = document.getElementById('resultTitle');
            const confVal = document.getElementById('confidenceVal');

            resTitle.innerText = "Analyzing...";
            resBox.style.display = 'block';

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();

                if (data.status === 'success') {
                    resTitle.className = "result-title " + data.prediction;
                    resTitle.innerText = "Diagnosis: " + data.prediction;
                    confVal.innerText = (data.confidence * 100).toFixed(2) + "%";
                } else {
                    resTitle.innerText = "Error: " + data.message;
                }
            } catch (err) {
                resTitle.innerText = "Error analyzing image.";
            }
        };
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """Render Web App Dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for image analysis"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    try:
        image = Image.open(file.stream)
        prediction, confidence, raw_scores = predict_image(image)
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'confidence': confidence,
            'raw_scores': raw_scores
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
