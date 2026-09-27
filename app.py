import os
import io
import time
import base64
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, render_template_string

# Initialize Flask Application
app = Flask(__name__)

# Import TensorFlow safely
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

IMG_SIZE = (128, 128)
CLASSES = ['Healthy', 'Disease']

# Load Haar Cascades for Eye and Face Detection
EYE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye.xml'
EYE_TREE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
eye_tree_cascade = cv2.CascadeClassifier(EYE_TREE_CASCADE_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def load_detection_model():
    """Load pre-trained Keras CNN model"""
    model_path = os.path.join(os.path.dirname(__file__), 'eye_disease_model.h5')
    if HAS_TF and os.path.exists(model_path):
        try:
            print(f"Loading retrained Keras model from {model_path}...")
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    print("Model file not found or TensorFlow unavailable. Using fallback heuristic classifier.")
    return None

model = load_detection_model()

def reload_model_if_needed():
    """Helper to reload model if retrained file updated"""
    global model
    model = load_detection_model()

def best_first_search(predictions, class_labels):
    """
    Implements Best First Search to select the most probable class node.
    Node 0: Healthy, Node 1: Disease
    """
    best_score = -1.0
    best_class = None
    
    for i, score in enumerate(predictions):
        label = class_labels[i]
        if score > best_score:
            best_score = float(score)
            best_class = label
            
    return best_class, best_score

def classify_eye_crop(eye_bgr):
    """Process an eye image crop (numpy array in BGR) and return classification results"""
    eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(eye_rgb).resize(IMG_SIZE)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Reload model if current reference is stale
    global model
    if model is None:
        model = load_detection_model()

    if model is not None:
        raw_predictions = model.predict(img_array, verbose=0)[0]
    else:
        # Fallback heuristic calculation if model is unavailable
        green_mean = np.mean(img_array[:, :, 1])
        red_mean = np.mean(img_array[:, :, 0])
        total = green_mean + red_mean + 1e-5
        healthy_score = float(green_mean / total)
        disease_score = float(red_mean / total)
        raw_predictions = [healthy_score, disease_score]

    result_class, confidence = best_first_search(raw_predictions, CLASSES)
    return {
        'prediction': result_class,
        'confidence': float(confidence),
        'raw_scores': [float(p) for p in raw_predictions]
    }

def is_valid_open_eye_crop(crop_bgr):
    """
    Validation check to ensure ROI is actually an open eye and not closed skin, hair, or nostrils.
    Computes contrast variance and pupil/sclera feature presence.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return False

    aspect_ratio = float(w) / float(h)
    if aspect_ratio < 0.55 or aspect_ratio > 2.3:
        return False

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Measure edge detail/variance using Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 35.0: # Very smooth skin patch / closed eye
        return False

    # Check intensity range (pupil vs sclera contrast)
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    if (max_val - min_val) < 40: # Uniform texture, no eye structure
        return False

    return True

def detect_and_classify_eyes_in_image(img_bgr):
    """
    Detect eye regions using strict face-constrained spatial filtering and feature validation.
    Eliminates false detections on non-eye facial features or closed eyes.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    img_h, img_w = img_bgr.shape[:2]
    
    eyes_detected = []
    
    # 1. Primary Strategy: Detect faces first, then search upper 15%-55% face region for eyes
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    if len(faces) > 0:
        for (fx, fy, fw, fh) in faces:
            # Upper face ROI for eyes
            roi_y_start = fy + int(fh * 0.15)
            roi_y_end = fy + int(fh * 0.55)
            roi_x_start = fx + int(fw * 0.05)
            roi_x_end = fx + int(fw * 0.95)
            
            roi_gray = gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            if roi_gray.size == 0:
                continue

            min_eye_size = (int(fw * 0.12), int(fh * 0.12))
            max_eye_size = (int(fw * 0.42), int(fh * 0.42))

            face_eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=6,
                minSize=min_eye_size, maxSize=max_eye_size
            )
            if len(face_eyes) == 0:
                face_eyes = eye_tree_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=6,
                    minSize=min_eye_size, maxSize=max_eye_size
                )

            for (ex, ey, ew, eh) in face_eyes:
                abs_x = roi_x_start + ex
                abs_y = roi_y_start + ey
                eye_crop = img_bgr[abs_y:abs_y+eh, abs_x:abs_x+ew]
                if is_valid_open_eye_crop(eye_crop):
                    eyes_detected.append((abs_x, abs_y, ew, eh))

    # 2. Secondary Strategy: If no face detected (e.g. single eye macro image uploaded), detect eyes directly
    if len(eyes_detected) == 0:
        direct_eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(35, 35))
        for (ex, ey, ew, eh) in direct_eyes:
            eye_crop = img_bgr[ey:ey+eh, ex:ex+ew]
            if is_valid_open_eye_crop(eye_crop):
                eyes_detected.append((ex, ey, ew, eh))

    # 3. Direct Macro Fallback: If image itself is a cropped single eye image (no face structure round it)
    if len(eyes_detected) == 0:
        full_crop = img_bgr[int(img_h*0.05):int(img_h*0.95), int(img_w*0.05):int(img_w*0.95)]
        if is_valid_open_eye_crop(full_crop) and (0.6 <= float(img_w)/float(img_h) <= 2.2):
            # Treat central crop as the eye ROI
            eyes_detected.append((int(img_w*0.05), int(img_h*0.05), int(img_w*0.9), int(img_h*0.9)))

    # Process and classify valid detected eyes
    results = []
    for (x, y, w, h) in eyes_detected:
        eye_crop = img_bgr[y:y+h, x:x+w]
        if eye_crop.size == 0:
            continue
        res = classify_eye_crop(eye_crop)
        res['x'] = int(x)
        res['y'] = int(y)
        res['w'] = int(w)
        res['h'] = int(h)
        results.append(res)
        
    return results

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eye Health AI - Real-Time Detection & Diagnostics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --card-bg: rgba(21, 28, 44, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-cyan: #06b6d4;
            --accent-blue: #3b82f6;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        
        body {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(59, 130, 246, 0.12) 0%, transparent 40%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px 16px;
        }

        header {
            text-align: center;
            margin-bottom: 24px;
            max-width: 800px;
        }

        header h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.05rem;
        }

        /* Navigation Tabs */
        .tabs-container {
            display: flex;
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 50px;
            padding: 5px;
            margin-bottom: 28px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 12px 28px;
            font-size: 0.95rem;
            font-weight: 600;
            border-radius: 40px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .tab-btn.active {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            color: white;
            box-shadow: 0 4px 16px rgba(6, 182, 212, 0.35);
        }

        /* Main Workspace Container */
        .workspace {
            width: 100%;
            max-width: 1050px;
        }

        .tab-content {
            display: none;
            animation: fadeIn 0.4s ease forwards;
        }

        .tab-content.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Layout Grid */
        .panel-grid {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 24px;
        }

        @media (max-width: 860px) {
            .panel-grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 24px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }

        /* Video Container */
        .video-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 4 / 3;
            background: #020617;
            border-radius: 16px;
            overflow: hidden;
            border: 2px solid rgba(255, 255, 255, 0.05);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #webcamVideo {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        #overlayCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .camera-placeholder {
            position: absolute;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 12px;
            color: var(--text-muted);
            text-align: center;
            padding: 20px;
        }

        .camera-placeholder svg {
            width: 56px;
            height: 56px;
            stroke: var(--accent-cyan);
            opacity: 0.8;
        }

        /* Controls */
        .controls-bar {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            justify-content: center;
        }

        .btn-action {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 12px 24px;
            border-radius: 12px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn-action:hover {
            background: rgba(51, 65, 85, 0.9);
            transform: translateY(-2px);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            border: none;
            color: white;
            box-shadow: 0 4px 16px rgba(6, 182, 212, 0.3);
        }

        .btn-primary:hover {
            opacity: 0.95;
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.45);
        }

        .btn-danger {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            border: none;
            color: white;
        }

        /* Stats Side Panel */
        .panel-header {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 16px;
            color: #38bdf8;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .status-badge {
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .status-badge.active { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
        .status-badge.inactive { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.3); }

        .stat-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }

        .stat-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }

        .stat-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .bfs-box {
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 14px;
            margin-top: 16px;
        }

        .bfs-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: #a855f7;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .bfs-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            padding: 4px 0;
            border-bottom: 1px dashed rgba(255, 255, 255, 0.05);
        }

        .bfs-row:last-child { border-bottom: none; }

        /* Drag and Drop & Paste Zone */
        .dropzone {
            border: 2px dashed rgba(6, 182, 212, 0.4);
            background: rgba(15, 23, 42, 0.5);
            border-radius: 16px;
            padding: 48px 24px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(6, 182, 212, 0.08);
            transform: scale(1.01);
        }

        .dropzone svg {
            width: 48px;
            height: 48px;
            stroke: var(--accent-cyan);
            margin-bottom: 12px;
        }

        .dropzone-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .dropzone-desc {
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .paste-chip {
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38bdf8;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-top: 10px;
        }

        #fileInput { display: none; }

        .upload-preview-wrapper {
            position: relative;
            width: 100%;
            margin-top: 20px;
            border-radius: 16px;
            overflow: hidden;
            display: none;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }

        #uploadCanvas {
            width: 100%;
            display: block;
            border-radius: 16px;
        }

        /* Result Banners */
        .result-banner {
            padding: 16px 20px;
            border-radius: 12px;
            margin-top: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 700;
            font-size: 1.1rem;
        }

        .result-banner.Healthy {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #4ade80;
        }

        .result-banner.Disease {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
        }
    </style>
</head>
<body>

    <header>
        <h1>Eye Health AI & Real-Time Diagnostics</h1>
        <p>Automated Ocular Screening with Retrained CNN & Best First Search (BFS)</p>
    </header>

    <div class="tabs-container">
        <button class="tab-btn active" onclick="switchTab('cameraTab', this)">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
            Real-Time Camera Scan
        </button>
        <button class="tab-btn" onclick="switchTab('uploadTab', this)">
            <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            Manual Check (File & Clipboard)
        </button>
    </div>

    <div class="workspace">
        <!-- TAB 1: REAL-TIME CAMERA SCAN -->
        <div id="cameraTab" class="tab-content active">
            <div class="panel-grid">
                <div class="card">
                    <div class="video-wrapper">
                        <video id="webcamVideo" autoplay playsinline muted></video>
                        <canvas id="overlayCanvas"></canvas>
                        <div id="cameraPlaceholder" class="camera-placeholder">
                            <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z"/><circle cx="12" cy="13" r="3.25"/></svg>
                            <p>Camera inactive. Click "Start Camera" to grant access and initiate live eye health detection.</p>
                        </div>
                    </div>
                    <div class="controls-bar">
                        <button id="startCamBtn" class="btn-action btn-primary" onclick="startCamera()">Start Camera</button>
                        <button id="stopCamBtn" class="btn-action btn-danger" onclick="stopCamera()" style="display: none;">Stop Camera</button>
                    </div>
                </div>

                <div class="card">
                    <div class="panel-header">
                        Live Diagnostics
                        <span id="camStatusBadge" class="status-badge inactive">Offline</span>
                    </div>

                    <div class="stat-card">
                        <div class="stat-label">Open Eyes Detected</div>
                        <div id="eyesCountVal" class="stat-value">0</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-label">Primary Status</div>
                        <div id="primaryStatusVal" class="stat-value" style="color: #94a3b8;">--</div>
                    </div>

                    <div class="stat-card">
                        <div class="stat-label">Confidence Score</div>
                        <div id="confidenceScoreVal" class="stat-value">--</div>
                    </div>

                    <div class="bfs-box">
                        <div class="bfs-title">
                            <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                            Best First Search (BFS)
                        </div>
                        <div class="bfs-row"><span>Node [Healthy]:</span><span id="bfsHealthyVal">--</span></div>
                        <div class="bfs-row"><span>Node [Disease]:</span><span id="bfsDiseaseVal">--</span></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 2: MANUAL CHECK (FOLDER & CLIPBOARD PASTE) -->
        <div id="uploadTab" class="tab-content">
            <div class="panel-grid">
                <div class="card">
                    <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                        <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5"><path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z"/></svg>
                        <div class="dropzone-title">Click, Drag & Drop, or Paste (Ctrl+V) Eye Image Here</div>
                        <div class="dropzone-desc">Select an eye image from a folder or press Ctrl+V to paste directly from clipboard</div>
                        <span class="paste-chip">⚡ Paste (Ctrl+V) Enabled</span>
                        <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
                    </div>

                    <div id="uploadPreviewWrapper" class="upload-preview-wrapper">
                        <canvas id="uploadCanvas"></canvas>
                    </div>
                </div>

                <div class="card">
                    <div class="panel-header">
                        Manual Screening Report
                    </div>

                    <div id="uploadResultBanner" class="result-banner" style="display: none;">
                        <span id="uploadResultText">--</span>
                        <span id="uploadResultConf">--%</span>
                    </div>

                    <div class="stat-card" style="margin-top: 16px;">
                        <div class="stat-label">Open Eyes Analyzed</div>
                        <div id="uploadEyeCount" class="stat-value">--</div>
                    </div>

                    <div class="bfs-box">
                        <div class="bfs-title">
                            <svg width="14" height="14" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                            Best First Search Heuristic
                        </div>
                        <div class="bfs-row"><span>Node [Healthy]:</span><span id="uploadBfsHealthy">--</span></div>
                        <div class="bfs-row"><span>Node [Disease]:</span><span id="uploadBfsDisease">--</span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // --- TAB SWITCHER ---
        function switchTab(tabId, btn) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            if (btn) btn.classList.add('active');
        }

        // --- WEBCAM REAL-TIME SCANNER ---
        const video = document.getElementById('webcamVideo');
        const overlayCanvas = document.getElementById('overlayCanvas');
        const ctx = overlayCanvas.getContext('2d');
        let stream = null;
        let scanInterval = null;
        let isProcessingFrame = false;

        // Session tracking for unhealthy eyes timer (0-5s "Unhealthy", 5-10s "Disease detected")
        const unhealthyEyeTimestamps = new Map();

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
                });
                video.srcObject = stream;
                document.getElementById('cameraPlaceholder').style.display = 'none';
                document.getElementById('startCamBtn').style.display = 'none';
                document.getElementById('stopCamBtn').style.display = 'flex';
                
                const badge = document.getElementById('camStatusBadge');
                badge.innerText = 'Live Scanning';
                badge.className = 'status-badge active';

                video.onloadedmetadata = () => {
                    overlayCanvas.width = video.clientWidth;
                    overlayCanvas.height = video.clientHeight;
                    scanInterval = setInterval(processWebcamFrame, 150); // Scan frame every 150ms
                };
            } catch (err) {
                alert("Camera access failed or permission denied: " + err.message);
            }
        }

        function stopCamera() {
            if (scanInterval) clearInterval(scanInterval);
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
            video.srcObject = null;
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            
            document.getElementById('cameraPlaceholder').style.display = 'flex';
            document.getElementById('startCamBtn').style.display = 'flex';
            document.getElementById('stopCamBtn').style.display = 'none';
            
            const badge = document.getElementById('camStatusBadge');
            badge.innerText = 'Offline';
            badge.className = 'status-badge inactive';

            document.getElementById('eyesCountVal').innerText = '0';
            document.getElementById('primaryStatusVal').innerText = '--';
            document.getElementById('primaryStatusVal').style.color = '#94a3b8';
            document.getElementById('confidenceScoreVal').innerText = '--';
            document.getElementById('bfsHealthyVal').innerText = '--';
            document.getElementById('bfsDiseaseVal').innerText = '--';
        }

        async function processWebcamFrame() {
            if (isProcessingFrame || !video.videoWidth) return;
            isProcessingFrame = true;

            try {
                // Resize video frame to temp canvas for sending to backend
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 480;
                tempCanvas.height = 360;
                const tempCtx = tempCanvas.getContext('2d');
                tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
                
                const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.7);

                const response = await fetch('/api/predict_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl })
                });

                const data = await response.json();
                if (data.status === 'success') {
                    renderWebcamBoundingBoxes(data.eyes, tempCanvas.width, tempCanvas.height);
                }
            } catch (e) {
                console.error("Frame processing error:", e);
            } finally {
                isProcessingFrame = false;
            }
        }

        function renderWebcamBoundingBoxes(eyes, srcW, srcH) {
            overlayCanvas.width = video.clientWidth;
            overlayCanvas.height = video.clientHeight;
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

            const scaleX = overlayCanvas.width / srcW;
            const scaleY = overlayCanvas.height / srcH;

            const now = Date.now();
            document.getElementById('eyesCountVal').innerText = eyes.length;

            if (eyes.length > 0) {
                const primary = eyes[0];
                const statusVal = document.getElementById('primaryStatusVal');
                statusVal.innerText = primary.prediction;
                statusVal.style.color = primary.prediction === 'Healthy' ? '#22c55e' : '#ef4444';
                
                document.getElementById('confidenceScoreVal').innerText = (primary.confidence * 100).toFixed(1) + '%';
                document.getElementById('bfsHealthyVal').innerText = (primary.raw_scores[0] * 100).toFixed(2) + '%';
                document.getElementById('bfsDiseaseVal').innerText = (primary.raw_scores[1] * 100).toFixed(2) + '%';
            } else {
                document.getElementById('primaryStatusVal').innerText = 'No open eye';
                document.getElementById('primaryStatusVal').style.color = '#94a3b8';
                document.getElementById('confidenceScoreVal').innerText = '--';
                document.getElementById('bfsHealthyVal').innerText = '--';
                document.getElementById('bfsDiseaseVal').innerText = '--';
            }

            const currentEyeKeys = new Set();

            eyes.forEach((eye, idx) => {
                const x = eye.x * scaleX;
                const y = eye.y * scaleY;
                const w = eye.w * scaleX;
                const h = eye.h * scaleY;

                const eyeId = `eye_${idx}`;
                currentEyeKeys.add(eyeId);

                let isHealthy = (eye.prediction === 'Healthy');
                let boxColor = isHealthy ? '#22c55e' : '#ef4444';

                // --- 1. Draw Rectangular Box around Eye ---
                ctx.lineWidth = 3;
                ctx.strokeStyle = boxColor;
                ctx.strokeRect(x, y, w, h);

                // Corner Accents
                const cornerLen = Math.min(w, h) * 0.25;
                ctx.lineWidth = 4;
                // Top-Left
                ctx.beginPath(); ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y); ctx.stroke();
                // Top-Right
                ctx.beginPath(); ctx.moveTo(x + w - cornerLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cornerLen); ctx.stroke();
                // Bottom-Left
                ctx.beginPath(); ctx.moveTo(x, y + h - cornerLen); ctx.lineTo(x, y + h); ctx.lineTo(x + cornerLen, y + h); ctx.stroke();
                // Bottom-Right
                ctx.beginPath(); ctx.moveTo(x + w - cornerLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cornerLen); ctx.stroke();

                // --- 2. Determine Text Label Message ---
                let textMessage = "";
                if (isHealthy) {
                    textMessage = "Healthy";
                    unhealthyEyeTimestamps.delete(eyeId);
                } else {
                    if (!unhealthyEyeTimestamps.has(eyeId)) {
                        unhealthyEyeTimestamps.set(eyeId, now);
                    }
                    const startTime = unhealthyEyeTimestamps.get(eyeId);
                    const elapsedSec = (now - startTime) / 1000.0;

                    if (elapsedSec < 5.0) {
                        textMessage = "Unhealthy";
                    } else {
                        textMessage = "Disease detected";
                    }
                }

                // --- 3. Draw BOLD Text Message on Top Right Corner ---
                ctx.font = "bold 15px 'Plus Jakarta Sans', 'Segoe UI', sans-serif";
                const textMetrics = ctx.measureText(textMessage);
                const textWidth = textMetrics.width;
                const textHeight = 22;

                // Position badge at top-right corner of bounding box
                const badgeX = (x + w) - textWidth - 12;
                const badgeY = y - textHeight - 6 < 5 ? y + 6 : y - textHeight - 6;

                // Pill background
                ctx.fillStyle = boxColor;
                ctx.beginPath();
                ctx.roundRect(badgeX, badgeY, textWidth + 14, textHeight, 6);
                ctx.fill();

                // BOLD white text
                ctx.fillStyle = '#ffffff';
                ctx.textBaseline = 'middle';
                ctx.fillText(textMessage, badgeX + 7, badgeY + textHeight / 2);
            });
        }


        // --- TAB 2: MANUAL DROPBOX IMAGE UPLOAD & CLIPBOARD PASTE ---
        const dropzone = document.getElementById('dropzone');

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) processUploadedFile(files[0]);
        });

        // --- GLOBAL CLIPBOARD PASTE EVENT (CTRL+V) ---
        document.addEventListener('paste', (e) => {
            const clipboardData = e.clipboardData || window.clipboardData;
            if (!clipboardData) return;

            const items = clipboardData.items;
            if (!items) return;

            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    const blob = items[i].getAsFile();
                    if (blob) {
                        // Switch to upload tab automatically if currently on camera tab
                        const uploadTabBtn = document.querySelectorAll('.tab-btn')[1];
                        switchTab('uploadTab', uploadTabBtn);
                        processUploadedFile(blob);
                        break;
                    }
                }
            }
        });

        function handleFileSelect(e) {
            const files = e.target.files;
            if (files.length > 0) processUploadedFile(files[0]);
        }

        async function processUploadedFile(file) {
            const formData = new FormData();
            formData.append('file', file, file.name || 'clipboard_image.png');

            const previewWrapper = document.getElementById('uploadPreviewWrapper');
            const uploadCanvas = document.getElementById('uploadCanvas');
            const uCtx = uploadCanvas.getContext('2d');

            previewWrapper.style.display = 'block';

            const img = new Image();
            img.onload = async () => {
                uploadCanvas.width = img.width;
                uploadCanvas.height = img.height;
                uCtx.drawImage(img, 0, 0);

                try {
                    const response = await fetch('/api/predict_upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();

                    if (data.status === 'success') {
                        renderUploadResults(data, img, uploadCanvas, uCtx);
                    }
                } catch (err) {
                    alert("Error processing image: " + err.message);
                }
            };
            img.src = URL.createObjectURL(file);
        }

        function renderUploadResults(data, img, canvas, ctx) {
            ctx.drawImage(img, 0, 0);

            const eyes = data.eyes || [];
            document.getElementById('uploadEyeCount').innerText = eyes.length;

            const banner = document.getElementById('uploadResultBanner');
            const text = document.getElementById('uploadResultText');
            const conf = document.getElementById('uploadResultConf');

            if (eyes.length > 0) {
                const primary = eyes[0];

                banner.style.display = 'flex';
                banner.className = `result-banner ${primary.prediction}`;
                text.innerText = `Diagnosis: ${primary.prediction}`;
                conf.innerText = (primary.confidence * 100).toFixed(1) + '%';

                document.getElementById('uploadBfsHealthy').innerText = (primary.raw_scores[0] * 100).toFixed(2) + '%';
                document.getElementById('uploadBfsDisease').innerText = (primary.raw_scores[1] * 100).toFixed(2) + '%';
            } else {
                banner.style.display = 'flex';
                banner.className = 'result-banner Disease';
                text.innerText = 'No open eyes detected in image';
                conf.innerText = '--';

                document.getElementById('uploadBfsHealthy').innerText = '--';
                document.getElementById('uploadBfsDisease').innerText = '--';
            }

            eyes.forEach((eye) => {
                const x = eye.x;
                const y = eye.y;
                const w = eye.w;
                const h = eye.h;

                let isHealthy = (eye.prediction === 'Healthy');
                let boxColor = isHealthy ? '#22c55e' : '#ef4444';
                let textMessage = isHealthy ? "Healthy" : "Unhealthy";

                // Draw Rectangular Box
                ctx.lineWidth = Math.max(3, Math.round(canvas.width / 200));
                ctx.strokeStyle = boxColor;
                ctx.strokeRect(x, y, w, h);

                // Draw BOLD Label on Top Right Corner
                const fontSize = Math.max(14, Math.round(canvas.width / 40));
                ctx.font = `bold ${fontSize}px 'Plus Jakarta Sans', sans-serif`;
                const textMetrics = ctx.measureText(textMessage);
                const textWidth = textMetrics.width;
                const textHeight = fontSize * 1.4;

                const badgeX = (x + w) - textWidth - (fontSize * 0.8);
                const badgeY = y - textHeight - 6 < 5 ? y + 6 : y - textHeight - 6;

                ctx.fillStyle = boxColor;
                ctx.beginPath();
                ctx.roundRect(badgeX, badgeY, textWidth + fontSize, textHeight, 6);
                ctx.fill();

                ctx.fillStyle = '#ffffff';
                ctx.textBaseline = 'middle';
                ctx.fillText(textMessage, badgeX + (fontSize * 0.5), badgeY + textHeight / 2);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict_frame', methods=['POST'])
def predict_frame():
    """Endpoint for processing live camera base64 frame"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'Missing frame image'}), 400

        header, encoded = data['image'].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'}), 400

        eyes_results = detect_and_classify_eyes_in_image(img_bgr)
        return jsonify({
            'status': 'success',
            'eyes': eyes_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict_upload', methods=['POST'])
def predict_upload():
    """Endpoint for processing uploaded image from dropbox or clipboard paste"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({'status': 'error', 'message': 'Invalid image format'}), 400

        eyes_results = detect_and_classify_eyes_in_image(img_bgr)
        return jsonify({
            'status': 'success',
            'eyes': eyes_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("Starting Eye Disease Detection Web Interface on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
