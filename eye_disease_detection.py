import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os

# --- 1. CONFIGURATION ---
IMG_SIZE = (128, 128)  # Increased size for better detail
BATCH_SIZE = 32
CLASSES = ['Healthy', 'Disease'] # Define our classes

# --- 2. THE MODEL (Improved CNN) ---
def create_model():
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flattening
        Flatten(),
        
        # Dense Layers
        Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5), # Reduce overfitting
        Dense(2, activation='softmax') 
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 3. BEST FIRST SEARCH ALGORITHM ---
def best_first_search(predictions, class_labels):
    """
    Implements Best First Search to find the most likely class.
    
    Concept:
    - Treat each class probability as a 'node'.
    - The 'heuristic' is the probability score itself.
    - We want to find the node with the highest score (Best).
    """
    best_score = -1.0
    best_class = None
    
    print("\n--- Starting Best First Search ---")
    
    # Iterate through all prediction nodes (classes)
    for i, score in enumerate(predictions):
        label = class_labels[i]
        print(f"Node: {label}, Heuristic (Score): {score:.4f}")
        
        # Greedy Step: If this node is better than the current best, take it.
        if score > best_score:
            best_score = score
            best_class = label
            print(f"  -> New Best Node Found: {label}")
            
    print(f"--- Search Complete. Winner: {best_class} ---\n")
    return best_class

# --- 4. TRAINING FUNCTION (Simplified) ---
def train_system():
    # Check if dataset exists (just a safeguard for this script)
    if not os.path.exists('dataset'):
        print("Dataset folder not found. Please create 'dataset/healthy' and 'dataset/disease'.")
        return None

    # Data Generator: Loads and normalizes images on the fly
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Load Training Data
    train_generator = datagen.flow_from_directory(
        'dataset',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['healthy', 'disease'], # Explicitly define order to match CLASSES ['Healthy', 'Disease']
        subset='training'
    )
    
    model = create_model()
    
    print("Starting Training (Improved Model)...")
    # Train for more epochs
    model.fit(train_generator, epochs=10) 
    
    # Save the trained model
    model.save('eye_disease_model.h5')
    print("Model saved as 'eye_disease_model.h5'")
    return model

# --- 5. DETECTION FUNCTION ---
def detect_disease(image_path):
    # Load model (if valid model file exists, else create a new untrained one for demo)
    try:
        model = tf.keras.models.load_model('eye_disease_model.h5')
    except:
        print("Model not found, initializing new model for demo...")
        model = create_model()

    # Preprocessing
    try:
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array /= 255.0 # Normalize
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Get Predictions (Probabilities)
    # Returns something like [[0.1, 0.9]]
    raw_predictions = model.predict(img_array)[0] 
    
    # Use CUSTOM BEST FIRST SEARCH to interpret results
    result = best_first_search(raw_predictions, CLASSES)
    
    print(f"Final Diagnosis for {image_path}: {result}")
    return result

# --- 6. MAIN EXECUTION ---
if __name__ == "__main__":
    # Create dummy folders for user understanding
    os.makedirs("dataset/healthy", exist_ok=True)
    os.makedirs("dataset/disease", exist_ok=True)
    
    print("System Ready.")
    print("Retraining improved model...")
    # train_system() 
    
    print("\n--- Interactive Eye Disease Detection ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("\nEnter image path (e.g., dataset/disease/Mild_DR.png): ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting system. Goodbye!")
            break
            
        if not user_input:
            continue
            
        if not os.path.exists(user_input):
            print(f"Error: File '{user_input}' does not exist. Please check the path.")
            continue
            
        detect_disease(user_input)
