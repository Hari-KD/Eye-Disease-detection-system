import numpy as np
from PIL import Image
import os
import random

def create_dummy_data():
    classes = ['healthy', 'disease']
    for label in classes:
        os.makedirs(f'dataset/{label}', exist_ok=True)
        print(f"Generating images for {label}...")
        for i in range(10):  # Generate 10 images per class
            # Create random noise image
            if label == 'healthy':
                # Green-ish tint for healthy
                data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                data[:, :, 1] = 200 # Make green channel dominant
            else:
                # Red-ish tint for disease
                data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                data[:, :, 0] = 200 # Make red channel dominant
                
            img = Image.fromarray(data)
            img.save(f'dataset/{label}/sample_{i}.jpg')
    print("Dummy data creation complete.")

if __name__ == "__main__":
    create_dummy_data()
