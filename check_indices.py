from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def check_mapping():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        'dataset',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        classes=['healthy', 'disease'],
        subset='training'
    )
    print("Class Indices:", train_generator.class_indices)
    
if __name__ == "__main__":
    check_mapping()
