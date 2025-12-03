"""
CNN Model for Cat/Dog Classification
"""
from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_shape=(150, 150, 3), num_classes=2):
    """
    Tạo CNN model cho phân loại chó/mèo
    
    Args:
        input_shape: Kích thước input image (height, width, channels)
        num_classes: Số lượng classes (2: cat, dog)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        # Flatten và Dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model