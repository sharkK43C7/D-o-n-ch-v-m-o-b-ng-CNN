"""
Ensemble prediction để cải thiện accuracy
Sử dụng nhiều model hoặc nhiều lần predict với augmentation
"""
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf


def predict_with_augmentation(model, img, num_augments=5):
    """
    Predict với data augmentation để cải thiện accuracy
    
    Args:
        model: Keras model
        img: PIL Image
        num_augments: Số lần augment
    
    Returns:
        Average prediction
    """
    predictions = []
    
    # Convert sang RGB
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Original prediction
    img_resized = img.resize((150, 150), Image.Resampling.LANCZOS)
    img_array = image.img_to_array(img_resized)
    if img_array.shape[2] != 3:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    img_array = img_array.astype(np.float32)
    pred = model.predict(img_array, verbose=0)
    predictions.append(pred[0])
    
    # Augmented predictions
    for i in range(num_augments - 1):
        # Random rotation
        angle = np.random.uniform(-10, 10)
        img_rotated = img.rotate(angle, fillcolor=(255, 255, 255))
        
        # Random flip
        if np.random.random() > 0.5:
            img_rotated = img_rotated.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Resize và predict
        img_resized = img_rotated.resize((150, 150), Image.Resampling.LANCZOS)
        img_array = image.img_to_array(img_resized)
        if img_array.shape[2] != 3:
            img_array = img_array[:, :, :3]
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        img_array = img_array.astype(np.float32)
        pred = model.predict(img_array, verbose=0)
        predictions.append(pred[0])
    
    # Average predictions
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction


def predict_ensemble(models, img, class_names=['cat', 'dog']):
    """
    Ensemble prediction với nhiều models
    
    Args:
        models: List of models
        img: PIL Image
        class_names: Class names
    
    Returns:
        Ensemble prediction
    """
    all_predictions = []
    
    for model in models:
        pred = predict_with_augmentation(model, img, num_augments=3)
        all_predictions.append(pred)
    
    # Average across models
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    predicted_class_idx = np.argmax(ensemble_pred)
    confidence = ensemble_pred[predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, ensemble_pred


