"""
Prediction script cho Cat/Dog CNN
"""
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass


def preprocess_image(img_path, target_size=(150, 150)):
    """
    Preprocess ảnh để predict với xử lý tốt hơn
    
    Args:
        img_path: Đường dẫn đến ảnh
        target_size: Kích thước resize
    
    Returns:
        Preprocessed image array
    """
    # Load ảnh
    img = Image.open(img_path)
    
    # Convert sang RGB nếu cần
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize với chất lượng tốt hơn
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert sang array
    img_array = image.img_to_array(img)
    
    # Đảm bảo 3 channels
    if len(img_array.shape) == 3 and img_array.shape[2] != 3:
        img_array = img_array[:, :, :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.astype(np.float32)
    
    return img_array


def predict_image(model_path, img_path, class_names=['cat', 'dog']):
    """
    Predict một ảnh
    
    Args:
        model_path: Đường dẫn đến model file
        img_path: Đường dẫn đến ảnh cần predict
        class_names: Tên các classes
    
    Returns:
        Prediction result
    """
    # Load model
    print(f"Đang load model từ: {model_path}")
    model = load_model(model_path)
    
    # Preprocess image
    img_array = preprocess_image(img_path)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    predicted_class = class_names[predicted_class_idx]
    
    print(f"\nKết quả dự đoán:")
    print(f"  Class: {predicted_class}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"\nChi tiết:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {predictions[0][i]:.2%}")
    
    return predicted_class, confidence, predictions[0]


def predict_batch(model_path, img_dir, class_names=['cat', 'dog']):
    """
    Predict nhiều ảnh trong một thư mục
    
    Args:
        model_path: Đường dẫn đến model file
        img_dir: Thư mục chứa ảnh
        class_names: Tên các classes
    """
    model = load_model(model_path)
    
    # Lấy tất cả ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for file in os.listdir(img_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(img_dir, file))
    
    print(f"Tìm thấy {len(image_files)} ảnh")
    print("=" * 50)
    
    results = []
    for img_path in image_files:
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = class_names[predicted_class_idx]
        
        results.append({
            'file': os.path.basename(img_path),
            'class': predicted_class,
            'confidence': confidence
        })
        
        print(f"{os.path.basename(img_path)}: {predicted_class} ({confidence:.2%})")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict Cat/Dog from Image')
    parser.add_argument('--model', type=str, required=True,
                        help='Đường dẫn đến model file (.h5)')
    parser.add_argument('--image', type=str, default=None,
                        help='Đường dẫn đến ảnh cần predict')
    parser.add_argument('--dir', type=str, default=None,
                        help='Thư mục chứa nhiều ảnh cần predict')
    parser.add_argument('--classes', type=str, nargs='+', default=['cat', 'dog'],
                        help='Tên các classes')
    
    args = parser.parse_args()
    
    if args.image:
        predict_image(args.model, args.image, args.classes)
    elif args.dir:
        predict_batch(args.model, args.dir, args.classes)
    else:
        print("Vui lòng cung cấp --image hoặc --dir")

