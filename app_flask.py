"""
Flask Web App cho Cat/Dog CNN Classification
Giao diện đơn giản - chỉ upload và predict
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import webbrowser
import threading
import time

# Đã bỏ phần đánh giá model khỏi web

# Note: Encoding handling removed to avoid I/O errors on Windows
# If you encounter encoding issues, set PYTHONIOENCODING=utf-8 environment variable

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model - chỉ dùng 1 model
MODEL_PATH = 'models/cat_dog_model_v2_final.h5'
model = None

def load_model_once():
    """Load model khi start app"""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = load_model(MODEL_PATH)
                print(f"Model loaded from {MODEL_PATH}")
                return model
            except Exception as e:
                print(f"Error loading {MODEL_PATH}: {e}")
                print(f"Please ensure the model file exists at {MODEL_PATH}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            print("Please train a model first or ensure the model file exists.")
    return model

def allowed_file(filename):
    """Kiểm tra file extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img):
    """Preprocess ảnh để predict"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/')
def index():
    """Trang chủ"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để predict tối đa 6 ảnh"""
    try:
        if not request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        files = []
        if 'files' in request.files:
            files = request.files.getlist('files')
        elif 'file' in request.files:
            files = request.files.getlist('file')

        # Lọc bỏ file trống
        files = [f for f in files if f and f.filename]

        if not files:
            return jsonify({'error': 'No file selected'}), 400

        if len(files) > 6:
            return jsonify({'error': 'Chỉ hỗ trợ tối đa 6 ảnh mỗi lần'}), 400

        model = load_model_once()
        if model is None:
            return jsonify({'error': 'Model not found. Please train a model first.'}), 500

        batch_data = []
        file_names = []
        for idx, file in enumerate(files, start=1):
            if not allowed_file(file.filename):
                return jsonify({'error': f'Invalid file type for {file.filename}. Only PNG, JPG, JPEG allowed'}), 400

            img = Image.open(file.stream)
            img_array = preprocess_image(img)
            batch_data.append(img_array)
            safe_name = secure_filename(file.filename) or f'image_{idx}.png'
            file_names.append(safe_name)

        batch_input = np.vstack(batch_data)
        predictions = model.predict(batch_input, verbose=0)
        class_names = ['cat', 'dog']

        results = []
        for name, probs in zip(file_names, predictions):
            predicted_class_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_class_idx])
            results.append({
                'filename': name,
                'class': class_names[predicted_class_idx],
                'confidence': round(confidence * 100, 2),
                'details': {
                    'cat': round(float(probs[0]) * 100, 2),
                    'dog': round(float(probs[1]) * 100, 2)
                }
            })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/<path:filename>')
def serve_model_files(filename):
    """Serve files from models directory"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    return send_from_directory(models_dir, filename)

def open_default_browser(target_url):
    """Mở URL trên trình duyệt mặc định ngoài Cursor."""
    try:
        if os.name == 'nt':
            os.startfile(target_url)  # type: ignore[attr-defined]
        else:
            webbrowser.open_new_tab(target_url)
        print(f"✓ Đã mở trình duyệt mặc định tại: {target_url}\n")
    except Exception as exc:
        print(f"Không thể tự mở trình duyệt ({exc}). Vui lòng truy cập: {target_url}\n")


if __name__ == '__main__':
    load_model_once()
    url = "http://127.0.0.1:5000"
    
    print("=" * 60)
    print("  CAT/DOG CNN - WEB APPLICATION")
    print("=" * 60)
    print(f"\nModel loaded: {model is not None}")
    if model is not None:
        print(f"Model path: {MODEL_PATH}")
    print(f"\nURL: {url}")
    print(f"URL: http://localhost:5000")
    print("\nĐang khởi động server...")
    print("Trình duyệt sẽ tự mở sau vài giây, nếu không hãy copy URL ở trên.")
    print("Nhấn CTRL+C để dừng server")
    print("=" * 60 + "\n")
    
    browser_thread = threading.Thread(target=lambda: (time.sleep(2), open_default_browser(url)))
    browser_thread.daemon = True
    browser_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
