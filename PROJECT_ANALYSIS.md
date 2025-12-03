# Phân Tích Cấu Trúc Dự Án Cat/Dog CNN

Tổng quan toàn bộ thư mục `cat_dog_cnn_project`, mô tả chức năng từng thành phần và các kỹ thuật chính được áp dụng kèm đoạn code cụ thể.

## Thư mục & File chính

### `train.py`
- **Chức năng**: Script huấn luyện CNN phân loại chó/mèo.
- **Kỹ thuật & Code cụ thể**:

#### 1. Data Augmentation với ImageDataGenerator
**Vị trí**: Dòng 37-46
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```
- **Kỹ thuật**: Augmentation tự động (xoay ±40°, dịch 20%, shear 20%, zoom 20%, lật ngang) để tăng dữ liệu và giảm overfitting.

#### 2. Validation Generator (chỉ rescale)
**Vị trí**: Dòng 49
```python
validation_datagen = ImageDataGenerator(rescale=1./255)
```
- **Kỹ thuật**: Validation không augmentation, chỉ chuẩn hóa pixel về [0,1].

#### 3. Flow từ Directory
**Vị trí**: Dòng 52-64
```python
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)
```
- **Kỹ thuật**: Tự động load ảnh từ cấu trúc thư mục, tạo nhãn categorical (one-hot).

#### 4. ModelCheckpoint Callback
**Vị trí**: Dòng 99-104
```python
ModelCheckpoint(
    os.path.join(models_dir, 'cat_dog_model.h5'),
    monitor='val_accuracy' if val_gen.samples > 0 else 'accuracy',
    save_best_only=True,
    verbose=1
)
```
- **Kỹ thuật**: Tự động lưu model tốt nhất dựa trên validation accuracy.

#### 5. EarlyStopping Callback
**Vị trí**: Dòng 105-110
```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
```
- **Kỹ thuật**: Dừng sớm nếu validation loss không cải thiện sau 10 epochs, khôi phục weights tốt nhất.

#### 6. ReduceLROnPlateau Callback
**Vị trí**: Dòng 111-117
```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
```
- **Kỹ thuật**: Giảm learning rate 50% khi loss không cải thiện sau 5 epochs.

#### 7. Tính steps_per_epoch an toàn
**Vị trí**: Dòng 132
```python
steps_per_epoch = max(1, train_gen.samples // batch_size) if train_gen.samples > 0 else 1
```
- **Kỹ thuật**: Đảm bảo steps_per_epoch >= 1 để tránh lỗi khi dữ liệu ít.

#### 8. Vẽ đồ thị Training History
**Vị trí**: Dòng 155-185
```python
def plot_training_history(history, models_dir='models'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    # ... tương tự cho loss
    plt.savefig(history_path, dpi=150)
```
- **Kỹ thuật**: Matplotlib vẽ 2 subplot (accuracy & loss) và lưu PNG.

#### 9. Argument Parser
**Vị trí**: Dòng 191-197
```python
parser = argparse.ArgumentParser(description='Train Cat/Dog CNN Model')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
```
- **Kỹ thuật**: CLI arguments để tùy chỉnh tham số training.

#### 10. Fix Encoding cho Windows
**Vị trí**: Dòng 13-16
```python
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```
- **Kỹ thuật**: Xử lý encoding UTF-8 trên Windows để in tiếng Việt không lỗi.

---

### `model.py`
- **Chức năng**: Định nghĩa kiến trúc CNN.
- **Kỹ thuật & Code cụ thể**:

#### 1. Sequential Model với Conv2D + MaxPooling
**Vị trí**: Dòng 19-40
```python
model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```
- **Kỹ thuật**: 
  - 4 khối Conv2D (32→64→128→128 filters) với ReLU
  - MaxPooling 2x2 sau mỗi Conv để giảm kích thước
  - Dropout 0.5 để giảm overfitting
  - Dense 512 + Softmax output

#### 2. Compile Model
**Vị trí**: Dòng 42-46
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```
- **Kỹ thuật**: Adam optimizer, categorical crossentropy loss cho multi-class, metric accuracy.

---

### `app_flask.py`
- **Chức năng**: Ứng dụng Flask phục vụ giao diện web dự đoán.
- **Kỹ thuật & Code cụ thể**:

#### 1. Flask App Configuration
**Vị trí**: Dòng 22-25
```python
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
```
- **Kỹ thuật**: Cấu hình giới hạn upload 16MB, định dạng ảnh cho phép.

#### 2. Load Model một lần (Singleton)
**Vị trí**: Dòng 34-49
```python
def load_model_once():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            return model
    return model
```
- **Kỹ thuật**: Load model một lần khi khởi động, tái sử dụng cho mọi request.

#### 3. Kiểm tra File Extension
**Vị trí**: Dòng 51-53
```python
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
```
- **Kỹ thuật**: Validate extension file upload.

#### 4. Preprocess Ảnh
**Vị trí**: Dòng 55-63
```python
def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
```
- **Kỹ thuật**: 
  - Chuyển sang RGB nếu cần
  - Resize 150x150 (khớp input model)
  - Chuẩn hóa pixel về [0,1]
  - Thêm batch dimension

#### 5. Route Index
**Vị trí**: Dòng 65-68
```python
@app.route('/')
def index():
    return render_template('index.html')
```
- **Kỹ thuật**: Flask route phục vụ template HTML.

#### 6. Route Predict với Batch Processing
**Vị trí**: Dòng 70-129
```python
@app.route('/predict', methods=['POST'])
def predict():
    files = []
    if 'files' in request.files:
        files = request.files.getlist('files')
    elif 'file' in request.files:
        files = request.files.getlist('file')
    
    files = [f for f in files if f and f.filename]
    if len(files) > 6:
        return jsonify({'error': 'Chỉ hỗ trợ tối đa 6 ảnh mỗi lần'}), 400
    
    batch_data = []
    for file in files:
        img = Image.open(file.stream)
        img_array = preprocess_image(img)
        batch_data.append(img_array)
    
    batch_input = np.vstack(batch_data)
    predictions = model.predict(batch_input, verbose=0)
```
- **Kỹ thuật**: 
  - Nhận tối đa 6 ảnh
  - Xử lý batch bằng `np.vstack` để predict cùng lúc
  - Trả JSON với kết quả từng ảnh

#### 7. Secure Filename
**Vị trí**: Dòng 105
```python
safe_name = secure_filename(file.filename) or f'image_{idx}.png'
```
- **Kỹ thuật**: `werkzeug.utils.secure_filename` để bảo vệ khỏi path traversal.

#### 8. Mở Browser tự động (Windows)
**Vị trí**: Dòng 137-146
```python
def open_default_browser(target_url):
    if os.name == 'nt':
        os.startfile(target_url)
    else:
        webbrowser.open_new_tab(target_url)
```
- **Kỹ thuật**: Dùng `os.startfile` trên Windows để mở browser mặc định ngoài Cursor.

#### 9. Thread mở Browser
**Vị trí**: Dòng 166-168
```python
browser_thread = threading.Thread(target=lambda: (time.sleep(2), open_default_browser(url)))
browser_thread.daemon = True
browser_thread.start()
```
- **Kỹ thuật**: Thread daemon đợi 2 giây rồi mở browser, không block server.

---

### `templates/index.html`
- **Chức năng**: Frontend UI single-page.
- **Kỹ thuật & Code cụ thể**:

#### 1. CSS Grid Layout
**Vị trí**: Dòng 93-97, 127-131
```css
.previews-grid {
    display: grid;
    gap: 15px;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
}
.results-grid {
    display: grid;
    gap: 20px;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}
```
- **Kỹ thuật**: CSS Grid responsive tự động điều chỉnh số cột theo màn hình.

#### 2. File Input Multiple
**Vị trí**: Dòng 250
```html
<input type="file" id="fileInput" accept="image/png,image/jpeg,image/jpg" multiple>
```
- **Kỹ thuật**: HTML5 `multiple` để chọn nhiều file cùng lúc.

#### 3. FileReader để Preview Ảnh
**Vị trí**: Dòng 318-323
```javascript
const reader = new FileReader();
reader.onload = (event) => {
    previewCard.querySelector('.preview-image').src = event.target.result;
};
reader.readAsDataURL(file);
```
- **Kỹ thuật**: `FileReader.readAsDataURL` để hiển thị preview ảnh ngay khi chọn (không cần upload).

#### 4. Fetch API với FormData
**Vị trí**: Dòng 334-341
```javascript
const formData = new FormData();
selectedFiles.forEach(file => formData.append('files', file));

const response = await fetch('/predict', {
    method: 'POST',
    body: formData
});
```
- **Kỹ thuật**: `FormData` + `fetch` để gửi multipart/form-data (upload file).

#### 5. Dynamic DOM Creation
**Vị trí**: Dòng 359-383
```javascript
const card = document.createElement('div');
card.className = 'result-card';
card.innerHTML = `...`;
resultsContainer.appendChild(card);
```
- **Kỹ thuật**: Tạo element động bằng JavaScript để render kết quả từng ảnh.

#### 6. Array Slice để giới hạn 6 ảnh
**Vị trí**: Dòng 287
```javascript
selectedFiles = Array.from(e.target.files).slice(0, 6);
```
- **Kỹ thuật**: `Array.slice(0, 6)` để chỉ lấy 6 ảnh đầu tiên nếu chọn nhiều hơn.

#### 7. CSS Animation Spinner
**Vị trí**: Dòng 218-231
```css
.spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
```
- **Kỹ thuật**: CSS animation để hiển thị loading spinner.

---

### `evaluate_model.py`
- **Chức năng**: Đánh giá model trên tập validation, in bảng `classification_report`.
- **Kỹ thuật & Code cụ thể**:

#### 1. ImageDataGenerator với shuffle=False
**Vị trí**: Dòng 24-31
```python
datagen = ImageDataGenerator(rescale=1.0 / 255)
generator = datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,  # Quan trọng!
)
```
- **Kỹ thuật**: `shuffle=False` để giữ thứ tự nhãn, mapping đúng `generator.classes` với predictions.

#### 2. Batch Prediction
**Vị trí**: Dòng 56
```python
predictions = model.predict(generator, verbose=1)
```
- **Kỹ thuật**: Predict toàn bộ validation set qua generator, trả về probabilities.

#### 3. Argmax để lấy Class dự đoán
**Vị trí**: Dòng 58
```python
y_pred = np.argmax(predictions, axis=1)
```
- **Kỹ thuật**: `np.argmax` lấy index class có probability cao nhất.

#### 4. Mapping Class Indices
**Vị trí**: Dòng 60-62
```python
class_indices = generator.class_indices
idx_to_class = {idx: name for name, idx in class_indices.items()}
class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
```
- **Kỹ thuật**: Đảo ngược dictionary `class_indices` để map index → tên class.

#### 5. Classification Report (sklearn)
**Vị trí**: Dòng 64-70
```python
report_text = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0,
)
print(report_text)
```
- **Kỹ thuật**: `sklearn.metrics.classification_report` để in bảng Precision/Recall/F1 giống Google Colab.

#### 6. Argument Parser với nargs
**Vị trí**: Dòng 82
```python
parser.add_argument('--img_size', type=int, nargs=2, default=(150, 150), 
                    help='Kích thước resize ảnh, ví dụ: --img_size 150 150')
```
- **Kỹ thuật**: `nargs=2` để nhận 2 giá trị cho img_size (width, height).

---

### `requirements.txt`
- **Chức năng**: Liệt kê thư viện cần thiết.
- **Nội dung**:
```
tensorflow>=2.15.0
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=10.0.0
scikit-learn>=1.3.0
flask>=2.3.0
werkzeug>=2.3.0
```

### `README.md`
- **Chức năng**: Tài liệu hướng dẫn sử dụng dự án.

### `Bao_Cao_Training.html`
- **Chức năng**: Báo cáo training (HTML) – lưu kết quả/ảnh huấn luyện.

## Thư mục dữ liệu & kết quả

### `data/`
- **Chức năng**: Chứa dữ liệu huấn luyện & validation.
- **Cấu trúc**:
  ```
  data/
    train/
      cats/
      dogs/
    validation/
      cats/
      dogs/
  ```
- **Kỹ thuật**: Phù hợp chuẩn `flow_from_directory` của Keras, tự động tạo nhãn từ tên thư mục.

### `models/`
- **Chức năng**: Lưu checkpoint/kết quả sau khi train và đánh giá.
- **Tệp đáng chú ý**:
  - `cat_dog_model_v2_final.h5`: model đã huấn luyện (được load trong `app_flask.py` dòng 31, 40).
  - `confusion_matrix.png`, `training_history_v2.png`: ảnh thống kê.

### `uploads/`
- **Chức năng**: Thư mục tạm lưu file upload (nếu cần) khi sử dụng web app.
- **Kỹ thuật**: Được Flask cấu hình (`app_flask.py` dòng 24, 28) và đảm bảo tồn tại khi khởi động.

## Quy trình chính

1. **Huấn luyện**: 
   ```bash
   python train.py --data_dir data --epochs 50 --batch_size 32
   ```
   - Tạo model, train với callbacks, lưu checkpoint và đồ thị.

2. **Đánh giá**: 
   ```bash
   python evaluate_model.py --model_path models/cat_dog_model_v2_final.h5
   ```
   - Load model, predict validation set, in bảng classification report.

3. **Triển khai web**: 
   ```bash
   python app_flask.py
   ```
   - Khởi động Flask server, tự mở browser ngoài Cursor, upload & xem dự đoán nhiều ảnh.

## Kỹ thuật tổng thể

- **Deep Learning**: CNN với TensorFlow/Keras, augmentation mạnh để giảm overfitting.
- **Model management**: Callback (checkpoint, early stopping, giảm LR), lưu biểu đồ training.
- **Web inference**: Flask API + frontend thuần JS, xử lý batch, preview ảnh client-side.
- **Đánh giá**: Sử dụng scikit-learn để tính precision/recall/F1 và in bảng chuẩn.

File này nhằm giúp nắm ngay cấu trúc dự án, thuận tiện bảo trì hoặc bàn giao.
