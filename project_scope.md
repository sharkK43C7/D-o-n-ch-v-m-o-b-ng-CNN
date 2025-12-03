CHƯƠNG 3: TRIỂN KHAI CHI TIẾT ỨNG DỤNG

3.1. Lựa chọn và thiết kế kiến trúc mô hình (CNN)
Sau khi đã phân tích và tiền xử lý dữ liệu ở Chương 2, bước tiếp theo là xây dựng một mô hình học máy có khả năng học được các đặc trưng (features) từ dữ liệu ảnh 150x150 pixel để phân loại chúng vào 2 lớp (Cat và Dog).

Lựa chọn kiến trúc: Đối với các bài toán thị giác máy tính và nhận dạng hình ảnh, đặc biệt là phân loại ảnh động vật, Mạng nơ-ron Tích chập (Convolutional Neural Network - CNN) đã được chứng minh là kiến trúc hiệu quả và mạnh mẽ nhất. Không giống như các mạng nơ-ron truyền thống, CNN có khả năng tự động học và trích xuất các đặc trưng không gian (spatial features) từ ảnh, chẳng hạn như cạnh, góc, hình dạng mắt, tai, mũi, và các đặc điểm đặc trưng của chó và mèo thông qua các bộ lọc (convolutional filters).

Thiết kế kiến trúc mô hình: Dựa trên các thực tiễn tốt nhất cho bài toán phân loại ảnh nhị phân (binary classification), nhóm đã thiết kế một kiến trúc CNN tuần tự (Sequential) sử dụng Keras. Mô hình này bao gồm các khối (block) tích chập và các lớp kết nối đầy đủ (fully-connected) để thực hiện phân loại:
•	Khối Tích chập 1: Bao gồm lớp Conv2D với 32 bộ lọc (filters) kích thước 3x3 để học các đặc trưng cơ bản như cạnh và đường nét, theo sau là một lớp MaxPooling2D (2x2) để giảm chiều dữ liệu và tăng tính bất biến.
•	Khối Tích chập 2: Bao gồm lớp Conv2D với 64 bộ lọc để học các đặc trưng phức tạp hơn, theo sau là MaxPooling2D (2x2).
•	Khối Tích chập 3: Bao gồm lớp Conv2D với 128 bộ lọc để học các đặc trưng chi tiết và phức tạp, theo sau là MaxPooling2D (2x2).
•	Khối Tích chập 4: Tiếp tục với lớp Conv2D 128 bộ lọc để tinh chỉnh các đặc trưng, theo sau là MaxPooling2D (2x2).
•	Khối Phân loại: Lớp Flatten được sử dụng để "làm phẳng" dữ liệu 2D từ khối tích chập thành một vector 1D. Vector này sau đó được đưa qua một lớp Dropout (50%) để giảm thiểu hiện tượng học vẹt (overfitting), tiếp theo là lớp Dense (512 nơ-ron) với hàm kích hoạt ReLU và cuối cùng là lớp Dense (2 nơ-ron) với hàm kích hoạt softmax để đưa ra xác suất phân loại cho 2 lớp (Cat và Dog).
•	Tối ưu hóa: Mô hình sử dụng Adam optimizer với hàm mất mát categorical_crossentropy vì nhãn (labels) của chúng ta được mã hóa dưới dạng one-hot encoding.

Xây dựng kiến trúc mô hình (CNN)
# Define the CNN architecture
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(150, 150, 3)),
    
    # Convolutional Block 1
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Convolutional Block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Convolutional Block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Convolutional Block 4
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Classification Block
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

3.2. Huấn luyện và tinh chỉnh mô hình (Sử dụng cat_dog_model_v2_final.h5)
Tăng cường dữ liệu (Data Augmentation)
Để giúp mô hình có khả năng tổng quát hóa tốt hơn và giảm thiểu học vẹt (overfitting), đặc biệt quan trọng với dữ liệu ảnh động vật có nhiều biến thể về góc chụp, ánh sáng, và vị trí, chúng ta sử dụng kỹ thuật Tăng cường dữ liệu. Kỹ thuật này sẽ tạo ra các phiên bản mới, hơi khác biệt của ảnh huấn luyện trong mỗi kỷ nguyên (epoch) bằng cách áp dụng các phép biến đổi ngẫu nhiên như xoay, dịch chuyển, zoom, cắt xén, và lật ngang ảnh.

Tăng cường dữ liệu
# DATA AUGMENTATION cho training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,          # Xoay ảnh trong khoảng ±40 độ
    width_shift_range=0.2,      # Dịch chuyển ngang 20%
    height_shift_range=0.2,     # Dịch chuyển dọc 20%
    shear_range=0.2,            # Biến dạng cắt 20%
    zoom_range=0.2,             # Zoom 20%
    horizontal_flip=True,       # Lật ngang ảnh
    fill_mode='nearest'         # Điền pixel bằng phương pháp nearest
)

# Chỉ rescale cho validation (không augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Tạo generators từ thư mục
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

Biên dịch mô hình (Compiling)
Trước khi huấn luyện, mô hình cần được biên dịch với các thành phần sau:
•	Hàm tối ưu (Optimizer): Chúng ta sử dụng adam, một thuật toán tối ưu hiệu quả và phổ biến, giúp điều chỉnh tốc độ học (learning rate) một cách thích ứng, đặc biệt phù hợp với bài toán phân loại ảnh.
•	Hàm mất mát (Loss Function): categorical_crossentropy được chọn vì đây là bài toán phân loại đa lớp (2 lớp) và các nhãn (labels) của chúng ta được mã hóa dưới dạng one-hot vector (ví dụ: [1, 0] cho Cat, [0, 1] cho Dog).
•	Chỉ số đánh giá (Metrics): Chúng ta theo dõi chỉ số accuracy (độ chính xác) trong suốt quá trình huấn luyện để đánh giá hiệu suất mô hình.

Biên dịch mô hình
# Compile the CNN model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

Sử dụng Callbacks để tinh chỉnh
Callbacks là các hàm được gọi tại các thời điểm khác nhau trong quá trình huấn luyện, cho phép chúng ta tự động hóa việc tinh chỉnh mô hình:
•	ModelCheckpoint: Đây là callback quan trọng nhất. Nó sẽ theo dõi val_accuracy (accuracy trên tập validation) và lưu lại phiên bản tốt nhất của mô hình vào tệp cat_dog_model.h5. Đây chính là tệp mô hình mà sau này chúng ta sử dụng trong ứng dụng web Flask.
•	EarlyStopping: Theo dõi val_loss (loss trên tập validation). Nếu val_loss không cải thiện sau 10 epochs (patience=10), quá trình huấn luyện sẽ tự động dừng lại và khôi phục weights tốt nhất (restore_best_weights=True) để tránh overfitting.
•	ReduceLROnPlateau: Nếu val_loss không cải thiện sau 5 epochs (patience=5), tốc độ học sẽ được giảm đi một nửa (factor=0.5) với learning rate tối thiểu là 1e-7 để giúp mô hình "tìm đường" tốt hơn trong quá trình tối ưu.

callbacks = [
    ModelCheckpoint(
        'models/cat_dog_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

Huấn luyện mô hình (Training)
Quá trình huấn luyện sử dụng data generator để load ảnh từ thư mục một cách tự động, áp dụng augmentation cho tập training và chỉ rescale cho tập validation. Mô hình được huấn luyện với số epochs mặc định là 50, batch_size là 32. Sau khi hoàn thành, mô hình tốt nhất được lưu và đồ thị training history (accuracy và loss) được vẽ và lưu lại để phân tích.

steps_per_epoch = max(1, train_gen.samples // batch_size)
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=val_gen,
    validation_steps=val_gen.samples // batch_size,
    callbacks=callbacks,
    verbose=1
)

# Lưu model cuối cùng
model.save('models/cat_dog_model_final.h5')

# Vẽ và lưu đồ thị training history
plot_training_history(history, 'models')

3.3. Biểu diễn và đánh giá kết quả mô hình

3.3.1. Các chỉ số đánh giá (Evaluation Metrics: Accuracy, Precision, Recall, F1-Score)
Để đánh giá chất lượng mô hình một cách toàn diện, chúng ta sử dụng các chỉ số đánh giá chuẩn trong học máy:
•	Accuracy (Độ chính xác): Tỷ lệ số dự đoán đúng trên tổng số dự đoán. Đây là chỉ số tổng quát nhất, nhưng có thể không phản ánh đúng hiệu suất khi dữ liệu mất cân bằng.
•	Precision (Độ chính xác dự đoán): Tỷ lệ số dự đoán dương tính thực sự trong tất cả các dự đoán dương tính. Precision = TP / (TP + FP), trong đó TP là True Positive và FP là False Positive.
•	Recall (Độ nhạy): Tỷ lệ số dự đoán dương tính thực sự trong tất cả các trường hợp thực sự dương tính. Recall = TP / (TP + FN), trong đó FN là False Negative. Recall cho biết khả năng mô hình phát hiện được các trường hợp thực sự là Cat hoặc Dog.
•	F1-Score: Trung bình điều hòa giữa Precision và Recall, giúp cân bằng giữa hai chỉ số này. F1-Score = 2 × (Precision × Recall) / (Precision + Recall).

Đánh giá mô hình với các chỉ số
# Đánh giá model trên tập validation
model = load_model('models/cat_dog_model_v2_final.h5')

# Tạo generator cho validation (không shuffle để giữ thứ tự nhãn)
datagen = ImageDataGenerator(rescale=1.0 / 255)
generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Quan trọng để mapping đúng với predictions
)

# Predict toàn bộ validation set
predictions = model.predict(generator, verbose=1)
y_true = generator.classes
y_pred = np.argmax(predictions, axis=1)

# Tính các chỉ số đánh giá
from sklearn.metrics import classification_report

class_indices = generator.class_indices
idx_to_class = {idx: name for name, idx in class_indices.items()}
class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]

# In báo cáo phân loại chi tiết
report_text = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)
print(report_text)

3.3.2. Trực quan hóa kết quả (Ma trận nhầm lẫn - Confusion Matrix)
Ma trận nhầm lẫn (Confusion Matrix) là một công cụ trực quan hóa quan trọng để hiểu rõ hơn về hiệu suất mô hình. Ma trận này cho biết số lượng dự đoán đúng và sai cho từng lớp, giúp xác định mô hình nhầm lẫn giữa các lớp như thế nào. Đối với bài toán phân loại Cat/Dog, ma trận nhầm lẫn có dạng 2x2:

```
                Predicted: Cat    Predicted: Dog
Actual: Cat     TP (True Positive)  FN (False Negative)
Actual: Dog     FP (False Positive) TN (True Negative)
```

Trong đó:
•	TP (True Positive): Số lượng ảnh Cat được dự đoán đúng là Cat.
•	FN (False Negative): Số lượng ảnh Cat bị dự đoán sai là Dog.
•	FP (False Positive): Số lượng ảnh Dog bị dự đoán sai là Cat.
•	TN (True Negative): Số lượng ảnh Dog được dự đoán đúng là Dog.

Vẽ Ma trận nhầm lẫn
# Vẽ Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Cat/Dog Classification')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('models/confusion_matrix.png', dpi=150)
plt.close()

3.3.3. Code và Output đánh giá trên tập Test
Script đánh giá hoàn chỉnh sử dụng sklearn.metrics để tính toán và in ra các chỉ số đánh giá chi tiết trên tập validation/test. Output bao gồm Precision, Recall, F1-Score cho từng lớp (Cat và Dog), cùng với Macro Average và Weighted Average.

Code đánh giá trên tập Test
# File: evaluate_model.py
def evaluate_model(model_path='models/cat_dog_model_v2_final.h5', 
                   data_dir='data', batch_size=32, img_size=(150, 150)):
    """Đánh giá model và in kết quả ra terminal."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")
    
    print("=" * 60)
    print("ĐANG ĐÁNH GIÁ MODEL CAT/DOG")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data dir: {data_dir}")
    
    # Tạo generator cho validation
    generator = load_validation_generator(data_dir, img_size=img_size, batch_size=batch_size)
    if generator.samples == 0:
        raise ValueError("Không có mẫu validation nào để đánh giá.")
    
    # Load model và predict
    model = load_model(model_path)
    predictions = model.predict(generator, verbose=1)
    y_true = generator.classes
    y_pred = np.argmax(predictions, axis=1)
    
    # Mapping class names
    class_indices = generator.class_indices
    idx_to_class = {idx: name for name, idx in class_indices.items()}
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]
    
    # In báo cáo phân loại
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    print("\nBẢNG BÁO CÁO PHÂN LOẠI (tương tự Colab sklearn):\n")
    print(report_text)
    print("=" * 60)

Output mẫu đánh giá
```
              precision    recall  f1-score   support
        cats     0.8276    0.7676    0.7965      1601
        dogs     0.7832    0.8400    0.8106      1600

    accuracy                         0.8038      3201
   macro avg     0.8054    0.8038    0.8036      3201
weighted avg     0.8054    0.8038    0.8036      3201
```

3.4. Tích hợp mô hình vào máy chủ Backend (Flask API)

3.4.1. Tải mô hình đã huấn luyện (load_model)
Trước khi có thể sử dụng mô hình để dự đoán, chúng ta cần load mô hình đã được huấn luyện từ file đã lưu. Trong ứng dụng Flask, chúng ta load mô hình một lần khi khởi động ứng dụng (singleton pattern) để tránh phải load lại nhiều lần cho mỗi request, giúp tăng hiệu suất.

Tải mô hình khi khởi động Flask
# File: app_flask.py
from tensorflow.keras.models import load_model
import os

MODEL_PATH = 'models/cat_dog_model_v2_final.h5'
model = None

def load_model_once():
    """Load model khi start app - chỉ load một lần"""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = load_model(MODEL_PATH)
                print(f"Model loaded from {MODEL_PATH}")
                return model
            except Exception as e:
                print(f"Error loading {MODEL_PATH}: {e}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
    return model

# Load model khi start app
if __name__ == '__main__':
    load_model_once()
    app.run(debug=True, host='0.0.0.0', port=5000)

3.4.2. Xây dựng API dự đoán (/predict)
API endpoint `/predict` nhận file ảnh từ client (tối đa 6 ảnh mỗi lần), tiền xử lý ảnh (resize về 150x150, chuẩn hóa pixel về [0,1]), sau đó sử dụng mô hình đã load để dự đoán. Kết quả được trả về dưới dạng JSON bao gồm tên file, lớp dự đoán (Cat hoặc Dog), độ tin cậy (confidence), và chi tiết xác suất cho từng lớp.

API endpoint dự đoán
# File: app_flask.py
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def preprocess_image(img):
    """Preprocess ảnh để predict"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150))  # Resize về kích thước input của model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
    img_array = img_array / 255.0  # Chuẩn hóa về [0, 1]
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint để predict tối đa 6 ảnh"""
    try:
        files = []
        if 'files' in request.files:
            files = request.files.getlist('files')
        elif 'file' in request.files:
            files = request.files.getlist('file')
        
        files = [f for f in files if f and f.filename]
        
        if not files:
            return jsonify({'error': 'No file selected'}), 400
        
        if len(files) > 6:
            return jsonify({'error': 'Chỉ hỗ trợ tối đa 6 ảnh mỗi lần'}), 400
        
        model = load_model_once()
        if model is None:
            return jsonify({'error': 'Model not found'}), 500
        
        # Xử lý batch: preprocess và predict cùng lúc
        batch_data = []
        file_names = []
        for idx, file in enumerate(files, start=1):
            img = Image.open(file.stream)
            img_array = preprocess_image(img)
            batch_data.append(img_array)
            file_names.append(file.filename or f'image_{idx}.png')
        
        # Predict batch
        batch_input = np.vstack(batch_data)
        predictions = model.predict(batch_input, verbose=0)
        class_names = ['cat', 'dog']
        
        # Format kết quả
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

3.5. Xây dựng giao diện trực quan hóa (Frontend)

3.5.1. Thu thập dữ liệu tải lên (Upload)
Frontend được xây dựng để cho phép người dùng upload ảnh chó hoặc mèo và nhận được kết quả dự đoán ngay lập tức. Hệ thống hỗ trợ upload nhiều file ảnh cùng lúc (tối đa 6 ảnh) thông qua input file HTML với thuộc tính `multiple`. Để cải thiện trải nghiệm người dùng, khi người dùng chọn file, ảnh được preview ngay lập tức bằng FileReader API mà không cần upload lên server, giúp người dùng xác nhận và kiểm tra ảnh trước khi gửi request dự đoán.

Các tính năng của phần upload:
•	Giới hạn số lượng: Tối đa 6 ảnh mỗi lần để đảm bảo hiệu suất xử lý tốt.
•	Validation định dạng: Chỉ chấp nhận các định dạng ảnh phổ biến (PNG, JPG, JPEG).
•	Giới hạn kích thước: Mỗi file tối đa 16MB để tránh quá tải server.
•	Preview real-time: Hiển thị preview ảnh ngay khi chọn, không cần upload.
•	Giao diện thân thiện: Sử dụng drag-and-drop area với thiết kế trực quan.

Thu thập dữ liệu upload
<!-- File: templates/index.html -->
<div class="upload-area" onclick="document.getElementById('fileInput').click()">
    <p style="font-size: 18px; margin-bottom: 10px;">📤 Click để chọn ảnh</p>
    <p style="color: #666; font-size: 14px;">PNG, JPG, JPEG (tối đa 6 ảnh, 16MB/ảnh)</p>
    <input type="file" id="fileInput" accept="image/png,image/jpeg,image/jpg" multiple>
</div>

<script>
    const fileInput = document.getElementById('fileInput');
    const previewsContainer = document.getElementById('previewsContainer');
    let selectedFiles = [];
    
    fileInput.addEventListener('change', function(e) {
        selectedFiles = Array.from(e.target.files).slice(0, 6);
        
        if (e.target.files.length > 6) {
            alert('Chỉ chọn tối đa 6 ảnh, hệ thống đã lấy 6 ảnh đầu tiên.');
        }
        
        // Preview ảnh đã chọn
        previewsContainer.innerHTML = '';
        selectedFiles.forEach((file, idx) => {
            const previewCard = document.createElement('div');
            previewCard.className = 'preview-card';
            previewCard.innerHTML = `
                <img class="preview-image" alt="Preview ${idx + 1}">
                <div class="preview-name">${file.name}</div>
            `;
            previewsContainer.appendChild(previewCard);
            
            const reader = new FileReader();
            reader.onload = (event) => {
                previewCard.querySelector('.preview-image').src = event.target.result;
            };
            reader.readAsDataURL(file);
        });
        
        // Kích hoạt nút Predict
        document.getElementById('predictBtn').disabled = selectedFiles.length === 0;
    });
</script>

3.5.2. Gửi request dự đoán và xử lý phản hồi
Sau khi người dùng chọn ảnh và nhấn nút "Phân Tích", frontend sẽ tạo FormData chứa các file ảnh đã chọn và gửi POST request đến endpoint `/predict` của Flask backend. Trong quá trình xử lý, một spinner loading được hiển thị để thông báo cho người dùng biết hệ thống đang xử lý. Khi nhận được phản hồi từ server, frontend sẽ xử lý JSON response chứa kết quả dự đoán cho từng ảnh, bao gồm lớp dự đoán (cat hoặc dog), độ tin cậy (confidence), và chi tiết xác suất cho cả hai lớp.

Xử lý lỗi: Frontend có cơ chế xử lý lỗi tốt, hiển thị thông báo rõ ràng khi có lỗi xảy ra (ví dụ: file không hợp lệ, server lỗi, model chưa được load).

Gửi request dự đoán
<!-- File: templates/index.html -->
<script>
    async function predictImage() {
        if (!selectedFiles.length) return;
        
        // Hiển thị loading
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const resultsContainer = document.getElementById('resultsContainer');
        loading.style.display = 'block';
        error.style.display = 'none';
        resultsContainer.innerHTML = '';
        
        // Tạo FormData và gửi request
        const formData = new FormData();
        selectedFiles.forEach(file => formData.append('files', file));
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Có lỗi xảy ra');
            }
            
            if (!data.results || !data.results.length) {
                throw new Error('Không nhận được kết quả từ server');
            }
            
            // Xử lý và hiển thị kết quả
            displayResults(data.results);
            
        } catch (err) {
            error.textContent = '❌ ' + err.message;
            error.style.display = 'block';
        } finally {
            loading.style.display = 'none';
        }
    }
</script>

3.5.3. Hiển thị kết quả dự đoán và xác suất
Sau khi nhận được kết quả từ API `/predict`, frontend sẽ hiển thị kết quả dự đoán cho từng ảnh một cách trực quan và dễ hiểu. Mỗi kết quả được hiển thị trong một card riêng biệt bao gồm:
•	Ảnh gốc đã upload: Hiển thị ảnh mà người dùng đã chọn.
•	Tên file: Tên của file ảnh.
•	Lớp dự đoán: Hiển thị "MÈO" hoặc "CHÓ" kèm theo emoji tương ứng (🐱 hoặc 🐶).
•	Độ tin cậy: Phần trăm độ tin cậy của dự đoán (ví dụ: 85.32%).
•	Progress bar: Thanh tiến trình trực quan hóa độ tin cậy.
•	Chi tiết xác suất: Hiển thị xác suất cho cả hai lớp (Cat và Dog) để người dùng có thể so sánh.

Kết quả được hiển thị dưới dạng grid responsive, tự động điều chỉnh số cột theo kích thước màn hình (desktop, tablet, mobile), đảm bảo trải nghiệm tốt trên mọi thiết bị.

Hiển thị kết quả dự đoán
<!-- File: templates/index.html -->
<script>
    function displayResults(results) {
        const resultsContainer = document.getElementById('resultsContainer');
        const resultsWrapper = document.getElementById('resultsWrapper');
        
        resultsContainer.innerHTML = '';
        
        results.forEach((item, idx) => {
            const emoji = item.class === 'cat' ? '🐱' : '🐶';
            const label = item.class === 'cat' ? 'MÈO' : 'CHÓ';
            
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <img class="result-image" alt="${item.filename}">
                <div class="result-filename">${item.filename}</div>
                <div class="result-class">${emoji} ${label}</div>
                <div class="result-confidence">${item.confidence}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${item.confidence}%;">
                        ${item.confidence}%
                    </div>
                </div>
                <div class="details">
                    <div class="detail-item">
                        <div class="detail-label">🐱 Mèo</div>
                        <div class="detail-value">${item.details.cat}%</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">🐶 Chó</div>
                        <div class="detail-value">${item.details.dog}%</div>
                    </div>
                </div>
            `;
            
            // Load ảnh vào card
            const reader = new FileReader();
            reader.onload = (e) => {
                card.querySelector('.result-image').src = e.target.result;
            };
            reader.readAsDataURL(selectedFiles[idx]);
            
            resultsContainer.appendChild(card);
        });
        
        resultsWrapper.style.display = 'block';
    }
</script>

<!-- CSS cho responsive grid -->
<style>
    .results-grid {
        display: grid;
        gap: 20px;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }
    
    .result-card {
        background: #f9f9f9;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    
    .result-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    
    .result-class {
        font-size: 20px;
        font-weight: bold;
        margin: 10px 0;
        color: #333;
    }
    
    .result-confidence {
        font-size: 24px;
        font-weight: bold;
        color: #667eea;
        margin: 10px 0;
    }
    
    .progress-bar {
        width: 100%;
        height: 25px;
        background: #e0e0e0;
        border-radius: 12px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        transition: width 0.5s ease;
    }
</style>

3.6. Kết quả thử nghiệm và đánh giá hiệu quả hệ thống
Sau khi hoàn thành việc triển khai, hệ thống đã được thử nghiệm và đánh giá trên tập validation với các kết quả như sau:
•	Accuracy: Khoảng 80.38% - cho thấy mô hình có khả năng phân loại đúng khoảng 8/10 ảnh.
•	Precision: Khoảng 80.54% - cho thấy trong số các dự đoán là Cat hoặc Dog, có khoảng 80.54% là đúng.
•	Recall: Khoảng 80.38% - cho thấy mô hình có thể phát hiện được khoảng 80.38% số lượng Cat/Dog thực sự trong tập dữ liệu.
•	F1-Score: Khoảng 80.36% - chỉ số cân bằng giữa Precision và Recall.

Đối với ứng dụng web:
•	Hệ thống có thể xử lý tối đa 6 ảnh cùng lúc, giúp tăng trải nghiệm người dùng.
•	Thời gian dự đoán nhanh (thường dưới 1 giây cho mỗi ảnh) nhờ batch processing.
•	Giao diện trực quan, dễ sử dụng với preview ảnh và hiển thị kết quả rõ ràng.
•	Xử lý lỗi tốt với các thông báo rõ ràng khi file không hợp lệ hoặc model chưa được load.

3.7. Đề xuất hướng cải tiến và phát triển trong tương lai
Dựa trên kết quả hiện tại và các hạn chế của hệ thống, một số hướng cải tiến có thể được đề xuất:
•	Nâng cao độ chính xác mô hình: Sử dụng Transfer Learning với các mô hình pre-trained như VGG16, ResNet, hoặc EfficientNet để tận dụng các đặc trưng đã được học từ dữ liệu lớn (ImageNet). Điều này có thể giúp nâng accuracy lên trên 90%.
•	Tăng cường dữ liệu: Thu thập thêm dữ liệu ảnh với nhiều điều kiện khác nhau (ánh sáng yếu, góc chụp đặc biệt, các giống chó/mèo hiếm) để cải thiện khả năng tổng quát hóa.
•	Mở rộng phân loại: Thay vì chỉ phân loại Cat/Dog, có thể mở rộng để phân loại nhiều loại động vật khác hoặc thậm chí phân loại theo giống (breed classification).
•	Cải thiện giao diện: Thêm tính năng drag-and-drop để upload ảnh, thêm animation khi hiển thị kết quả, hỗ trợ responsive tốt hơn cho mobile.
•	Tối ưu hiệu suất: Triển khai model trên GPU server hoặc sử dụng TensorFlow Lite để tối ưu hóa inference trên thiết bị di động. Có thể sử dụng Redis để cache kết quả cho các ảnh đã được xử lý.
•	Bảo mật: Thêm xác thực người dùng, giới hạn số lượng request, và xử lý tốt hơn các tấn công như file upload attack.
•	Deployment: Triển khai lên cloud (AWS, Google Cloud, Azure) với Docker containerization để dễ dàng scale và quản lý.
