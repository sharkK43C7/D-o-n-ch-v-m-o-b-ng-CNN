# Cat/Dog CNN Classification Project

Dự án phân loại chó/mèo sử dụng Convolutional Neural Network (CNN) với TensorFlow/Keras.

## Cấu trúc thư mục

```
cat_dog_cnn_project/
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── validation/
│       ├── cats/
│       └── dogs/
├── models/          # Lưu các model đã train
├── templates/
│   └── index.html   # Frontend HTML (giao diện đơn giản)
├── model.py         # Định nghĩa CNN model
├── train.py         # Script training (có data augmentation)
├── evaluate.py      # Script đánh giá (Precision, Recall, F1-Score)
├── app_flask.py     # Flask web app (giao diện đơn giản)
├── requirements.txt
└── README.md
```

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

2. Chuẩn bị dữ liệu:
- Tạo thư mục `data/train` và `data/validation`
- Trong mỗi thư mục, tạo 2 thư mục con: `cats/` và `dogs/`
- Đặt ảnh chó vào thư mục `dogs/`, ảnh mèo vào thư mục `cats/`

## Sử dụng

### 🌐 Web Interface

Chạy giao diện web (trình duyệt sẽ tự động mở):
```bash
python app_flask.py
```

**Chức năng:**
- Upload tối đa 5 ảnh chó/mèo cùng lúc
- Xem preview ảnh trước khi phân tích
- Xem kết quả dự đoán với độ tin cậy (%)

### 📊 Training

Train model với data augmentation:
```bash
python train.py --data_dir data --epochs 50 --batch_size 32
```

**Các tham số:**
- `--data_dir`: Đường dẫn đến thư mục data (mặc định: `data`)
- `--epochs`: Số epochs (mặc định: 50)
- `--batch_size`: Batch size (mặc định: 32)

**Data Augmentation:**
- Rotation (±40°)
- Translation (shift 20%)
- Zoom (±20%)
- Horizontal flip
- Shear transformation

**Callbacks:**
- ModelCheckpoint: Lưu model tốt nhất
- EarlyStopping: Dừng sớm khi không cải thiện
- ReduceLROnPlateau: Tự động điều chỉnh learning rate

### 📈 Evaluation

Đánh giá model bằng 3 độ đo: **Precision, Recall, F1-Score**:
```bash
python evaluate.py --model models/cat_dog_model.h5 --data_dir data
```

**Output:**
- Precision, Recall, F1-Score cho từng class
- Macro và Weighted averages
- Confusion Matrix (vẽ đồ thị)
- Classification Report chi tiết

## Model Architecture

CNN Model:
- 4 Convolutional layers (32, 64, 128, 128 filters)
- MaxPooling sau mỗi Conv layer
- Dense layer 512 units
- Dropout 0.5
- Softmax output (2 classes: cat, dog)

## Features

- ✅ Training với data augmentation
- ✅ Callbacks tự động (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
- ✅ Đánh giá với 3 độ đo: Precision, Recall, F1-Score
- ✅ Giao diện web đơn giản
- ✅ Tự động lưu model tốt nhất
- ✅ Vẽ đồ thị training history

## Lưu ý

- Kích thước ảnh input: 150x150 pixels
- Model output: 2 classes (cat, dog)
- Loss: categorical crossentropy
- Optimizer: Adam
