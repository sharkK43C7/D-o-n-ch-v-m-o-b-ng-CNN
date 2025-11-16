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
├── src/
│   ├── model.py     # Định nghĩa CNN models
│   ├── train.py     # Script training
│   └── predict.py   # Script prediction
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

### 🌐 Web Interface (Khuyến nghị cho đồ án)

Chạy giao diện web với Streamlit:
```bash
python run_app.py
```

Hoặc:
```bash
streamlit run app.py
```

Giao diện sẽ mở tự động trong trình duyệt. Bạn có thể:
- Upload ảnh chó/mèo
- Xem kết quả dự đoán với confidence score
- Xem chi tiết phân tích

### Training

Train model với dữ liệu mặc định:
```bash
cd src
python train.py
```

Train với các tùy chọn:
```bash
python train.py --data_dir ../data --model_version v2 --epochs 50 --batch_size 32
```

**Các tham số:**
- `--data_dir`: Đường dẫn đến thư mục data (mặc định: `../data`)
- `--model_version`: Phiên bản model - `v1` hoặc `v2` (mặc định: `v1`)
- `--epochs`: Số epochs (mặc định: 50)
- `--batch_size`: Batch size (mặc định: 32)

### Prediction (Command Line)

Predict một ảnh:
```bash
python run_predict.py --model models/cat_dog_model_v2.h5 --image path/to/image.jpg
```

Predict nhiều ảnh trong thư mục:
```bash
python run_predict.py --model models/cat_dog_model_v2.h5 --dir path/to/images/
```

### Evaluation

Đánh giá model với metrics chi tiết:
```bash
python run_evaluate.py --model models/cat_dog_model_v2.h5
```

## Models

### Model v1
- 4 Convolutional layers (32, 64, 128, 128 filters)
- MaxPooling sau mỗi Conv layer
- Dense layer 512 units
- Dropout 0.5

### Model v2 (Cải tiến)
- 4 Convolutional layers với BatchNormalization (32, 64, 128, 256 filters)
- 2 Dense layers (512, 256 units)
- BatchNormalization và Dropout để giảm overfitting

## Features

- Data augmentation tự động (rotation, shift, zoom, flip)
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Tự động lưu model tốt nhất
- Vẽ đồ thị training history
- Hỗ trợ predict single image hoặc batch

## Lưu ý

- Kích thước ảnh input: 150x150 pixels
- Model output: 2 classes (cat, dog)
- Sử dụng categorical crossentropy loss
- Optimizer: Adam

