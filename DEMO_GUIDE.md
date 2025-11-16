# 🎬 HƯỚNG DẪN DEMO ĐỒ ÁN - Cat/Dog CNN Classification

## 🚀 Chạy Giao Diện Web

### Bước 1: Khởi động app
```bash
python run_app.py
```

### Bước 2: Trình duyệt tự động mở
- URL: http://localhost:8501
- Nếu không tự mở, copy URL vào trình duyệt

### Bước 3: Demo
1. **Upload ảnh** - Kéo thả hoặc click để chọn ảnh
2. **Click "Phân tích"** - Xem kết quả ngay lập tức
3. **Xem kết quả** - Class, confidence, chi tiết

---

## 📊 Nội dung trình bày đồ án

### 1. Giới thiệu dự án
- **Mục tiêu:** Phân loại ảnh chó/mèo bằng CNN
- **Công nghệ:** TensorFlow/Keras, Python, Streamlit
- **Dataset:** 16,000+ ảnh từ Kaggle

### 2. Kiến trúc Model
- **Model v2:** CNN với BatchNormalization
- **Layers:** 4 Conv layers + Dense layers
- **Input:** 150x150 pixels
- **Output:** 2 classes (cat, dog)

### 3. Kết quả Training
- **Accuracy:** 80.38%
- **Precision:** Cats 82.76%, Dogs 78.32%
- **Recall:** Cats 76.76%, Dogs 84.00%
- **F1-Score:** Cats 79.65%, Dogs 81.06%

### 4. Demo thực tế
- Upload ảnh và xem kết quả
- Hiển thị confidence score
- Phân tích chi tiết

### 5. Kết luận
- Model đạt 80%+ accuracy
- Sẵn sàng sử dụng thực tế
- Có thể cải thiện với transfer learning

---

## 💡 Tips cho đồ án

1. **Chuẩn bị ảnh test:**
   - Một số ảnh chó rõ ràng
   - Một số ảnh mèo rõ ràng
   - Một số ảnh khó (để show model vẫn hoạt động)

2. **Trình bày:**
   - Show confusion matrix
   - Show training history graph
   - Demo live với giao diện web

3. **Nói về cải thiện:**
   - Transfer learning (VGG16, ResNet)
   - Data augmentation đã sử dụng
   - Hyperparameter tuning

---

## 📁 Files để trình bày

- `models/training_history_v2.png` - Đồ thị training
- `models/confusion_matrix.png` - Confusion matrix
- `FINAL_REPORT.md` - Báo cáo tổng kết
- `PROJECT_STATUS.md` - Tiến độ dự án

---

**Chúc bạn bảo vệ đồ án thành công!** 🎓🎉


