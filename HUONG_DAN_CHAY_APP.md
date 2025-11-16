# 🚀 HƯỚNG DẪN CHẠY GIAO DIỆN WEB

## Cách 1: Chạy trực tiếp (Đơn giản nhất)

```bash
python run_app.py
```

## Cách 2: Chạy với Streamlit

```bash
streamlit run app.py
```

## Cách 3: Chạy với port tùy chỉnh

```bash
streamlit run app.py --server.port 8501
```

---

## 📋 Yêu cầu

1. Đã cài đặt Streamlit:
   ```bash
   pip install streamlit
   ```

2. Model đã được train và có trong thư mục `models/`:
   - `models/cat_dog_model_v2.h5` (khuyến nghị)
   - `models/cat_dog_model_v2_final.h5`

---

## 🎯 Cách sử dụng

1. **Chạy lệnh** `python run_app.py`
2. **Trình duyệt** sẽ tự động mở (thường là http://localhost:8501)
3. **Upload ảnh** chó hoặc mèo
4. **Click "Phân tích"** để xem kết quả
5. **Xem kết quả** với confidence score và chi tiết

---

## ✨ Tính năng

- ✅ Upload ảnh dễ dàng (drag & drop hoặc click)
- ✅ Hiển thị ảnh đã upload
- ✅ Kết quả dự đoán với confidence score
- ✅ Progress bar trực quan
- ✅ Chi tiết phân tích cho cả 2 class
- ✅ Giao diện đẹp, hiện đại
- ✅ Responsive design

---

## 🐛 Xử lý lỗi

### Lỗi: "ModuleNotFoundError: No module named 'streamlit'"
**Giải pháp:**
```bash
pip install streamlit
```

### Lỗi: "Không tìm thấy model"
**Giải pháp:** Đảm bảo file model có trong thư mục `models/`

### Lỗi: Port đã được sử dụng
**Giải pháp:** 
```bash
streamlit run app.py --server.port 8502
```

---

## 📸 Screenshot

Giao diện bao gồm:
- **Sidebar:** Cài đặt model, thông tin, hướng dẫn
- **Main area:** Upload ảnh và hiển thị kết quả
- **Kết quả:** Class dự đoán, confidence, chi tiết

---

**Chúc bạn demo đồ án thành công!** 🎉


