"""
Giao diện Web cho Cat/Dog CNN Classification
Sử dụng Streamlit
"""
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import sys

# Fix encoding
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Cấu hình trang
st.set_page_config(
    page_title="Cat/Dog Classifier",
    page_icon="🐱🐶",
    layout="wide"
)

# Load model (cache để không load lại mỗi lần)
@st.cache_resource
def load_cnn_model(model_path):
    """Load CNN model"""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None

# Import ensemble prediction
try:
    from src.predict_ensemble import predict_with_augmentation
    USE_ENSEMBLE = True
except:
    USE_ENSEMBLE = False

# Hàm predict với xử lý ảnh tốt hơn và augmentation
def predict_image(model, img, class_names=['cat', 'dog'], use_ensemble=True):
    """Predict một ảnh với xử lý ảnh cải thiện và augmentation"""
    try:
        # Convert ảnh sang RGB nếu có alpha channel (RGBA -> RGB)
        if img.mode == 'RGBA':
            # Tạo background trắng
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])  # Sử dụng alpha channel làm mask
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Sử dụng ensemble với augmentation để cải thiện accuracy
        if use_ensemble and USE_ENSEMBLE:
            try:
                avg_prediction = predict_with_augmentation(model, img, num_augments=5)
                predicted_class_idx = np.argmax(avg_prediction)
                confidence = avg_prediction[predicted_class_idx]
                predicted_class = class_names[predicted_class_idx]
                return predicted_class, confidence, avg_prediction
            except:
                # Fallback nếu ensemble lỗi
                pass
        
        # Standard prediction (fallback)
        # Resize ảnh về 150x150 với chất lượng tốt hơn
        img_resized = img.resize((150, 150), Image.Resampling.LANCZOS)
        
        # Convert sang array
        img_array = image.img_to_array(img_resized)
        
        # Đảm bảo shape đúng (150, 150, 3)
        if img_array.shape[2] != 3:
            # Nếu không phải 3 channels, convert lại
            img_array = img_array[:, :, :3]
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize
        
        # Đảm bảo dtype đúng
        img_array = img_array.astype(np.float32)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, 0.0, None

# UI
def main():
    st.title("🐱🐶 Cat/Dog Classification AI")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # Chọn model
        model_options = {
            "Model v2 (Best)": "models/cat_dog_model_v2.h5",
            "Model v2 Final": "models/cat_dog_model_v2_final.h5"
        }
        
        selected_model = st.selectbox(
            "Chọn Model:",
            list(model_options.keys())
        )
        
        model_path = model_options[selected_model]
        
        # Kiểm tra model có tồn tại không
        if not os.path.exists(model_path):
            st.error(f"Không tìm thấy model: {model_path}")
            st.stop()
        
        # Load model
        with st.spinner("Đang load model..."):
            model = load_cnn_model(model_path)
        
        if model is None:
            st.error("Không thể load model!")
            st.stop()
        
        st.success("✅ Model đã sẵn sàng!")
        
        st.markdown("---")
        st.markdown("### 📊 Thông tin Model")
        
        # Toggle ensemble
        use_ensemble = st.checkbox("✨ Sử dụng Augmentation (Cải thiện accuracy)", value=True)
        
        st.info(f"**Model:** {selected_model}\n\n**Accuracy:** 80.38%\n\n**Input size:** 150x150 pixels\n\n**Augmentation:** {'Bật' if use_ensemble else 'Tắt'}")
        
        st.markdown("---")
        st.markdown("### 📝 Hướng dẫn")
        st.markdown("""
        1. Upload ảnh chó hoặc mèo
        2. Xem kết quả dự đoán
        3. Xem confidence score
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Ảnh")
        
        uploaded_file = st.file_uploader(
            "Chọn ảnh chó hoặc mèo",
            type=['jpg', 'jpeg', 'png'],
            help="Upload ảnh có định dạng JPG, JPEG hoặc PNG"
        )
        
        if uploaded_file is not None:
            # Hiển thị ảnh
            img = Image.open(uploaded_file)
            st.image(img, caption="Ảnh đã upload", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Phân tích", type="primary", use_container_width=True):
                with st.spinner("Đang phân tích..."):
                    predicted_class, confidence, all_predictions = predict_image(model, img, use_ensemble=use_ensemble)
                
                # Kiểm tra lỗi
                if predicted_class is None or all_predictions is None:
                    st.error("Có lỗi xảy ra khi phân tích ảnh. Vui lòng thử lại với ảnh khác.")
                    return
                
                # Hiển thị kết quả
                with col2:
                    st.header("📊 Kết quả")
                    
                    # Icon và class
                    if predicted_class == 'cat':
                        st.markdown("### 🐱 **Kết quả: MÈO**")
                    else:
                        st.markdown("### 🐶 **Kết quả: CHÓ**")
                    
                    # Confidence bar
                    st.markdown(f"**Độ tin cậy: {confidence*100:.2f}%**")
                    st.progress(confidence)
                    
                    # Chi tiết
                    st.markdown("---")
                    st.markdown("### 📈 Chi tiết:")
                    
                    col_cat, col_dog = st.columns(2)
                    
                    with col_cat:
                        cat_conf = all_predictions[0] * 100
                        st.metric("🐱 Cat", f"{cat_conf:.2f}%")
                        st.progress(all_predictions[0])
                    
                    with col_dog:
                        dog_conf = all_predictions[1] * 100
                        st.metric("🐶 Dog", f"{dog_conf:.2f}%")
                        st.progress(all_predictions[1])
                    
                    # Thông báo
                    if confidence > 0.8:
                        st.success("✅ Độ tin cậy cao!")
                    elif confidence > 0.6:
                        st.warning("⚠️ Độ tin cậy trung bình")
                    else:
                        st.error("❌ Độ tin cậy thấp")
        else:
            with col2:
                st.header("📊 Kết quả")
                st.info("👆 Upload ảnh để bắt đầu phân tích")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Cat/Dog CNN Classification - Powered by TensorFlow/Keras</p>
        <p>Model Accuracy: 80.38% | Dataset: 16,000+ images</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

