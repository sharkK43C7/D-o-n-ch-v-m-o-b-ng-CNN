"""
Script đánh giá chất lượng Cat/Dog CNN bằng Precision, Recall, F1-score.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_validation_generator(data_dir, img_size=(150, 150), batch_size=32):
    """Tạo generator cho tập validation (không shuffle để giữ thứ tự nhãn)."""
    validation_dir = os.path.join(data_dir, 'validation')
    if not os.path.isdir(validation_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục validation tại: {validation_dir}. "
            "Hãy đảm bảo cấu trúc data gồm data/train và data/validation."
        )

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    generator = datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    return generator


def evaluate_model(
    model_path='models/cat_dog_model_v2_final.h5',
    data_dir='data',
    batch_size=32,
    img_size=(150, 150),
):
    """Đánh giá model và in kết quả ra terminal."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại: {model_path}")

    print("=" * 60)
    print("ĐANG ĐÁNH GIÁ MODEL CAT/DOG")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data dir: {data_dir}")

    generator = load_validation_generator(data_dir, img_size=img_size, batch_size=batch_size)
    if generator.samples == 0:
        raise ValueError("Không có mẫu validation nào để đánh giá.")

    model = load_model(model_path)
    predictions = model.predict(generator, verbose=1)
    y_true = generator.classes
    y_pred = np.argmax(predictions, axis=1)

    class_indices = generator.class_indices
    idx_to_class = {idx: name for name, idx in class_indices.items()}
    class_names = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nBẢNG BÁO CÁO PHÂN LOẠI (tương tự Colab sklearn):\n")
    print(report_text)
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Đánh giá Cat/Dog CNN bằng Precision, Recall, F1-score')
    parser.add_argument('--model_path', type=str, default='models/cat_dog_model_v2_final.h5', help='Đường dẫn file model (.h5)')
    parser.add_argument('--data_dir', type=str, default='data', help='Thư mục chứa data (cần có data/validation)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size cho generator')
    parser.add_argument('--img_size', type=int, nargs=2, default=(150, 150), help='Kích thước resize ảnh, ví dụ: --img_size 150 150')

    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
    )

