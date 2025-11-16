"""
Script đánh giá model chi tiết với metrics
"""
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Fix encoding
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass


def evaluate_model(model_path, data_dir, batch_size=32, img_size=(150, 150)):
    """
    Đánh giá model với metrics chi tiết
    
    Args:
        model_path: Đường dẫn đến model file
        data_dir: Đường dẫn đến thư mục data (có validation folder)
        batch_size: Batch size
        img_size: Kích thước ảnh
    """
    print("=" * 60)
    print("DANH GIA MODEL CHI TIET")
    print("=" * 60)
    
    # Load model
    print(f"\nDang load model: {model_path}")
    model = load_model(model_path)
    
    # Chuẩn bị validation data
    validation_dir = os.path.join(data_dir, 'validation')
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Quan trọng để giữ thứ tự cho confusion matrix
    )
    
    print(f"\nSo luong anh validation: {validation_generator.samples}")
    print(f"Classes: {validation_generator.class_indices}")
    
    # Predict
    print("\nDang predict...")
    predictions = model.predict(validation_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # True labels
    true_classes = validation_generator.classes
    
    # Class names
    class_names = list(validation_generator.class_indices.keys())
    
    # Tính metrics
    print("\n" + "=" * 60)
    print("KET QUA DANH GIA")
    print("=" * 60)
    
    # Overall accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_names,
        digits=4
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)
    print(f"\n{'':>10}", end='')
    for name in class_names:
        print(f"{name:>15}", end='')
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name:>10}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>15}", end='')
        print()
    
    # Tính precision, recall, F1 cho từng class
    print("\n" + "-" * 60)
    print("METRICS CHO TUNG CLASS")
    print("-" * 60)
    
    for i, class_name in enumerate(class_names):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Support:   {cm[i, :].sum()}")
    
    # Vẽ confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Lưu đồ thị
    output_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else 'models'
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix da duoc luu tai: {cm_path}")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': classification_report(
            true_classes, predicted_classes, 
            target_names=class_names, output_dict=True
        )
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Cat/Dog CNN Model')
    parser.add_argument('--model', type=str, required=True,
                        help='Đường dẫn đến model file (.h5)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Đường dẫn đến thư mục data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )


