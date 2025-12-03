"""
Training script cho Cat/Dog CNN
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from model import create_model


def prepare_data_generators(data_dir, img_size=(150, 150), batch_size=32):
    """
    Chuẩn bị data generators với data augmentation
    
    Args:
        data_dir: Đường dẫn đến thư mục chứa data (có subfolders train/validation)
        img_size: Kích thước resize ảnh
        batch_size: Batch size cho training
    
    Returns:
        train_generator, validation_generator
    """
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')
    
    # Data augmentation cho training
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
    
    # Chỉ rescale cho validation (không augmentation)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Tạo generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator


def train_model(data_dir='data', epochs=50, batch_size=32):
    """
    Train model
    
    Args:
        data_dir: Đường dẫn đến data
        epochs: Số epochs
        batch_size: Batch size
    """
    print("=" * 50)
    print("Bắt đầu training Cat/Dog CNN Model")
    print("=" * 50)
    
    # Tạo model
    model = create_model()
    model.summary()
    
    # Chuẩn bị data
    print("\nChuẩn bị data generators...")
    train_gen, val_gen = prepare_data_generators(data_dir, batch_size=batch_size)
    
    print(f"\nSố classes: {train_gen.num_classes}")
    print(f"Class indices: {train_gen.class_indices}")
    
    # Xác định đường dẫn models
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, 'cat_dog_model.h5'),
            monitor='val_accuracy' if val_gen.samples > 0 else 'accuracy',
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
    
    # Training
    print("\nBắt đầu training...")
    
    # Xử lý trường hợp không có validation data
    validation_data = val_gen if val_gen.samples > 0 else None
    validation_steps = val_gen.samples // batch_size if val_gen.samples > 0 else None
    
    if validation_data is None:
        print("Cảnh báo: Không có validation data, chỉ training không có validation")
        callbacks = [c for c in callbacks if not isinstance(c, (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau))]
    
    # Đảm bảo steps_per_epoch >= 1
    steps_per_epoch = max(1, train_gen.samples // batch_size) if train_gen.samples > 0 else 1
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=callbacks if validation_data else [],
        verbose=1
    )
    
    # Lưu model cuối cùng
    final_model_path = os.path.join(models_dir, 'cat_dog_model_final.h5')
    model.save(final_model_path)
    print(f"\nModel đã được lưu tại: {final_model_path}")
    
    # Vẽ đồ thị
    plot_training_history(history, models_dir)
    
    return model, history


def plot_training_history(history, models_dir='models'):
    """
    Vẽ đồ thị accuracy và loss
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    history_path = os.path.join(models_dir, 'training_history.png')
    plt.savefig(history_path, dpi=150)
    print(f"Đồ thị training đã được lưu tại: {history_path}")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Cat/Dog CNN Model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Đường dẫn đến thư mục data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Số epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )