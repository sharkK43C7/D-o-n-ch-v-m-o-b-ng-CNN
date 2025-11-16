"""
Script chạy training nhanh từ thư mục gốc
"""
import sys
import os

# Thêm src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import từ src module
import importlib.util
spec = importlib.util.spec_from_file_location("train", os.path.join(os.path.dirname(__file__), "src", "train.py"))
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
train_model = train_module.train_model

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Cat/Dog CNN Model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Đường dẫn đến thư mục data')
    parser.add_argument('--model_version', type=str, default='v1',
                        choices=['v1', 'v2'],
                        help='Phiên bản model (v1 hoặc v2)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Số epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)
    
    train_model(
        data_dir=args.data_dir,
        model_version=args.model_version,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

