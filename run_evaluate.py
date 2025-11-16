"""
Script chạy đánh giá model từ thư mục gốc
"""
import sys
import os

# Thêm src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import từ src module
import importlib.util
spec = importlib.util.spec_from_file_location("evaluate", os.path.join(os.path.dirname(__file__), "src", "evaluate.py"))
evaluate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_module)
evaluate_model = evaluate_module.evaluate_model

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Cat/Dog CNN Model')
    parser.add_argument('--model', type=str, default='models/cat_dog_model_v2.h5',
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


