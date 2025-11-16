"""
Script chạy prediction nhanh từ thư mục gốc
"""
import sys
import os

# Thêm src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import predict_image, predict_batch

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict Cat/Dog from Image')
    parser.add_argument('--model', type=str, required=True,
                        help='Đường dẫn đến model file (.h5)')
    parser.add_argument('--image', type=str, default=None,
                        help='Đường dẫn đến ảnh cần predict')
    parser.add_argument('--dir', type=str, default=None,
                        help='Thư mục chứa nhiều ảnh cần predict')
    parser.add_argument('--classes', type=str, nargs='+', default=['cat', 'dog'],
                        help='Tên các classes')
    
    args = parser.parse_args()
    
    if args.image:
        predict_image(args.model, args.image, args.classes)
    elif args.dir:
        predict_batch(args.model, args.dir, args.classes)
    else:
        print("Vui lòng cung cấp --image hoặc --dir")


