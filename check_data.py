"""
Script kiểm tra dữ liệu trước khi training
"""
import os
from pathlib import Path

def check_data(data_dir='data'):
    """Kiểm tra xem có đủ dữ liệu để training không"""
    train_cats = len(list(Path(data_dir, 'train', 'cats').glob('*.jpg'))) + \
                 len(list(Path(data_dir, 'train', 'cats').glob('*.jpeg'))) + \
                 len(list(Path(data_dir, 'train', 'cats').glob('*.png')))
    
    train_dogs = len(list(Path(data_dir, 'train', 'dogs').glob('*.jpg'))) + \
                 len(list(Path(data_dir, 'train', 'dogs').glob('*.jpeg'))) + \
                 len(list(Path(data_dir, 'train', 'dogs').glob('*.png')))
    
    val_cats = len(list(Path(data_dir, 'validation', 'cats').glob('*.jpg'))) + \
               len(list(Path(data_dir, 'validation', 'cats').glob('*.jpeg'))) + \
               len(list(Path(data_dir, 'validation', 'cats').glob('*.png')))
    
    val_dogs = len(list(Path(data_dir, 'validation', 'dogs').glob('*.jpg'))) + \
               len(list(Path(data_dir, 'validation', 'dogs').glob('*.jpeg'))) + \
               len(list(Path(data_dir, 'validation', 'dogs').glob('*.png')))
    
    print("=" * 50)
    print("KIEM TRA DU LIEU")
    print("=" * 50)
    print(f"Train - Cats: {train_cats} anh")
    print(f"Train - Dogs: {train_dogs} anh")
    print(f"Validation - Cats: {val_cats} anh")
    print(f"Validation - Dogs: {val_dogs} anh")
    print("=" * 50)
    
    total_train = train_cats + train_dogs
    total_val = val_cats + val_dogs
    
    if total_train == 0:
        print("\nCHUA CO DU LIEU TRAINING!")
        print("\nVui long:")
        print("1. Tai dataset tu Kaggle: https://www.kaggle.com/datasets/salader/dogs-vs-cats")
        print("2. Hoac chay: python download_data.py --kaggle (neu co Kaggle API)")
        print("3. Dat anh vao cac thu muc:")
        print("   - data/train/cats/")
        print("   - data/train/dogs/")
        print("   - data/validation/cats/")
        print("   - data/validation/dogs/")
        return False
    
    if total_val == 0:
        print("\nCANH BAO: Chua co du lieu validation!")
        print("Khong co validation data, model se kho danh gia duoc.")
    
    print(f"\nTong cong: {total_train} anh training, {total_val} anh validation")
    print("San sang de training!")
    return True

if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    check_data(data_dir)


