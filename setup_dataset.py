"""
Script tự động tải và tổ chức dataset Dogs vs Cats
"""
import os
import sys
import zipfile
import shutil
from pathlib import Path
import random

# Fix encoding
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

def organize_dataset_from_zip(zip_path, output_dir='data'):
    """
    Tổ chức dataset từ file zip đã tải về
    """
    print(f"Dang giai nen va to chuc dataset tu: {zip_path}")
    
    if not os.path.exists(zip_path):
        print(f"Khong tim thay file: {zip_path}")
        return False
    
    # Tạo thư mục temp
    temp_dir = 'temp_extract'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Giải nén
        print("Dang giai nen...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Tìm tất cả ảnh
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(temp_dir).rglob(ext))
        
        print(f"Tim thay {len(image_files)} anh")
        
        if len(image_files) == 0:
            print("Khong tim thay anh nao trong file zip!")
            return False
        
        # Phân loại và chia train/validation
        cats = [f for f in image_files if 'cat' in f.name.lower()]
        dogs = [f for f in image_files if 'dog' in f.name.lower()]
        
        print(f"  - Cats: {len(cats)} anh")
        print(f"  - Dogs: {len(dogs)} anh")
        
        # Shuffle và chia 80/20
        random.shuffle(cats)
        random.shuffle(dogs)
        
        split_cats = int(len(cats) * 0.8)
        split_dogs = int(len(dogs) * 0.8)
        
        train_cats = cats[:split_cats]
        val_cats = cats[split_cats:]
        train_dogs = dogs[:split_dogs]
        val_dogs = dogs[split_dogs:]
        
        # Copy vào thư mục đúng
        print("Dang copy anh vao thu muc...")
        
        def copy_files(files, dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            for i, src_file in enumerate(files):
                dest_file = os.path.join(dest_dir, f"{i:05d}_{src_file.name}")
                shutil.copy2(src_file, dest_file)
        
        copy_files(train_cats, os.path.join(output_dir, 'train', 'cats'))
        copy_files(train_dogs, os.path.join(output_dir, 'train', 'dogs'))
        copy_files(val_cats, os.path.join(output_dir, 'validation', 'cats'))
        copy_files(val_dogs, os.path.join(output_dir, 'validation', 'dogs'))
        
        print(f"\nHoan thanh!")
        print(f"  - Train: {len(train_cats)} cats, {len(train_dogs)} dogs")
        print(f"  - Validation: {len(val_cats)} cats, {len(val_dogs)} dogs")
        
        # Xóa thư mục temp
        shutil.rmtree(temp_dir)
        return True
        
    except Exception as e:
        print(f"Loi: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False

def download_from_kaggle_manual():
    """
    Hướng dẫn tải thủ công từ Kaggle
    """
    print("=" * 60)
    print("HUONG DAN TAI DATASET")
    print("=" * 60)
    print("\n1. Vao trang: https://www.kaggle.com/datasets/salader/dogs-vs-cats")
    print("   Hoac: https://www.kaggle.com/c/dogs-vs-cats/data")
    print("\n2. Dang nhap vao Kaggle (neu chua co tai khoan thi dang ky)")
    print("\n3. Tai file train.zip (khoang 500MB)")
    print("\n4. Dat file train.zip vao thu muc hien tai")
    print("\n5. Chay lai lenh:")
    print("   python setup_dataset.py train.zip")
    print("\nHoac neu ban da co file zip o vi tri khac:")
    print("   python setup_dataset.py <duong_dan_den_file_zip>")
    print("=" * 60)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Kiểm tra xem có file train.zip trong thư mục hiện tại không
        possible_files = ['train.zip', 'dogs-vs-cats.zip', 'dataset.zip']
        found_file = None
        
        for f in possible_files:
            if os.path.exists(f):
                found_file = f
                break
        
        if found_file:
            print(f"Tim thay file: {found_file}")
            organize_dataset_from_zip(found_file)
        else:
            download_from_kaggle_manual()
    else:
        zip_path = sys.argv[1]
        organize_dataset_from_zip(zip_path)


