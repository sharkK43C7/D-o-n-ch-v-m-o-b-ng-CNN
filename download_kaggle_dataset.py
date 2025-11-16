"""
Script tự động tải dataset từ Kaggle (không cần input)
"""
import os
import sys
import json
from pathlib import Path

# Fix encoding
if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

def check_kaggle_credentials():
    """Kiểm tra xem đã có Kaggle credentials chưa"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    if kaggle_file.exists():
        print(f"Tim thay kaggle.json tai: {kaggle_file}")
        return True
    else:
        print("=" * 60)
        print("CHUA CO KAGGLE CREDENTIALS")
        print("=" * 60)
        print("\nDe tai dataset tu Kaggle, ban can:")
        print("\n1. Vao: https://www.kaggle.com/settings")
        print("2. Scroll xuong phan 'API'")
        print("3. Click 'Create New Token' de tai file kaggle.json")
        print("4. Dat file vao thu muc:")
        print(f"   {kaggle_dir}")
        print("\nHoac neu ban da co username va API key, chay:")
        print("   python setup_kaggle_creds.py <username> <api_key>")
        print("\nSau do chay lai script nay.")
        return False

def download_dataset():
    """Tải dataset từ Kaggle"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print("\n" + "=" * 60)
        print("DANG TAI DATASET TU KAGGLE")
        print("=" * 60)
        
        # Thử tải từ competition trước (cần accept terms)
        print("\nThu tai tu competition 'dogs-vs-cats'...")
        print("Luu y: Neu chua accept terms, vao:")
        print("  https://www.kaggle.com/c/dogs-vs-cats/rules")
        print("  va accept terms truoc")
        
        try:
            # Tải từ competition
            api.competition_download_files('dogs-vs-cats', path='./temp_kaggle', unzip=True)
            print("Tai thanh cong tu competition!")
        except Exception as e1:
            print(f"Khong the tai tu competition: {e1}")
            print("\nThu tai tu dataset khac...")
            
            # Thử các dataset khác
            datasets = [
                'salader/dogs-vs-cats',
                'chetankv/dogs-cats-images',
                'bhavikjikadara/dogs-vs-cats-dataset',
            ]
            
            success = False
            for dataset in datasets:
                try:
                    print(f"\nThu dataset: {dataset}")
                    api.dataset_download_files(dataset, path='./temp_kaggle', unzip=True)
                    print(f"Tai thanh cong tu: {dataset}")
                    success = True
                    break
                except Exception as e2:
                    print(f"  Loi: {e2}")
                    continue
            
            if not success:
                raise Exception("Khong the tai tu bat ky nguon nao")
        
        # Tìm file train
        train_zip = None
        train_dir = None
        
        print("\nDang tim file train...")
        # Tìm tất cả các thư mục và file
        for root, dirs, files in os.walk('temp_kaggle'):
            # Tìm file zip
            for file in files:
                if 'train' in file.lower() and file.endswith('.zip'):
                    train_zip = os.path.join(root, file)
                    print(f"Tim thay file zip: {train_zip}")
                    break
            # Tìm thư mục train hoặc dataset
            for dir_name in dirs:
                if 'train' in dir_name.lower() or 'dataset' in dir_name.lower():
                    train_dir = os.path.join(root, dir_name)
                    print(f"Tim thay thu muc: {train_dir}")
                    break
            # Nếu không tìm thấy, kiểm tra xem có thư mục nào chứa ảnh không
            if not train_zip and not train_dir:
                # Đếm số file ảnh trong thư mục gốc
                image_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                if image_count > 0:
                    train_dir = root
                    print(f"Tim thay thu muc chua anh: {train_dir} ({image_count} anh)")
        
        if train_zip:
            return train_zip
        elif train_dir:
            # Tạo zip từ thư mục
            import zipfile
            zip_path = 'temp_kaggle/train.zip'
            print("Dang tao file zip tu thu muc...")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                count = 0
                for root, dirs, files in os.walk(train_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, train_dir)
                        zipf.write(file_path, arcname)
                        count += 1
                        if count % 1000 == 0:
                            print(f"  Da them {count} file...")
                print(f"  Tong cong: {count} file")
            return zip_path
        else:
            print("\nKhong tim thay file train!")
            return None
            
    except ImportError:
        print("\nCan cai dat: pip install kaggle")
        return None
    except Exception as e:
        print(f"\nLoi khi tai dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def organize_dataset(zip_path):
    """Tổ chức dataset"""
    if not zip_path or not os.path.exists(zip_path):
        print("Khong co file zip de xu ly!")
        return False
    
    print("\n" + "=" * 60)
    print("DANG TO CHUC DATASET")
    print("=" * 60)
    
    try:
        from setup_dataset import organize_dataset_from_zip
        result = organize_dataset_from_zip(zip_path)
        
        # Xóa thư mục temp
        if os.path.exists('temp_kaggle'):
            import shutil
            print("\nDang xoa thu muc tam...")
            shutil.rmtree('temp_kaggle')
        
        return result
    except Exception as e:
        print(f"Loi khi to chuc dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("TU DONG TAI DATASET TU KAGGLE")
    print("=" * 60)
    
    # Kiểm tra credentials
    if not check_kaggle_credentials():
        sys.exit(1)
    
    # Tải dataset
    zip_path = download_dataset()
    
    if not zip_path:
        print("\nKhong the tai dataset. Vui long kiem tra lai.")
        sys.exit(1)
    
    # Tổ chức dataset
    if organize_dataset(zip_path):
        print("\n" + "=" * 60)
        print("HOAN THANH!")
        print("=" * 60)
        print("\nDataset da duoc tai va to chuc thanh cong!")
        print("\nBan co the bat dau training:")
        print("  python run_training.py --model_version v2 --epochs 50")
    else:
        print("\nCo loi khi to chuc dataset.")

