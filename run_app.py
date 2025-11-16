"""
Script để chạy Streamlit app
"""
import subprocess
import sys
import os

def main():
    # Kiểm tra xem streamlit đã cài chưa
    try:
        import streamlit
    except ImportError:
        print("Streamlit chua duoc cai dat!")
        print("Dang cai dat...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "-q"])
        print("Da cai dat xong!")
    
    # Chạy app
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    main()


