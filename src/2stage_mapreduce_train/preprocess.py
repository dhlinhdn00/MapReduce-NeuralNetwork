#!/usr/bin/env python3
import os
import sys
import numpy as np
from PIL import Image
import argparse
import random

def preprocess(input_dir, output_dir, percent):
    subsets = ['train', 'test']
    
    for subset in subsets:
        subset_dir = os.path.join(input_dir, subset)
        output_file = os.path.join(output_dir, f"mnist_{subset}.txt")
        
        # Mở file đầu ra để ghi
        with open(output_file, 'w') as out_f:
            # Lặp qua các nhãn từ 0 đến 9
            for label in map(str, range(10)):
                label_dir = os.path.join(subset_dir, label)
                
                # Kiểm tra xem thư mục nhãn có tồn tại không
                if not os.path.isdir(label_dir):
                    print(f"Thư mục {label_dir} không tồn tại. Bỏ qua.")
                    continue
                
                # Lấy danh sách các file PNG trong thư mục nhãn
                files = [file for file in os.listdir(label_dir) if file.endswith('.png')]
                
                if not files:
                    print(f"Không có file PNG nào trong thư mục {label_dir}. Bỏ qua.")
                    continue
                
                # Tính số lượng file cần chọn dựa trên tỷ lệ phần trăm
                total_files = len(files)
                num_selected = int(total_files * percent / 100)
                
                if num_selected == 0:
                    print(f"Tỷ lệ phần trăm {percent}% quá thấp cho thư mục {label_dir}. Bỏ qua.")
                    continue
                
                random.seed(42)
                selected_files = random.sample(files, min(num_selected, total_files))
                
                # Lặp qua các file đã chọn và xử lý
                for file in selected_files:
                    path = os.path.join(label_dir, file)
                    try:
                        # Đọc và xử lý ảnh
                        image = Image.open(path).convert('L')  # Chuyển ảnh sang grayscale
                        image = image.resize((28, 28))  # Đảm bảo kích thước 28x28
                        pixels = np.array(image).flatten() / 255.0  # Chuẩn hóa pixel
                        label_int = int(label)  # Nhãn từ tên thư mục
                        pixels_str = ' '.join(map(str, pixels))
                        # Ghi vào file đầu ra
                        out_f.write(f"{label_int} {pixels_str}\n")
                    except Exception as e:
                        print(f"Error processing {path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MNIST PNG images to text format with optional data sampling.")
    parser.add_argument("input_dir", type=str, help="Thư mục đầu vào chứa dữ liệu MNIST đã được chia thành các tập con (train, test) và các nhãn (0-9).")
    parser.add_argument("output_dir", type=str, help="Thư mục đầu ra để lưu các file văn bản đã được xử lý.")
    parser.add_argument("--percent", type=float, default=100.0, help="Tỷ lệ phần trăm của dữ liệu để xử lý (0 < percent <= 100). Mặc định là 100%.")
    
    args = parser.parse_args()
    
    # Kiểm tra giá trị của percent
    if args.percent <= 0 or args.percent > 100:
        print("Tham số --percent phải nằm trong khoảng (0, 100].", file=sys.stderr)
        sys.exit(1)
    
    # Đảm bảo rằng tỷ lệ phần trăm là float
    percent = float(args.percent)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    preprocess(args.input_dir, args.output_dir, percent)
