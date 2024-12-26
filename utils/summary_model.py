#!/usr/bin/env python3
import json
import argparse
import numpy as np
from prettytable import PrettyTable

def load_model(filename):
    """
    Tải mô hình từ file JSON.

    Parameters:
    - filename: Đường dẫn đến file JSON chứa mô hình.

    Returns:
    - A dictionary chứa các trọng số và tham số của mô hình.
    """
    try:
        with open(filename, 'r') as f:
            model = json.load(f)
        return model
    except FileNotFoundError:
        print(f"File {filename} không tồn tại.")
        exit(1)
    except json.JSONDecodeError:
        print(f"File {filename} không phải là định dạng JSON hợp lệ.")
        exit(1)

def calculate_parameters(W, b):
    """
    Tính số lượng tham số từ ma trận trọng số và vector bias.

    Parameters:
    - W: Ma trận trọng số (numpy array).
    - b: Vector bias (numpy array).

    Returns:
    - Tổng số tham số trong lớp.
    """
    return W.size + b.size

def print_model_summary(model):
    """
    In ra tóm tắt chi tiết của mô hình neural network.

    Parameters:
    - model: Dictionary chứa các trọng số và tham số của mô hình.
    """
    # Lấy các trọng số và bias từ mô hình
    W1 = np.array(model.get('weights_input_hidden'))
    b1 = np.array(model.get('bias_hidden'))
    W2 = np.array(model.get('weights_hidden_output'))
    b2 = np.array(model.get('bias_output'))
    
    # Xác định kích thước các lớp
    input_size = W1.shape[1]
    hidden_size = W1.shape[0]
    output_size = W2.shape[0]
    
    # Tính số lượng tham số cho mỗi lớp
    params_input_hidden = calculate_parameters(W1, b1)
    params_hidden_output = calculate_parameters(W2, b2)
    total_params = params_input_hidden + params_hidden_output
    
    # Tỉ lệ phân bố tham số
    percent_input_hidden = (params_input_hidden / total_params) * 100
    percent_hidden_output = (params_hidden_output / total_params) * 100
    
    # Tạo bảng PrettyTable với thông tin chi tiết hơn
    table = PrettyTable()
    table.field_names = ["Layer (type)", "Output Shape", "Activation", "Param #", "Param %"]
    
    # Thêm các hàng vào bảng
    table.add_row(["Input", f"{input_size}", "-", "-", "-"])
    table.add_row(["Dense (Hidden)", f"{hidden_size}", "ReLU", f"{params_input_hidden}", f"{percent_input_hidden:.2f}%"])
    table.add_row(["Dense (Output)", f"{output_size}", "Softmax", f"{params_hidden_output}", f"{percent_hidden_output:.2f}%"])
    table.add_row(["Total Parameters", "-", "-", f"{total_params}", "100.00%"])
    
    # In ra summary
    print("====== Model Summary ======")
    print(table)
    print("===========================\n")
    
    # Thêm thông tin chi tiết về các lớp
    print("====== Detailed Layer Information ======")
    
    # Thông tin lớp Dense (Hidden)
    print("\nLayer 1: Dense (Hidden)")
    print(f" - Output Units: {hidden_size}")
    print(f" - Activation Function: ReLU")
    print(f" - Weights Shape: {W1.shape} (Hidden x Input)")
    print(f" - Bias Shape: {b1.shape}")
    print(f" - Number of Weights: {W1.size}")
    print(f" - Number of Biases: {b1.size}")
    print(f" - Total Parameters: {params_input_hidden}\n")
    
    # Thông tin lớp Dense (Output)
    print("Layer 2: Dense (Output)")
    print(f" - Output Units: {output_size}")
    print(f" - Activation Function: Softmax")
    print(f" - Weights Shape: {W2.shape} (Output x Hidden)")
    print(f" - Bias Shape: {b2.shape}")
    print(f" - Number of Weights: {W2.size}")
    print(f" - Number of Biases: {b2.size}")
    print(f" - Total Parameters: {params_hidden_output}\n")
    
    print(f"====== Total Parameters: {total_params} ======\n")

def main():
    parser = argparse.ArgumentParser(description="In ra tóm tắt chi tiết của mô hình neural network từ file JSON.")
    parser.add_argument("model_file", type=str, help="Đường dẫn đến file JSON chứa mô hình.")
    
    args = parser.parse_args()
    
    model = load_model(args.model_file)
    print_model_summary(model)

if __name__ == "__main__":
    main()
