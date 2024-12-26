#!/usr/bin/env python3
import numpy as np
import json
import psutil
import time
import os

def load_data(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            pixels = list(map(float, parts[1:]))
            X.append(pixels)
            y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


def init_model(input_size=784, hidden_size=128, output_size=10):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros(output_size)
    return W1, b1, W2, b2


def save_model(W1, b1, W2, b2, learning_rate, filename='./checkpoints/traditional_train_model_np.json'):
    model = {
        'weights_input_hidden': W1.tolist(),
        'bias_hidden': b1.tolist(),
        'weights_hidden_output': W2.tolist(),
        'bias_output': b2.tolist(),
        'learning_rate': learning_rate
    }
    with open(filename, 'w') as f:
        json.dump(model, f)


def relu(x):
    return np.maximum(0, x)


def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def forward_pass(x, W1, b1, W2, b2):
    hidden_input = W1.dot(x) + b1
    hidden_output = relu(hidden_input)
    output_input = W2.dot(hidden_output) + b2
    output = softmax(output_input)
    return hidden_input, hidden_output, output_input, output


def backward_pass(x, y, hidden_input, hidden_output, output_input, output,
                  W1, b1, W2, b2):
    # one-hot vector for label
    y_one_hot = np.zeros_like(output)
    y_one_hot[y] = 1.0

    # Delta output (Softmax + CrossEntropy)
    delta_output = output - y_one_hot  # shape (10,)

    grad_W2 = np.outer(delta_output, hidden_output)  # shape (10, 128)
    grad_b2 = delta_output  # shape (10,)

    delta_hidden = W2.T.dot(delta_output)  # shape (128,)
    delta_hidden[hidden_input <= 0] = 0  # Derivative ReLU

    grad_W1 = np.outer(delta_hidden, x)  # shape (128, 784)
    grad_b1 = delta_hidden  # shape (128,)

    return grad_W1, grad_b1, grad_W2, grad_b2


def train_local(
    train_file="mnist_train.txt",
    epochs=10,
    batch_size=100,
    learning_rate=0.001,
    save_model_path="model_local.json"
):
    log_file = r"/home/meos/Documents/MapReduceNeuralNetwork/src/traditional_train/log/training_log.csv"

    X, y = load_data(train_file)
    N = X.shape[0]
    input_size = 784
    hidden_size = 128
    output_size = 10
    W1, b1, W2, b2 = init_model(input_size, hidden_size, output_size)

    print(f"Loaded training data with {N} samples.")
    print(f"Start training for {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

    with open(log_file, 'w') as f:
        f.write("Epoch,Duration_Seconds,CPU_Percent,Memory_Usage_MB,Loss,Accuracy\n")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        indices = np.arange(N)
        np.random.shuffle(indices)

        total_loss = 0.0
        correct_count = 0

        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_idx = indices[start_idx:end_idx]
            batch_X = X[batch_idx]
            batch_y = y[batch_idx]

            acc_grad_W1 = np.zeros_like(W1)
            acc_grad_b1 = np.zeros_like(b1)
            acc_grad_W2 = np.zeros_like(W2)
            acc_grad_b2 = np.zeros_like(b2)

            for i in range(len(batch_idx)):
                xi = batch_X[i]
                yi = batch_y[i]

                h_in, h_out, o_in, o_out = forward_pass(xi, W1, b1, W2, b2)
                loss_i = -np.log(o_out[yi] + 1e-12)
                total_loss += loss_i

                pred_label = np.argmax(o_out)
                if pred_label == yi:
                    correct_count += 1

                gW1, gb1, gW2, gb2 = backward_pass(xi, yi, h_in, h_out, o_in, o_out, W1, b1, W2, b2)
                acc_grad_W1 += gW1
                acc_grad_b1 += gb1
                acc_grad_W2 += gW2
                acc_grad_b2 += gb2

            batch_len = len(batch_idx)
            W1 -= (learning_rate / batch_len) * acc_grad_W1
            b1 -= (learning_rate / batch_len) * acc_grad_b1
            W2 -= (learning_rate / batch_len) * acc_grad_W2
            b2 -= (learning_rate / batch_len) * acc_grad_b2

        avg_loss = total_loss / N
        accuracy = correct_count / N
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        used_memory_mb = memory_info.used / (1024 ** 2)

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy*100:.2f}%, Duration={epoch_duration:.2f}s, CPU={cpu_percent:.2f}%, Memory={used_memory_mb:.2f} MB")

        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{epoch_duration:.3f},{cpu_percent:.2f},{used_memory_mb:.2f},{avg_loss:.4f},{accuracy*100:.2f}\n")

    save_model(W1, b1, W2, b2, learning_rate, save_model_path)
    print(f"Model saved to {save_model_path}")


if __name__ == "__main__":
    start_time = time.time()

    train_local(
        train_file="./data/processed/mnist_train.txt",
        epochs=7,
        batch_size=100,
        learning_rate=0.01,
        save_model_path="./checkpoints/traditional_train_model_np.json"
    )

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.3f} seconds")
