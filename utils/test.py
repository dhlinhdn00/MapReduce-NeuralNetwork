#!/usr/bin/env python3
import numpy as np
import json

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


def load_model(model_path):
    with open(model_path, 'r') as f:
        model = json.load(f)
    W1 = np.array(model['weights_input_hidden'], dtype=np.float32)
    b1 = np.array(model['bias_hidden'], dtype=np.float32)
    W2 = np.array(model['weights_hidden_output'], dtype=np.float32)
    b2 = np.array(model['bias_output'], dtype=np.float32)
    learning_rate = model['learning_rate']
    return W1, b1, W2, b2, learning_rate


def relu(x):
    return np.maximum(0, x)


def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def forward_pass(x, W1, b1, W2, b2):
    hidden_input = W1.dot(x) + b1           # shape (128,)
    hidden_output = relu(hidden_input)      # shape (128,)
    output_input = W2.dot(hidden_output) + b2  # shape (10,)
    output = softmax(output_input)          # shape (10,)
    return output


def test_model(model_path="model_local.json", test_file="mnist_test.txt"):

    W1, b1, W2, b2, lr = load_model(model_path)
    print(f"Loaded model from {model_path}, learning_rate={lr}")

    X_test, y_test = load_data(test_file)
    N = X_test.shape[0]
    print(f"Loaded test data with {N} samples.")

    correct_count = 0
    total_loss = 0.0

    for i in range(N):
        x = X_test[i]     # shape (784,)
        y = y_test[i]     # 0..9

        # Forward
        probs = forward_pass(x, W1, b1, W2, b2)
        loss_i = -np.log(probs[y] + 1e-12)
        total_loss += loss_i

        pred_label = np.argmax(probs)
        if pred_label == y:
            correct_count += 1

    accuracy = correct_count / N
    avg_loss = total_loss / N

    print(f"Test Accuracy: {accuracy*100:.2f}%")
    # print(f"Test Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    import sys

    # model_path = "/home/meos/Documents/MR_NN/checkpoints/model_mr.json"
    # model_path = "/home/meos/Documents/MapReduceNeuralNetwork/src/layerwise_mapreduce_train/model_finetune.json"
    model_path = "/home/meos/Documents/MapReduceNeuralNetwork/src/mapreduce_train/model.json"
    # model_path = "/home/meos/Documents/MapReduceNeuralNetwork/checkpoints/traditional_train_model_np.json"
    # model_path = "/home/meos/Documents/MapReduceNeuralNetwork/src/aggregator_mapreduce_train/model.json"
    test_file = "/home/meos/Documents/MR_NN/data/processed/mnist_test.txt"
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    if len(sys.argv) >= 3:
        test_file = sys.argv[2]

    test_model(model_path, test_file)
