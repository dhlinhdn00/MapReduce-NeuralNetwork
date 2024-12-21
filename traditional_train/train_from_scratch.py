#!/usr/bin/env python3
import random
import math
import time

def load_data(file_path):
    X = []
    y = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            pixels = [float(p) for p in parts[1:]]
            X.append(pixels)
            y.append(label)
    return X, y

def init_weights(input_size, hidden_size, output_size):
    W1 = []
    for _ in range(hidden_size):
        row = [random.gauss(0, 0.01) for _ in range(input_size)]
        W1.append(row)  # row: 1 x input_size
    b1 = [0.0]*hidden_size

    W2 = []
    for _ in range(output_size):
        row = [random.gauss(0, 0.01) for _ in range(hidden_size)]
        W2.append(row)  # row: 1 x hidden_size
    b2 = [0.0]*output_size

    return W1, b1, W2, b2

def relu(vec):
    return [max(0.0, v) for v in vec]

def softmax(logits):
    # Counter overflow
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e/s for e in exps]

def dot_mv(matrix, vec):
    out = []
    for row in matrix:
        s = 0.0
        for i, val in enumerate(row):
            s += val * vec[i]
        out.append(s)
    return out

def add_vectors(a, b):
    return [av + bv for av, bv in zip(a, b)]

def outer_product(vecA, vecB):
    mat = []
    for a in vecA:
        row = [a * b for b in vecB]
        mat.append(row)
    return mat

def forward_pass(x, W1, b1, W2, b2):
    """
    x: list 784
    W1: 128x784, b1: 128
    W2: 10x128,  b2: 10

    hidden_input = W1.dot(x) + b1
    hidden_output = relu(hidden_input)
    output_input = W2.dot(hidden_output) + b2
    output = softmax(output_input)

    Return (hidden_input, hidden_output, output_input, output)
    """
    # hidden_input
    hidden_input = dot_mv(W1, x)
    hidden_input = add_vectors(hidden_input, b1)
    # ReLU
    hidden_output = relu(hidden_input)

    # output_input
    output_input = dot_mv(W2, hidden_output)
    output_input = add_vectors(output_input, b2)

    # softmax
    output = softmax(output_input)
    return hidden_input, hidden_output, output_input, output

def backward_pass(x, y, hidden_input, hidden_output, output_input, output,
                  W1, b1, W2, b2):
    """

    x: 784 list
    y: int (label)
    hidden_input: list 128
    hidden_output: list 128
    output_input: list 10
    output: list 10 (softmax)
    W1, b1, W2, b2 (list)

    Return (grad_W1, grad_b1, grad_W2, grad_b2)
    """

    # y_one_hot
    y_one_hot = [0.0]*len(output)
    y_one_hot[y] = 1.0

    # delta_output = output - y_one_hot
    delta_output = [o - yh for o,yh in zip(output, y_one_hot)]

    # grad_W2 = outer(delta_output, hidden_output)
    grad_W2 = []
    for do in delta_output:
        row = [do * ho for ho in hidden_output]
        grad_W2.append(row)

    grad_b2 = delta_output[:]

    # delta_hidden = W2.T dot delta_output
    # shape(W2): 10x128 => W2.T: 128x10
    # => delta_hidden: shape(128)
    # Then * derivative ReLU
    # derivative ReLU: 1 if hidden_input > 0 else 0
    delta_hidden = [0.0]*len(hidden_input)
    for j in range(len(hidden_input)):  # j ~ 128
        s = 0.0
        for k in range(len(delta_output)):  # k ~ 10
            s += W2[k][j] * delta_output[k]
        # ReLU derivative
        if hidden_input[j] > 0:
            delta_hidden[j] = s
        else:
            delta_hidden[j] = 0.0

    # grad_W1 = outer(delta_hidden, x)
    grad_W1 = []
    for dh in delta_hidden:
        row = [dh * xi for xi in x]
        grad_W1.append(row)

    grad_b1 = delta_hidden[:]

    return grad_W1, grad_b1, grad_W2, grad_b2

def subtract_update(W, gradW, lr, scale):
    """
    W, gradW: matrix (list of list),
    lr, scale: float
    W -= (lr/scale)*gradW
    """
    for i in range(len(W)):
        for j in range(len(W[i])):
            W[i][j] -= (lr/scale)*gradW[i][j]

def subtract_update_vec(vec, grad, lr, scale):
    """
    vec, grad: list[float]
    lr, scale: float
    """
    for i in range(len(vec)):
        vec[i] -= (lr/scale)*grad[i]

def train_local_pure(
    train_file,
    epochs=2,
    batch_size=100,
    learning_rate=0.01,
    save_model_path="model_local.json"
):
    # 1) Load data
    X, Y = load_data(train_file)
    N = len(X)
    input_size = 784
    hidden_size = 128
    output_size = 10

    # 2) init weights
    W1, b1, W2, b2 = init_weights(input_size, hidden_size, output_size)

    print(f"Loaded training data with {N} samples.")
    print(f"Start training for {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

    for epoch in range(epochs):
        indices = list(range(N))
        random.shuffle(indices)

        total_loss = 0.0
        correct_count = 0

        start_t = time.time()

        # Loop batch
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch_idx = indices[start_idx:end_idx]
            batch_len = len(batch_idx)

            #  gradient accumulator
            acc_grad_W1 = [[0.0]*input_size for _ in range(hidden_size)]
            acc_grad_b1 = [0.0]*hidden_size
            acc_grad_W2 = [[0.0]*hidden_size for _ in range(output_size)]
            acc_grad_b2 = [0.0]*output_size

            # through sample
            for iidx in batch_idx:
                xi = X[iidx]  # list 784
                yi = Y[iidx]

                # Forward
                h_in, h_out, o_in, o_out = forward_pass(xi, W1, b1, W2, b2)

                # loss
                loss_i = -math.log(o_out[yi] + 1e-12)
                total_loss += loss_i

                # accuracy
                pred_label = -1
                best_val = -9999999.0
                for k, valk in enumerate(o_out):
                    if valk > best_val:
                        best_val = valk
                        pred_label = k
                if pred_label == yi:
                    correct_count += 1

                # Backprop
                gW1, gb1, gW2, gb2 = backward_pass(
                    xi, yi,
                    h_in, h_out, o_in, o_out,
                    W1, b1, W2, b2
                )
                # Stack
                for r in range(hidden_size):
                    for c in range(input_size):
                        acc_grad_W1[r][c] += gW1[r][c]
                for r in range(hidden_size):
                    acc_grad_b1[r] += gb1[r]
                for r in range(output_size):
                    for c in range(hidden_size):
                        acc_grad_W2[r][c] += gW2[r][c]
                for r in range(output_size):
                    acc_grad_b2[r] += gb2[r]

            # Update W, b
            scale = float(batch_len)
            subtract_update(W1, acc_grad_W1, learning_rate, scale)
            subtract_update_vec(b1, acc_grad_b1, learning_rate, scale)
            subtract_update(W2, acc_grad_W2, learning_rate, scale)
            subtract_update_vec(b2, acc_grad_b2, learning_rate, scale)

        end_t = time.time()
        epoch_time = end_t - start_t

        avg_loss = total_loss / N
        accuracy = correct_count / N
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy*100:.2f}%, Time={epoch_time:.2f}s")

    print("Training done.")
    save_model_pure(W1, b1, W2, b2, learning_rate, save_model_path)
    print(f"Model saved to {save_model_path}")


def save_model_pure(W1, b1, W2, b2, lr, path):
    """
    W1, b1, W2, b2: list
    """
    import json
    model = {
        "weights_input_hidden": W1,
        "bias_hidden": b1,
        "weights_hidden_output": W2,
        "bias_output": b2,
        "learning_rate": lr
    }
    with open(path, 'w') as f:
        json.dump(model, f)


if __name__ == "__main__":
    import sys
    train_file = "./data/processed/mnist_train.txt"
    epochs = 10
    batch_size = 100
    lr = 0.01
    save_path = "./checkpoints/traditional_train_model_pure.json"

    if len(sys.argv) >= 2:
        train_file = sys.argv[1]
    if len(sys.argv) >= 3:
        epochs = int(sys.argv[2])
    if len(sys.argv) >= 4:
        batch_size = int(sys.argv[3])
    if len(sys.argv) >= 5:
        lr = float(sys.argv[4])

    start_time = time.time()
    train_local_pure(train_file, epochs, batch_size, lr, save_path)
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.3f} seconds (pure python).")
