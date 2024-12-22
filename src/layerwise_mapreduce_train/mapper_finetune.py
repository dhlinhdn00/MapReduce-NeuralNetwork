#!/usr/bin/env python3
import sys
import numpy as np
import json

with open('model_finetune.json', 'r') as f:
    model = json.load(f)

weights_input_hidden = np.array(model['weights_input_hidden'])
bias_hidden = np.array(model['bias_hidden'])
weights_hidden_output = np.array(model['weights_hidden_output'])
bias_output = np.array(model['bias_output'])

learning_rate = model['learning_rate']

batch_size = 100
batch_grad_weights_input_hidden = np.zeros_like(weights_input_hidden)
batch_grad_bias_hidden = np.zeros_like(bias_hidden)
batch_grad_weights_hidden_output = np.zeros_like(weights_hidden_output)
batch_grad_bias_output = np.zeros_like(bias_output)

count = 0

# Thêm biến tính loss, correct, samples
batch_loss = 0.0
batch_correct = 0
batch_samples = 0

for line in sys.stdin:
    parts = line.strip().split(' ')
    label = int(parts[0])
    pixels = np.array(list(map(float, parts[1:])))

    # Forward
    hidden_input = np.dot(weights_input_hidden, pixels) + bias_hidden
    hidden_output = np.maximum(hidden_input, 0)  # ReLU
    output_input = np.dot(weights_hidden_output, hidden_output) + bias_output
    exp_scores = np.exp(output_input - np.max(output_input))
    probabilities = exp_scores / np.sum(exp_scores)

    # Tính loss (cross entropy)
    loss_i = -np.log(probabilities[label] + 1e-12)
    batch_loss += loss_i
    batch_samples += 1

    # Đoán nhãn
    pred_label = np.argmax(probabilities)
    if pred_label == label:
        batch_correct += 1

    # Backprop
    delta_output = probabilities
    delta_output[label] -= 1

    grad_weights_hidden_output = np.outer(delta_output, hidden_output)
    grad_bias_output = delta_output

    delta_hidden = np.dot(weights_hidden_output.T, delta_output)
    delta_hidden[hidden_output <= 0] = 0

    grad_weights_input_hidden = np.outer(delta_hidden, pixels)
    grad_bias_hidden = delta_hidden

    # Cộng dồn gradient
    batch_grad_weights_input_hidden += grad_weights_input_hidden
    batch_grad_bias_hidden += grad_bias_hidden
    batch_grad_weights_hidden_output += grad_weights_hidden_output
    batch_grad_bias_output += grad_bias_output

    count += 1

    if count >= batch_size:
        # In ra JSON
        gradients = {
            'grad_weights_input_hidden': batch_grad_weights_input_hidden.tolist(),
            'grad_bias_hidden': batch_grad_bias_hidden.tolist(),
            'grad_weights_hidden_output': batch_grad_weights_hidden_output.tolist(),
            'grad_bias_output': batch_grad_bias_output.tolist(),
            # Thêm field cho metric
            'batch_loss': batch_loss,
            'batch_correct': batch_correct,
            'batch_samples': batch_samples
        }
        print(json.dumps(gradients))

        # Reset
        batch_grad_weights_input_hidden = np.zeros_like(weights_input_hidden)
        batch_grad_bias_hidden = np.zeros_like(bias_hidden)
        batch_grad_weights_hidden_output = np.zeros_like(weights_hidden_output)
        batch_grad_bias_output = np.zeros_like(bias_output)
        batch_loss = 0.0
        batch_correct = 0
        batch_samples = 0
        count = 0

# Nếu còn sót batch cuối
if count > 0:
    gradients = {
        'grad_weights_input_hidden': batch_grad_weights_input_hidden.tolist(),
        'grad_bias_hidden': batch_grad_bias_hidden.tolist(),
        'grad_weights_hidden_output': batch_grad_weights_hidden_output.tolist(),
        'grad_bias_output': batch_grad_bias_output.tolist(),
        'batch_loss': batch_loss,
        'batch_correct': batch_correct,
        'batch_samples': batch_samples
    }
    print(json.dumps(gradients))
