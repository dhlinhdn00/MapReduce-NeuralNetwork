#!/usr/bin/env python3
import sys
import numpy as np
import json

grad_weights_input_hidden = None
grad_bias_hidden = None
grad_weights_hidden_output = None
grad_bias_output = None

total_loss = 0.0
total_correct = 0
total_samples = 0
count = 0

for line in sys.stdin:
    data = json.loads(line.strip())

    if grad_weights_input_hidden is None:
        grad_weights_input_hidden = np.array(data['grad_weights_input_hidden'])
        grad_bias_hidden = np.array(data['grad_bias_hidden'])
        grad_weights_hidden_output = np.array(data['grad_weights_hidden_output'])
        grad_bias_output = np.array(data['grad_bias_output'])
    else:
        grad_weights_input_hidden += np.array(data['grad_weights_input_hidden'])
        grad_bias_hidden += np.array(data['grad_bias_hidden'])
        grad_weights_hidden_output += np.array(data['grad_weights_hidden_output'])
        grad_bias_output += np.array(data['grad_bias_output'])

    total_loss += data['batch_loss']
    total_correct += data['batch_correct']
    total_samples += data['batch_samples']

    count += 1

# Tính trung bình gradient
grad_weights_input_hidden /= count
grad_bias_hidden /= count
grad_weights_hidden_output /= count
grad_bias_output /= count

# Đọc model hiện tại
with open('model.json', 'r') as f:
    model = json.load(f)

# Update model
lr = model['learning_rate']
model['weights_input_hidden'] = (
    np.array(model['weights_input_hidden']) - lr * grad_weights_input_hidden
).tolist()
model['bias_hidden'] = (
    np.array(model['bias_hidden']) - lr * grad_bias_hidden
).tolist()
model['weights_hidden_output'] = (
    np.array(model['weights_hidden_output']) - lr * grad_weights_hidden_output
).tolist()
model['bias_output'] = (
    np.array(model['bias_output']) - lr * grad_bias_output
).tolist()

# Tính loss, acc cho cả epoch
if total_samples > 0:
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
else:
    avg_loss = 0.0
    accuracy = 0.0

# Ta in ra 1 JSON có cả model lẫn metric
output = {
    "model": model,  # toàn bộ model
    "metrics": {
        "epoch_loss": avg_loss,
        "epoch_accuracy": accuracy
    }
}

print(json.dumps(output))
