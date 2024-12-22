#!/usr/bin/env python3
import sys
import json
import numpy as np

# Đọc model
with open('model.json', 'r') as f:
    model = json.load(f)

lr = model['learning_rate']
Wih = np.array(model['weights_input_hidden'])
Bh  = np.array(model['bias_hidden'])
Who = np.array(model['weights_hidden_output'])
Bo  = np.array(model['bias_output'])

sum_weights_input_hidden = None
sum_bias_hidden = None
sum_weights_hidden_output = None
sum_bias_output = None

total_loss = 0.0
total_correct = 0
total_samples = 0
count_sum = 0  # tổng "count" do job1 gửi lên

for line in sys.stdin:
    partial = json.loads(line.strip())

    # Gộp gradient
    if sum_weights_input_hidden is None:
        sum_weights_input_hidden = np.array(partial["sum_weights_input_hidden"])
        sum_bias_hidden          = np.array(partial["sum_bias_hidden"])
        sum_weights_hidden_output= np.array(partial["sum_weights_hidden_output"])
        sum_bias_output          = np.array(partial["sum_bias_output"])
    else:
        sum_weights_input_hidden += np.array(partial["sum_weights_input_hidden"])
        sum_bias_hidden          += np.array(partial["sum_bias_hidden"])
        sum_weights_hidden_output+= np.array(partial["sum_weights_hidden_output"])
        sum_bias_output          += np.array(partial["sum_bias_output"])

    # Gộp metric
    total_loss    += partial["total_loss"]
    total_correct += partial["total_correct"]
    total_samples += partial["total_samples"]

    count_sum += partial["count"]

# Tính gradient trung bình & update model
if count_sum > 0:
    gradWih = sum_weights_input_hidden / count_sum
    gradBh  = sum_bias_hidden / count_sum
    gradWho = sum_weights_hidden_output / count_sum
    gradBo  = sum_bias_output / count_sum

    # Cập nhật
    Wih -= lr * gradWih
    Bh  -= lr * gradBh
    Who -= lr * gradWho
    Bo  -= lr * gradBo

    model['weights_input_hidden'] = Wih.tolist()
    model['bias_hidden']          = Bh.tolist()
    model['weights_hidden_output']= Who.tolist()
    model['bias_output']          = Bo.tolist()

# Tính loss & acc trung bình
if total_samples > 0:
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
else:
    avg_loss = 0.0
    accuracy = 0.0

output = {
    "model": model,
    "metrics": {
        "epoch_loss": avg_loss,
        "epoch_accuracy": accuracy
    }
}
print(json.dumps(output))
