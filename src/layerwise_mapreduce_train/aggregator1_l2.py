#!/usr/bin/env python3
import sys
import json
import numpy as np

with open('model_l2.json','r') as f:
    model = json.load(f)

W1 = np.array(model['enc_weights'])  # frozen
b1 = np.array(model['enc_bias'])
W2 = np.array(model['cls_weights'])
b2 = np.array(model['cls_bias'])
lr = model['learning_rate']

sum_w2 = None
sum_b2 = None
total_loss = 0.0
total_samples = 0
count_sum = 0

for line in sys.stdin:
    data = json.loads(line.strip())

    w2 = np.array(data['sum_w2'])
    b_2= np.array(data['sum_b2'])

    if sum_w2 is None:
        sum_w2 = w2
        sum_b2 = b_2
    else:
        sum_w2 += w2
        sum_b2 += b_2

    total_loss += data['total_loss']
    total_samples += data['total_samples']
    count_sum += data['count']

if count_sum > 0:
    gradW2 = sum_w2 / count_sum
    gradB2 = sum_b2 / count_sum
    W2 -= lr * gradW2
    b2 -= lr * gradB2

model['cls_weights'] = W2.tolist()
model['cls_bias'] = b2.tolist()

if total_samples > 0:
    avg_loss = total_loss / total_samples
else:
    avg_loss = 0.0

out = {
   "model": model,
   "metrics": {
     "epoch_loss": avg_loss,
     "epoch_accuracy": 0.0 
   }
}
print(json.dumps(out))
