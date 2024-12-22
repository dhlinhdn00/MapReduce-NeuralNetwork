#!/usr/bin/env python3
import sys
import json
import numpy as np

with open('model_l2.json','r') as f:
    m=json.load(f)

W1 = np.array(m['enc_weights'])  # (128,784)
b1 = np.array(m['enc_bias'])     # (128,)
W2 = np.array(m['cls_weights'])  # (10,128)
b2 = np.array(m['cls_bias'])     # (10,)

lr = m['learning_rate']

batch_size=100

# gradient
gW1 = np.zeros_like(W1)  # but we'll keep 0
gb1 = np.zeros_like(b1)
gW2 = np.zeros_like(W2)
gb2 = np.zeros_like(b2)

count=0
total_loss=0.0
total_samples=0

def relu(x):
    return np.maximum(0,x)

for line in sys.stdin:
    parts=line.strip().split()
    label=int(parts[0])
    pixels=np.array(list(map(float,parts[1:])),dtype=np.float32)

    # forward
    hidden_in = W1.dot(pixels)+b1   # shape(128,)
    hidden_out=relu(hidden_in)      # shape(128,)

    logits = W2.dot(hidden_out)+b2  # shape(10,)
    # softmax
    shift = logits - np.max(logits)
    exp_scores = np.exp(shift)
    probs = exp_scores/np.sum(exp_scores)
    loss_i = -np.log(probs[label]+1e-12)
    total_loss+=loss_i
    total_samples+=1

    # backprop to W2,b2
    delta_out = probs.copy()
    delta_out[label]-=1.0  # shape(10,)

    gW2 += np.outer(delta_out, hidden_out)
    gb2 += delta_out

    # backprop hidden => but W1,b1 freeze => we do NOT accumulate grad W1,b1
    # (explicitly ignoring or do the code but sum=0 => not used)
    #   delta_hidden = W2^T dot delta_out => ...
    #   but we do NOTHING with that => no update for W1,b1

    count+=1
    if count>=batch_size:
        out_data={
           "grad_w2": gW2.tolist(),
           "grad_b2": gb2.tolist(),
           "batch_loss": float(total_loss),
           "batch_samples": total_samples
        }
        print(json.dumps(out_data))
        gW2=np.zeros_like(W2)
        gb2=np.zeros_like(b2)
        total_loss=0
        total_samples=0
        count=0

# leftover
if count>0:
    out_data={
        "grad_w2": gW2.tolist(),
        "grad_b2": gb2.tolist(),
        "batch_loss": float(total_loss),
        "batch_samples": total_samples
    }
    print(json.dumps(out_data))
