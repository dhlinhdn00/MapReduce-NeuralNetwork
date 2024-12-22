#!/usr/bin/env python3
import sys
import numpy as np
import json

with open('model_l1.json','r') as f:
    model = json.load(f)

EncW = np.array(model['enc_weights'])
EncB = np.array(model['enc_bias'])
DecW = np.array(model['dec_weights'])
DecB = np.array(model['dec_bias'])

lr  = model['learning_rate']
batch_size = 100

gEncW = np.zeros_like(EncW)
gEncB = np.zeros_like(EncB)
gDecW = np.zeros_like(DecW)
gDecB = np.zeros_like(DecB)

count = 0
total_loss = 0.0
total_samples = 0

def relu(x):
    return np.maximum(0,x)
def relu_grad(x):
    return (x>0).astype(np.float32)

for line in sys.stdin:
    line=line.strip()
    if not line:
        continue
    parts = line.split()
    # Bỏ qua label ở parts[0], lấy pixel ở parts[1:]
    pixels = np.array(list(map(float, parts[1:])), dtype=np.float32)  # shape(784,)

    # Forward AE
    hidden_in = EncW.dot(pixels) + EncB    # (128,)
    hidden_out= relu(hidden_in)           # (128,)
    
    recon_in = DecW.dot(hidden_out) + DecB # (784,)
    diff = recon_in - pixels
    mse  = 0.5*np.sum(diff**2)
    total_loss += mse
    total_samples += 1
    
    # Backprop
    d_recon = diff  # (784,)
    gDecW += np.outer(d_recon, hidden_out)
    gDecB += d_recon
    
    d_hidden_out = DecW.T.dot(d_recon) 
    d_hidden_in  = d_hidden_out * relu_grad(hidden_in)
    gEncW += np.outer(d_hidden_in, pixels)
    gEncB += d_hidden_in
    
    count += 1
    if count >= batch_size:
        out_data = {
            "grad_encW": gEncW.tolist(),
            "grad_encB": gEncB.tolist(),
            "grad_decW": gDecW.tolist(),
            "grad_decB": gDecB.tolist(),
            "batch_loss": float(total_loss),
            "batch_samples": total_samples
        }
        print(json.dumps(out_data))
        
        # reset
        gEncW = np.zeros_like(EncW)
        gEncB = np.zeros_like(EncB)
        gDecW = np.zeros_like(DecW)
        gDecB = np.zeros_like(DecB)
        total_loss = 0.0
        total_samples = 0
        count=0

# batch cuối
if count>0:
    out_data = {
        "grad_encW": gEncW.tolist(),
        "grad_encB": gEncB.tolist(),
        "grad_decW": gDecW.tolist(),
        "grad_decB": gDecB.tolist(),
        "batch_loss": float(total_loss),
        "batch_samples": total_samples
    }
    print(json.dumps(out_data))
