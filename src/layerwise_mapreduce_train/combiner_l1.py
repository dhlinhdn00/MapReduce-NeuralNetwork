#!/usr/bin/env python3
import sys
import json
import numpy as np

gEncW=None
gEncB=None
gDecW=None
gDecB=None

total_loss=0.0
total_samples=0

for line in sys.stdin:
    data = json.loads(line.strip())
    encW = np.array(data['grad_encW'])
    encB = np.array(data['grad_encB'])
    decW = np.array(data['grad_decW'])
    decB = np.array(data['grad_decB'])
    
    if gEncW is None:
        gEncW = encW
        gEncB = encB
        gDecW = decW
        gDecB = decB
    else:
        gEncW += encW
        gEncB += encB
        gDecW += decW
        gDecB += decB
    
    total_loss += data['batch_loss']
    total_samples += data['batch_samples']

# Mỗi combiner out 1 record
if total_samples>0:
    out_data = {
        "grad_encW": gEncW.tolist(),
        "grad_encB": gEncB.tolist(),
        "grad_decW": gDecW.tolist(),
        "grad_decB": gDecB.tolist(),
        "batch_loss": total_loss,
        "batch_samples": total_samples
    }
    print(json.dumps(out_data))
