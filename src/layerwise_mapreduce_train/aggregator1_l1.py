#!/usr/bin/env python3
import sys
import json
import numpy as np

# Đọc model
with open('model_l1.json','r') as f:
    model=json.load(f)

EncW=np.array(model['enc_weights'])
EncB=np.array(model['enc_bias'])
DecW=np.array(model['dec_weights'])
DecB=np.array(model['dec_bias'])

lr=model['learning_rate']

sumEncW=None
sumEncB=None
sumDecW=None
sumDecB=None

total_loss=0.0
total_samples=0

for line in sys.stdin:
    data=json.loads(line.strip())

    encW=np.array(data['sum_encW'])
    encB=np.array(data['sum_encB'])
    decW=np.array(data['sum_decW'])
    decB=np.array(data['sum_decB'])
    
    if sumEncW is None:
        sumEncW=encW
        sumEncB=encB
        sumDecW=decW
        sumDecB=decB
    else:
        sumEncW+=encW
        sumEncB+=encB
        sumDecW+=decW
        sumDecB+=decB
    
    total_loss    += data['total_loss']
    total_samples += data['total_samples']

if total_samples>0:
    # Tính gradient trung bình trên toàn dataset
    gEncW = sumEncW / total_samples
    gEncB = sumEncB / total_samples
    gDecW = sumDecW / total_samples
    gDecB = sumDecB / total_samples

    # Update
    EncW -= lr*gEncW
    EncB -= lr*gEncB
    DecW -= lr*gDecW
    DecB -= lr*gDecB

    model['enc_weights'] = EncW.tolist()
    model['enc_bias']    = EncB.tolist()
    model['dec_weights'] = DecW.tolist()
    model['dec_bias']    = DecB.tolist()

    avg_loss = total_loss / total_samples
else:
    avg_loss = 0.0

# In kết quả cuối
out_data={
    "model": model,
    "metrics":{
        "epoch_loss": avg_loss,
        "epoch_accuracy": 0.0  # autoencoder thì thường không có acc
    }
}
print(json.dumps(out_data))
