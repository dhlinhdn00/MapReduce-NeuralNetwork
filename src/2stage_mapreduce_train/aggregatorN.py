#!/usr/bin/env python3
import sys
import numpy as np
import json

sum_weights_input_hidden = None
sum_bias_hidden = None
sum_weights_hidden_output = None
sum_bias_output = None

total_loss = 0.0
total_correct = 0
total_samples = 0
count = 0

for line in sys.stdin:
    data = json.loads(line.strip())

    gwih = np.array(data['grad_weights_input_hidden'])
    gbh  = np.array(data['grad_bias_hidden'])
    gwho = np.array(data['grad_weights_hidden_output'])
    gbo  = np.array(data['grad_bias_output'])

    if sum_weights_input_hidden is None:
        sum_weights_input_hidden = gwih
        sum_bias_hidden         = gbh
        sum_weights_hidden_output = gwho
        sum_bias_output         = gbo
    else:
        sum_weights_input_hidden += gwih
        sum_bias_hidden         += gbh
        sum_weights_hidden_output += gwho
        sum_bias_output         += gbo

    # Cộng dồn metric
    total_loss    += data['batch_loss']
    total_correct += data['batch_correct']
    total_samples += data['batch_samples']

    count += 1

# Nếu reducer này không nhận input => count=0 => không in gì
if count > 0:
    out_data = {
        # partial sums
        "sum_weights_input_hidden": sum_weights_input_hidden.tolist(),
        "sum_bias_hidden": sum_bias_hidden.tolist(),
        "sum_weights_hidden_output": sum_weights_hidden_output.tolist(),
        "sum_bias_output": sum_bias_output.tolist(),
        "count": count,
        # partial metrics
        "total_loss": total_loss,
        "total_correct": total_correct,
        "total_samples": total_samples
    }
    print(json.dumps(out_data))
