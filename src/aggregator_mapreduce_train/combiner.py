#!/usr/bin/env python3
import sys
import json
import numpy as np

grad_weights_input_hidden = None
grad_bias_hidden = None
grad_weights_hidden_output = None
grad_bias_output = None
count = 0

total_loss = 0.0
total_correct = 0
total_samples = 0

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

if count > 0:
    out_data = {
        'grad_weights_input_hidden': grad_weights_input_hidden.tolist(),
        'grad_bias_hidden': grad_bias_hidden.tolist(),
        'grad_weights_hidden_output': grad_weights_hidden_output.tolist(),
        'grad_bias_output': grad_bias_output.tolist(),
        'batch_loss': total_loss,
        'batch_correct': total_correct,
        'batch_samples': total_samples
    }
    print(json.dumps(out_data))
