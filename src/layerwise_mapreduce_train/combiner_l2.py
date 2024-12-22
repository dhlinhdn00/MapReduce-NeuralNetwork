#!/usr/bin/env python3
import sys
import json
import numpy as np

sum_w2 = None
sum_b2 = None
count = 0
total_loss = 0.0
total_samples = 0

for line in sys.stdin:
    data = json.loads(line.strip())
    gw2 = np.array(data["grad_w2"])
    gb2 = np.array(data["grad_b2"])

    if sum_w2 is None:
        sum_w2 = gw2
        sum_b2 = gb2
    else:
        sum_w2 += gw2
        sum_b2 += gb2

    total_loss += data["batch_loss"]
    total_samples += data["batch_samples"]
    count += 1

if count > 0:
    out_data = {
        "sum_w2": sum_w2.tolist(),
        "sum_b2": sum_b2.tolist(),
        "total_loss": total_loss,
        "total_samples": total_samples,
        "count": count
    }
    print(json.dumps(out_data))
