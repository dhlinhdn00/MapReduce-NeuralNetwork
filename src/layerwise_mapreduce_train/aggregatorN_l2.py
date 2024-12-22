#!/usr/bin/env python3
import sys
import json
import numpy as np

sum_w2 = None
sum_b2 = None
total_loss = 0.0
total_samples = 0
count = 0

for line in sys.stdin:
    data = json.loads(line.strip())
    w2 = np.array(data["sum_w2"])   # do combiner output "sum_w2"
    b2 = np.array(data["sum_b2"])   # do combiner output "sum_b2"

    if sum_w2 is None:
        sum_w2 = w2
        sum_b2 = b2
    else:
        sum_w2 += w2
        sum_b2 += b2

    total_loss += data["total_loss"]
    total_samples += data["total_samples"]
    count += data["count"]

# Cuối cùng in ra 1 record
if count > 0:
    out_data = {
        "sum_w2": sum_w2.tolist(),
        "sum_b2": sum_b2.tolist(),
        "total_loss": total_loss,
        "total_samples": total_samples,
        "count": count
    }
    print(json.dumps(out_data))
