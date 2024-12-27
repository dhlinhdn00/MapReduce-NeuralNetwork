#!/bin/bash

NUM_EPOCHS=10
INPUT_PATH=/user/meos/mr_nn/data/mnist_train.txt
OUTPUT1_BASE=/user/meos/mr_nn/l2_output/stage1_epoch
OUTPUT2_BASE=/user/meos/mr_nn/l2_output/stage2_epoch
LOCAL_MODEL_PATH=/home/meos/Documents/MapReduceNeuralNetwork/src/layerwise_mapreduce_train/model_l2.json
LOG_DIR=/home/meos/Documents/MapReduceNeuralNetwork/src/layerwise_mapreduce_train/logs
mkdir -p "$LOG_DIR"

# Initialize model_l2.json if it doesn't exist
if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "[Init L2] Creating model_l2 with W1,b1 from L1..."
    python init_model_l2.py
fi

for ((e=1;e<=NUM_EPOCHS;e++))
do
  echo "=== Pretrain L2: EPOCH $e ==="
  
  # Create directory on HDFS if it doesn't exist
  hdfs dfs -mkdir -p /user/meos/mr_nn/l2_output/
  
  # Upload the model to HDFS
  hdfs dfs -put -f "$LOCAL_MODEL_PATH" /user/meos/mr_nn/l2_output/model_l2.json

  # (1) Job1: map + combiner + aggregatorN
  OUT1="${OUTPUT1_BASE}${e}"
  hdfs dfs -rm -r -f "$OUT1" >/dev/null 2>&1

  echo "[Stage1] Aggregator N reducers => partial grads"
  hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
    -D mapreduce.job.reduces=4 \
    -D mapreduce.job.name="PretrainL2_Stage1_E${e}" \
    -input "$INPUT_PATH" \
    -output "$OUT1" \
    -mapper mapper_l2.py \
    -combiner combiner_l2.py \
    -reducer aggregatorN_l2.py \
    -file mapper_l2.py \
    -file combiner_l2.py \
    -file aggregatorN_l2.py \
    -file model_l2.json

  # (2) Job2: map=cat + aggregator1 (1 reducer)
  OUT2="${OUTPUT2_BASE}${e}"
  hdfs dfs -rm -r -f "$OUT2" >/dev/null 2>&1

  echo "[Stage2] Single reducer => update model"
  hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
    -D mapreduce.job.reduces=1 \
    -D mapreduce.job.name="PretrainL2_Stage2_Epoch_${e}" \
    -input "$OUT1" \
    -output "$OUT2" \
    -mapper cat \
    -reducer aggregator1_l2.py \
    -file aggregator1_l2.py \
    -file model_l2.json

  # Save the final output to local logs
  EPOCH_OUT_JSON="${LOG_DIR}/l2_epoch${e}_out.json"
  hdfs dfs -cat "${OUT2}/part-00000" > "$EPOCH_OUT_JSON"

  # Extract model and metrics; overwrite the local model file
  python <<EOF
import json

epoch = ${e}
f_in = "${EPOCH_OUT_JSON}"
f_model = "${LOCAL_MODEL_PATH}"
with open(f_in, 'r') as fi:
    data = json.load(fi)
model = data.get('model', {})
metrics = data.get('metrics', {})
with open(f_model, 'w') as fm:
    json.dump(model, fm)

loss = metrics.get('epoch_loss', 0.0)
acc = metrics.get('epoch_accuracy', 0.0)
print(f"[Pretrain L2] epoch={epoch}, loss={loss:.6f}, acc={acc:.2%}")
EOF

done

echo "=== Done pretraining layer2 (W2,b2), layer1 frozen ==="
