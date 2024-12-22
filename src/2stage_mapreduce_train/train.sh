#!/bin/bash

NUM_EPOCHS=10
INPUT_PATH=/user/meos/mr_nn/data/mnist_train.txt
OUTPUT1_BASE=/user/meos/mr_nn/output/stage1_epoch
OUTPUT2_BASE=/user/meos/mr_nn/output/stage2_epoch
LOCAL_MODEL_PATH=/home/meos/Documents/MR_NN/2stage_mr_flow/model.json
LOG_DIR=/home/meos/Documents/MR_NN/2stage_mr_flow/logs
mkdir -p "$LOG_DIR"

if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "Initializing model..."
    python initialize_model.py
    echo "Model initialized."
fi

for ((epoch=1; epoch<=NUM_EPOCHS; epoch++))
do
  echo "================= EPOCH $epoch ================="

  hdfs dfs -put -f "$LOCAL_MODEL_PATH" /user/meos/mr_nn/output/model.json

  # 2) Job1: Map + Combiner + aggregatorN (n reducers) => partial sums
  OUTPUT1_PATH="${OUTPUT1_BASE}${epoch}"
  hdfs dfs -rm -r -f "$OUTPUT1_PATH" >/dev/null 2>&1

  echo "[Job1] Summation with multiple reducers..."
  hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
    -D mapreduce.job.reduces=4 \
    -D mapreduce.job.name="NN_Stage1_Epoch_${epoch}" \
    -input "$INPUT_PATH" \
    -output "$OUTPUT1_PATH" \
    -mapper mapper.py \
    -combiner combiner.py \
    -reducer aggregatorN.py \
    -file mapper.py \
    -file combiner.py \
    -file aggregatorN.py \
    -file model.json

  # 3) Job2: Map = cat + aggregator1 (1 reducer) => final model
  OUTPUT2_PATH="${OUTPUT2_BASE}${epoch}"
  hdfs dfs -rm -r -f "$OUTPUT2_PATH" >/dev/null 2>&1

  echo "[Job2] Final aggregation (1 reducer) & update model..."
  hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
    -D mapreduce.job.reduces=1 \
    -D mapreduce.job.name="NN_Stage2_Epoch_${epoch}" \
    -input "$OUTPUT1_PATH" \
    -output "$OUTPUT2_PATH" \
    -mapper cat \
    -reducer aggregator1.py \
    -file aggregator1.py \
    -file model.json

  EPOCH_OUT_JSON="${LOG_DIR}/epoch${epoch}_output.json"
  hdfs dfs -cat "${OUTPUT2_PATH}/part-00000" > "$EPOCH_OUT_JSON"

  python <<EOF
import json
import sys

epoch_file = '${EPOCH_OUT_JSON}'
local_model = '${LOCAL_MODEL_PATH}'
epoch = ${epoch}

with open(epoch_file,'r') as f:
    data = json.load(f)

model = data.get('model', {})
metrics = data.get('metrics', {})

with open(local_model,'w') as fm:
    json.dump(model, fm)

loss = metrics.get('epoch_loss', 0.0)
acc  = metrics.get('epoch_accuracy', 0.0)
print(f'>>> EPOCH {epoch} => loss={loss:.6f}, acc={acc:.2%}')

EOF

  echo "================================================="
done

echo "Training done!"
