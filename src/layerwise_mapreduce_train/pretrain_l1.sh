#!/bin/bash
NUM_EPOCHS=10
INPUT_PATH=/user/meos/mr_nn/data/mnist_train.txt
OUTPUT1_BASE=/user/meos/mr_nn/l1_output/stage1_epoch
OUTPUT2_BASE=/user/meos/mr_nn/l1_output/stage2_epoch
LOCAL_MODEL_PATH=/home/meos/Documents/MapReduceNeuralNetwork/src/layerwise_mapreduce_train/model_l1.json
LOG_DIR=/home/meos/Documents/MapReduceNeuralNetwork/src/layerwise_mapreduce_train/logs
mkdir -p "$LOG_DIR"

# Initialize autoencoder model_l1.json if it doesn't exist
if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "[Init] Creating autoencoder model_l1.json..."
    python init_autoencoder_l1.py
fi

for ((e=1;e<=NUM_EPOCHS;e++))
do
  echo "=== Pretrain L1: EPOCH $e ==="
  
  # Upload the local model to HDFS
  hdfs dfs -put -f "$LOCAL_MODEL_PATH" /user/meos/mr_nn/l1_output/model_l1.json

  # Stage1: Map + Combiner + aggregatorN (n reducers)
  OUT1=${OUTPUT1_BASE}${e}
  hdfs dfs -rm -r -f "$OUT1"

  hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
    -D mapreduce.job.reduces=4 \
    -D mapreduce.job.name="PretrainL1_Stage1_E${e}" \
    -input "$INPUT_PATH" \
    -output "$OUT1" \
    -mapper mapper_l1.py \
    -combiner combiner_l1.py \
    -reducer aggregatorN_l1.py \
    -file mapper_l1.py \
    -file combiner_l1.py \
    -file aggregatorN_l1.py \
    -file model_l1.json

  # Stage2: Map=cat + aggregator1 (1 reducer)
  OUT2=${OUTPUT2_BASE}${e}
  hdfs dfs -rm -r -f "$OUT2"

  hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
    -D mapreduce.job.reduces=1 \
    -D mapreduce.job.name="PretrainL1_Stage2_E${e}" \
    -input "$OUT1" \
    -output "$OUT2" \
    -mapper cat \
    -reducer aggregator1_l1.py \
    -file aggregator1_l1.py \
    -file model_l1.json

  # Save the final output to local logs
  EPOCH_OUT_JSON="${LOG_DIR}/l1_epoch${e}_out.json"
  hdfs dfs -cat "${OUT2}/part-00000" > "$EPOCH_OUT_JSON"

  # Extract model and metrics; overwrite the local model file
  python3 <<EOF
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
print(f"[Pretrain L1] epoch={epoch}, loss={loss:.6f}, acc={acc:.2%}")
EOF

done

echo "=== Done pretraining L1 autoencoder ==="
