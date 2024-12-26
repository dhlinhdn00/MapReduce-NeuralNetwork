#!/bin/bash

EPOCHS=10

# Đường dẫn input training data trên HDFS
INPUT_PATH=/user/meos/mr_nn/data/mnist_train.txt

# Các thư mục output tạm trên HDFS cho 2-stage aggregator
OUTPUT1_BASE=/user/meos/mr_nn/output/finetune_stage1_epoch
OUTPUT2_BASE=/user/meos/mr_nn/output/finetune_stage2_epoch

# Đường dẫn file model cục bộ (đã chứa W1,b1, W2,b2)
MODEL_PATH=/home/meos/Documents/MapReduceNeuralNetwork/src/layerwise_mapreduce_train/model_finetune.json

# Thư mục log local
LOG_DIR=/home/meos/Documents/MapReduceNeuralNetwork/src/layerwise_mapreduce_train/logs
mkdir -p "$LOG_DIR"

# Nếu chưa có model_finetune.json => init
if [ ! -f "$MODEL_PATH" ]; then
   echo "[FINETUNE] Initializing model_finetune.json..."
   python init_finetune.py
fi

for ((e=1; e<=EPOCHS; e++))
do
  echo "=== FINETUNE EPOCH $e / $EPOCHS ==="

  # 1) Upload model lên HDFS để mapper/reducer đọc
  hdfs dfs -put -f "$MODEL_PATH" /user/meos/mr_nn/output/model_finetune.json

  # 2) Stage1: Map + Combiner + aggregatorN (n reducers), tạo partial sums
  OUT1="${OUTPUT1_BASE}${e}"
  hdfs dfs -rm -r -f "$OUT1" >/dev/null 2>&1

  echo "[Stage1] Aggregator N reducers => partial grads"
  hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
    -D mapreduce.job.reduces=4 \
    -D mapreduce.job.name="Finetune_Stage1_Epoch_${e}" \
    -input "$INPUT_PATH" \
    -output "$OUT1" \
    -mapper mapper_finetune.py \
    -combiner combiner_finetune.py \
    -reducer aggregatorN_finetune.py \
    -file mapper_finetune.py \
    -file combiner_finetune.py \
    -file aggregatorN_finetune.py \
    -file model_finetune.json

  # 3) Stage2: Map=cat + aggregator1 (1 reducer), update model
  OUT2="${OUTPUT2_BASE}${e}"
  hdfs dfs -rm -r -f "$OUT2" >/dev/null 2>&1

  echo "[Stage2] Single reducer => update model"
  hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
    -D mapreduce.job.reduces=1 \
    -D mapreduce.job.name="Finetune_Stage2_Epoch_${e}" \
    -input "$OUT1" \
    -output "$OUT2" \
    -mapper cat \
    -reducer aggregator1_finetune.py \
    -file aggregator1_finetune.py \
    -file model_finetune.json

  # 4) Tải output cuối (model + metrics) về local
  EPOCH_OUT_JSON="${LOG_DIR}/finetune_epoch${e}_out.json"
  hdfs dfs -cat "${OUT2}/part-00000" > "$EPOCH_OUT_JSON"

  # 5) Tách model + metrics; ghi đè model_finetune.json
  python <<EOF
import json
import sys

f_in="${EPOCH_OUT_JSON}"
f_model="${MODEL_PATH}"
with open(f_in,'r') as fi:
    data=json.load(fi)

model = data.get('model', {})
metrics= data.get('metrics', {})

with open(f_model,'w') as fm:
    json.dump(model, fm)

loss  = metrics.get('epoch_loss', 0.0)
acc   = metrics.get('epoch_accuracy', 0.0)
epoch = ${e}
print(f"[FINETUNE] epoch={epoch}, loss={loss:.6f}, acc={acc:.2%}")
EOF

done

echo "[FINETUNE] All done after $EPOCHS epochs."
