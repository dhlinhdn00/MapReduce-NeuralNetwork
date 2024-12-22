#!/bin/bash

NUM_EPOCHS=10
INPUT_PATH=/user/meos/mr_nn/data/mnist_train.txt
OUTPUT_BASE=/user/meos/mr_nn/output/epoch

# Đường dẫn file model cục bộ (đã train xong mỗi epoch sẽ ghi đè)
LOCAL_MODEL_PATH=/home/meos/Documents/MR_NN/new_flow/model.json

# Thư mục log
LOG_DIR=/home/meos/Documents/MR_NN/new_flow/log
mkdir -p "$LOG_DIR"  # Tạo nếu chưa tồn tại

# File log chính
LOG_FILE="${LOG_DIR}/train_log.txt"

# Nếu chưa có model.json -> khởi tạo
if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "Initializing model..."
    python initialize_model.py
    echo "Model initialized."
fi

# Ghi header CSV cho log
echo "Epoch,Start_Time,End_Time,Duration_Seconds,CPU_Percent,Memory_Used_Bytes" > "$LOG_FILE"

for ((i=1; i<=NUM_EPOCHS; i++))
do
  OUTPUT_PATH="${OUTPUT_BASE}${i}"
  # Tên file output tạm (sau reducer), ta tải về
  EPOCH_OUT_JSON="epoch${i}_output.json"

  echo "==========================================="
  echo "Starting Epoch ${i}..."

  # Xóa output cũ trên HDFS (nếu có)
  hdfs dfs -rm -r -f "$OUTPUT_PATH" >/dev/null 2>&1

  # Put model hiện tại lên HDFS để mapper/reducer đọc
  hdfs dfs -put -f "$LOCAL_MODEL_PATH" /user/meos/mr_nn/output/model.json

  # Dùng lệnh `time` để đo thời gian job
  {
    time hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
      -D mapreduce.job.reduces=1 \
      -D mapreduce.job.name="NN_Epoch_${i}" \
      -input "$INPUT_PATH" \
      -output "$OUTPUT_PATH" \
      -mapper mapper.py \
      -combiner combiner.py \
      -reducer reducer.py \
      -file mapper.py \
      -file combiner.py \
      -file reducer.py \
      -file model.json
  } 2> "${LOG_DIR}/epoch${i}_time.txt"

  # Tải file output từ reducer
  hdfs dfs -cat "${OUTPUT_PATH}/part-00000" > "${LOG_DIR}/${EPOCH_OUT_JSON}"

  #
  # -- ĐOẠN TÁCH MODEL & METRICS (NẾU BẠN IN RA {"model": {...}, "metrics": {...}}) --
  # Dùng Python heredoc, gán biến epoch = i (int), v.v.
  #
  python <<EOF
import json

# Gán biến epoch từ shell sang python (chú ý i là integer)
epoch = int(${i})

out_file   = "${LOG_DIR}/${EPOCH_OUT_JSON}"
local_model = "${LOCAL_MODEL_PATH}"
metrics_file = "${LOG_DIR}/epoch${i}_metrics.json"

# Đọc output JSON từ reducer (có thể là {"model": {...}, "metrics": {...}})
with open(out_file, 'r') as f:
    data = json.load(f)

# Nếu reducer chỉ in model cũ, thay code này cho phù hợp.
model = data.get("model", data)
metrics = data.get("metrics", {})

# Ghi model để epoch sau train tiếp
with open(local_model, 'w') as fm:
    json.dump(model, fm)

# Ghi metrics riêng ra file (nếu có)
with open(metrics_file, 'w') as fmet:
    json.dump(metrics, fmet)

# In ra 1 dòng log
epoch_loss = metrics.get('epoch_loss', 'N/A')
epoch_acc  = metrics.get('epoch_accuracy', 'N/A')
print(f"Epoch {epoch} => Loss = {epoch_loss}, Acc = {epoch_acc}")
EOF

  #
  # -- KẾT THÚC ĐOẠN TÁCH --
  #

  # Lấy thời gian "real" từ file epoch${i}_time.txt
  START_TIME=$(grep "real" "${LOG_DIR}/epoch${i}_time.txt" | awk '{print $2}')
  DURATION=$(grep "real" "${LOG_DIR}/epoch${i}_time.txt" | awk '{print $2}')

  # Lấy CPU và Memory của tiến trình shell (không phản ánh cluster)
  CPU_PERCENT=$(ps -p $$ -o %cpu=)
  MEMORY_USED=$(ps -p $$ -o rss=)

  # Ghi log CSV
  echo "${i},${START_TIME},${DURATION},${CPU_PERCENT},${MEMORY_USED}" >> "$LOG_FILE"

  echo "Epoch ${i} completed. Duration: ${DURATION} seconds."
done

echo "==========================================="
echo "Training finished after $NUM_EPOCHS epochs."
echo "Logs in: $LOG_FILE"
