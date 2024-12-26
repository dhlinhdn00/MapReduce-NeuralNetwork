#!/bin/bash

# Number of training epochs
NUM_EPOCHS=10

# HDFS and local paths
INPUT_PATH=/user/meos/mr_nn/data/mnist_train.txt
OUTPUT1_BASE=/user/meos/mr_nn/output/stage1_epoch
OUTPUT2_BASE=/user/meos/mr_nn/output/stage2_epoch
LOCAL_MODEL_PATH=/home/meos/Documents/MapReduceNeuralNetwork/src/aggregator_mapreduce_train/model.json
LOG_DIR=/home/meos/Documents/MapReduceNeuralNetwork/src/aggregator_mapreduce_train/log

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Initialize log file with headers
LOG_FILE="${LOG_DIR}/train_log.txt"
echo "Epoch,Start_Time,End_Time,Duration_Seconds,CPU_Percent,Memory_Used_Bytes,Loss,Accuracy" > "$LOG_FILE"

# Initialize the model if it doesn't exist
if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "Initializing model..."
    python initialize_model.py
    echo "Model initialized."
fi

# Iterate over each epoch
for ((epoch=1; epoch<=NUM_EPOCHS; epoch++))
do
  # Define output paths for the current epoch
  OUTPUT1_PATH="${OUTPUT1_BASE}${epoch}"
  OUTPUT2_PATH="${OUTPUT2_BASE}${epoch}"
  EPOCH_OUT_JSON="${LOG_DIR}/epoch${epoch}_output.json"

  echo "==========================================="
  echo "Starting Epoch ${epoch}..."

  # Record the start time
  START_TIME=$(date +%s)

  # Upload the local model to HDFS
  hdfs dfs -put -f "$LOCAL_MODEL_PATH" /user/meos/mr_nn/output/model.json

  # Remove previous output from HDFS if it exists
  hdfs dfs -rm -r -f "$OUTPUT1_PATH" >/dev/null 2>&1

  echo "[Job1] Summation with multiple reducers..."
  
  # Execute Job1 with timing
  {
    time hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
      -D mapreduce.job.reduces=4 \
      -D mapreduce.job.maps=2 \
      -D mapreduce.map.memory.mb=1024 \
      -D mapreduce.reduce.memory.mb=2048 \
      -D mapreduce.map.cpu.vcores=1 \
      -D mapreduce.reduce.cpu.vcores=1 \
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
  } 2> "${LOG_DIR}/epoch${epoch}_job1_time.txt"

  echo "[Job2] Final aggregation (1 reducer) & update model..."
  
  # Remove previous output from HDFS if it exists
  hdfs dfs -rm -r -f "$OUTPUT2_PATH" >/dev/null 2>&1

  # Execute Job2 with timing
  {
    time hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.5.jar \
      -D mapreduce.job.reduces=4 \
      -D mapreduce.job.maps=2 \
      -D mapreduce.map.memory.mb=1024 \
      -D mapreduce.reduce.memory.mb=2048 \
      -D mapreduce.map.cpu.vcores=1 \
      -D mapreduce.reduce.cpu.vcores=1 \
      -D mapreduce.job.name="NN_Stage2_Epoch_${epoch}" \
      -input "$OUTPUT1_PATH" \
      -output "$OUTPUT2_PATH" \
      -mapper cat \
      -reducer aggregator1.py \
      -file aggregator1.py \
      -file model.json
  } 2> "${LOG_DIR}/epoch${epoch}_job2_time.txt"

  # Retrieve the output from HDFS
  hdfs dfs -cat "${OUTPUT2_PATH}/part-00000" > "$EPOCH_OUT_JSON"

  # Parse the output and update the local model, capturing loss and accuracy
  metrics=$(python3 <<EOF
import json

epoch_file = '${EPOCH_OUT_JSON}'
local_model = '${LOCAL_MODEL_PATH}'
epoch = ${epoch}

with open(epoch_file, 'r') as f:
    data = json.load(f)

model = data.get('model', {})
metrics = data.get('metrics', {})

with open(local_model, 'w') as fm:
    json.dump(model, fm)

loss = metrics.get('epoch_loss', 0.0)
acc  = metrics.get('epoch_accuracy', 0.0)
print(f"{loss},{acc}")
EOF
)

  # Extract loss and accuracy from the Python output
  IFS=',' read LOSS ACC <<< "$metrics"

  # Record the end time
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))

  # Capture CPU and memory usage of the current script
  CPU_PERCENT=$(ps -p $$ -o %cpu= | tr -d ' ')
  MEMORY_USED=$(ps -p $$ -o rss= | tr -d ' ')

  # Display epoch results to the terminal
  echo "Epoch ${epoch} => Loss = ${LOSS}, Acc = ${ACC}"

  # Format start and end times for logging
  START_DATETIME=$(date -d @${START_TIME} "+%Y-%m-%d %H:%M:%S")
  END_DATETIME=$(date -d @${END_TIME} "+%Y-%m-%d %H:%M:%S")

  # Log the epoch details to the log file
  echo "${epoch},${START_DATETIME},${END_DATETIME},${DURATION},${CPU_PERCENT},${MEMORY_USED},${LOSS},${ACC}" >> "$LOG_FILE"

  echo "Epoch ${epoch} completed."
done

echo "==========================================="
echo "Training finished after $NUM_EPOCHS epochs."
echo "Logs are available in: $LOG_FILE"
