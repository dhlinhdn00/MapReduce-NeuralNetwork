
# MapReduceNeuralNetwork

Build Neural Networks Using MapReduce

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Install Java](#1-install-java)
  - [2. Install Hadoop 3.3.5](#2-install-hadoop-335)
  - [3. Configure Hadoop](#3-configure-hadoop)
  - [4. Start Hadoop Single Node Cluster](#4-start-hadoop-single-node-cluster)
- [Usage](#usage)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Local Training](#2-local-training)
    - [Train from Scratch](#train-from-scratch)
    - [Train Using NumPy](#train-using-numpy)
  - [3. MapReduce Training](#3-mapreduce-training)
    - [Aggregator MapReduce Training](#aggregator-mapreduce-training)
    - [Layerwise MapReduce Training](#layerwise-mapreduce-training)
- [Important Scripts](#important-scripts)
- [Logging and Monitoring](#logging-and-monitoring)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

**MapReduce NeuralNetwork** is a project that leverages the MapReduce programming model to build and train neural networks. By distributing the training process across multiple nodes, this project aims to handle large datasets efficiently and scale the training process.

This repository includes:

- Data preprocessing scripts to prepare the MNIST dataset.
- Local training implementations using pure Python and NumPy.
- MapReduce-based training scripts for distributed model training.
- Utilities for downloading, preprocessing, and visualizing data.

## Features

- **Distributed Training**: Utilize Hadoop's MapReduce framework to train neural networks across multiple nodes.
- **Local Training**: Train models locally using both pure Python and optimized NumPy implementations.
- **Data Augmentation**: Enhance training data with various augmentation techniques to improve model robustness.
- **Layerwise Training**: Support for layer-wise pretraining and finetuning of neural networks.
- **Comprehensive Logging**: Monitor training progress with detailed logs capturing loss, accuracy, CPU, and memory usage.

## Directory Structure

```
.
├── dir_map.txt
├── gen_dir_map.sh
├── README.md
├── requirements.txt
├── src
│   ├── aggregator_mapreduce_train
│   │   ├── aggregator1.py
│   │   ├── aggregatorN.py
│   │   ├── combiner.py
│   │   ├── initialize_model.py
│   │   ├── mapper.py
│   │   ├── model_old.json
│   │   ├── reducer.py
│   │   └── train.sh
│   ├── layerwise_mapreduce_train
│   │   ├── aggregator1_finetune.py
│   │   ├── aggregator1_l1.py
│   │   ├── aggregator1_l2.py
│   │   ├── aggregatorN_finetune.py
│   │   ├── aggregatorN_l1.py
│   │   ├── aggregatorN_l2.py
│   │   ├── combiner_finetune.py
│   │   ├── combiner_l1.py
│   │   ├── combiner_l2.py
│   │   ├── finetune.sh
│   │   ├── init_autoencoder_l1.py
│   │   ├── init_finetune.py
│   │   ├── init_model_l2.py
│   │   ├── mapper_finetune.py
│   │   ├── mapper_l1.py
│   │   ├── mapper_l2.py
│   │   ├── pretrain_l1.sh
│   │   └── pretrain_l2.sh
│   ├── mapreduce_train
│   │   ├── combiner.py
│   │   ├── initialize_model.py
│   │   ├── mapper.py
│   │   ├── reducer.py
│   │   └── train.sh
│   └── local_train
│       ├── train_from_scratch.py
│       ├── train_using_numpy_backup.py
│       └── train_using_numpy.py
└── utils
    ├── download_MNIST.py
    ├── preprocess.py
    ├── summary_model.py
    ├── test.py
    └── visualize.ipynb

6 directories, 43 files
```

### Key Directories and Files

- **src/local_train**: Scripts for local training using pure Python and NumPy.
- **src/mapreduce_train**: Scripts for distributed training using MapReduce.
- **src/aggregator_mapreduce_train**: Aggregator scripts for MapReduce training.
- **src/layerwise_mapreduce_train**: Scripts for layer-wise pretraining and finetuning.
- **utils/preprocess.py**: Preprocessing script to convert MNIST PNG images to text format with optional data augmentation.

## Prerequisites

Before setting up the project, ensure you have the following installed on your Ubuntu 22.04 system:

- **Java JDK 8 or higher**: Required for Hadoop.
- **Python 3.6 or higher**: For running Python scripts.
- **pip**: Python package installer.
- **Git**: For cloning the repository.

## Installation

### 1. Install Java

Hadoop requires Java to run. Install OpenJDK:

```bash
sudo apt update
sudo apt install openjdk-11-jdk -y
```

Verify the installation:

```bash
java -version
```

### 2. Install Hadoop 3.3.5

#### Download Hadoop

```bash
wget https://downloads.apache.org/hadoop/common/hadoop-3.3.5/hadoop-3.3.5.tar.gz
```

#### Extract Hadoop

```bash
tar -xzvf hadoop-3.3.5.tar.gz
sudo mv hadoop-3.3.5 /usr/local/hadoop
```

#### Set Environment Variables

Add Hadoop environment variables to `.bashrc`:

```bash
echo 'export HADOOP_HOME=/usr/local/hadoop' >> ~/.bashrc
echo 'export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop' >> ~/.bashrc
echo 'export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin' >> ~/.bashrc
source ~/.bashrc
```

### 3. Configure Hadoop

#### Configure `hadoop-env.sh`

Edit `$HADOOP_HOME/etc/hadoop/hadoop-env.sh` to set Java home:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

#### Configure Core Hadoop Files

Edit the following configuration files located in `$HADOOP_HOME/etc/hadoop/`:

##### `core-site.xml`

```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://localhost:9000</value>
    </property>
</configuration>
```

##### `hdfs-site.xml`

```xml
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:///usr/local/hadoop/data/namenode</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:///usr/local/hadoop/data/datanode</value>
    </property>
</configuration>
```

##### `mapred-site.xml`

Create `mapred-site.xml` by copying the template:

```bash
cp $HADOOP_HOME/etc/hadoop/mapred-site.xml.template $HADOOP_HOME/etc/hadoop/mapred-site.xml
```

Edit `mapred-site.xml`:

```xml
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
```

##### `yarn-site.xml`

```xml
<configuration>
    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>
</configuration>
```

#### Format HDFS Namenode

```bash
hdfs namenode -format
```

### 4. Start Hadoop Single Node Cluster

Start HDFS and YARN:

```bash
start-dfs.sh
start-yarn.sh
```

Verify HDFS is running:

```bash
hdfs dfs -ls /
```

You should see directories like `/user` and `/tmp`.

## Usage

### 1. Data Preprocessing

The preprocessing script converts MNIST PNG images into a text format suitable for training.

#### Download MNIST Data

Use the provided utility script:

```bash
python3 utils/download_MNIST.py --output_dir ./data/raw
```

#### Preprocess Data

```bash
python3 utils/preprocess.py ./data/raw ./data/processed --percent 100 --augment
```

- **Parameters**:
  - `input_dir`: Directory containing raw MNIST data split into `train` and `test` subsets.
  - `output_dir`: Directory to save processed text files.
  - `--percent`: Percentage of data to process (default: 100).
  - `--augment`: Enable data augmentation.

### 2. Local Training

#### Train from Scratch

Train a neural network locally using pure Python.

```bash
python3 src/local_train/train_from_scratch.py ./data/processed/mnist_train.txt 10 100 0.01
```

- **Parameters**:
  - `train_file`: Path to the training data file.
  - `epochs`: Number of training epochs (default: 10).
  - `batch_size`: Size of each training batch (default: 100).
  - `lr`: Learning rate (default: 0.01).

#### Train Using NumPy

Train a neural network locally using NumPy for optimized performance.

```bash
python3 src/local_train/train_using_numpy.py ./data/processed/mnist_train.txt 10 100 0.01
```

- **Parameters**:
  - `train_file`: Path to the training data file.
  - `epochs`: Number of training epochs (default: 10).
  - `batch_size`: Size of each training batch (default: 100).
  - `learning_rate`: Learning rate (default: 0.01).

### 3. MapReduce Training

Ensure Hadoop is running and HDFS is accessible.

#### Aggregator MapReduce Training

Navigate to the aggregator MapReduce training directory and execute the training script.

```bash
cd src/aggregator_mapreduce_train
bash train.sh
```

- **Description**: This script initializes the model and runs MapReduce jobs for each epoch, updating the model iteratively.

#### Layerwise MapReduce Training

Layerwise training involves pretraining individual layers before finetuning the entire network.

##### Pretrain Layer 1

```bash
cd src/layerwise_mapreduce_train
bash pretrain_l1.sh
```

##### Pretrain Layer 2

```bash
bash pretrain_l2.sh
```

##### Finetune the Model

```bash
bash finetune.sh
```

- **Description**: These scripts handle the pretraining of individual layers and the finetuning process using MapReduce jobs.

## Important Scripts

### Utils

- **preprocess.py**: Preprocesses MNIST PNG images into normalized text format with optional data augmentation.
- **download_MNIST.py**: Downloads the MNIST dataset.
- **visualize.ipynb**: Jupyter Notebook for visualizing the dataset and model performance.

### Local Training

- **train_from_scratch.py**: Pure Python implementation for training a neural network from scratch.
- **train_using_numpy.py**: NumPy-optimized implementation for faster local training.

### MapReduce Training

- **mapper.py / combiner.py / reducer.py**: MapReduce components for distributed training.
- **aggregator1.py / aggregatorN.py**: Aggregator scripts to combine gradients from reducers.
- **train.sh**: Shell script to orchestrate MapReduce training across epochs.

### Layerwise MapReduce Training

- **mapper_l1.py / combiner_l1.py / aggregatorN_l1.py**: Scripts for pretraining layer 1.
- **mapper_l2.py / combiner_l2.py / aggregatorN_l2.py**: Scripts for pretraining layer 2.
- **mapper_finetune.py / combiner_finetune.py / aggregatorN_finetune.py**: Scripts for finetuning the model.
- **finetune.sh / pretrain_l1.sh / pretrain_l2.sh**: Shell scripts to execute respective training stages.

## Logging and Monitoring

- **Log Files**: Training scripts generate log files capturing loss, accuracy, CPU usage, and memory consumption.
  - Local training logs are printed to the console and saved as specified in the scripts.
  - MapReduce training logs are saved in the `log` directories within the respective training script folders.
- **Training Metrics**: Each epoch's metrics are saved in JSON format for easy analysis and visualization.

## Troubleshooting

- **Hadoop Not Starting**: Ensure Java is correctly installed and environment variables are properly set. Check Hadoop logs located in `$HADOOP_HOME/logs`.
- **Permission Issues**: Run Hadoop commands with appropriate permissions. Avoid using `sudo` unless necessary.
- **Insufficient Memory**: Adjust Hadoop's memory settings in the `train.sh` scripts if you encounter memory-related errors.
- **Data Not Found**: Ensure that the preprocessing step has been completed and the data paths provided to training scripts are correct.

---

**Note**: This project was developed and tested on Ubuntu 22.04 with Hadoop 3.3.5 in a single-node configuration. For multi-node setups or other environments, additional configuration may be required.
