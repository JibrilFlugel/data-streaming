# MNIST Data Streaming

## Prerequisites

- Python 3.8+.
- Apache Spark 3.5.5

## Setup

### 1. Install Python Dependencies
Install required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Generate MNIST Batch Files
Run the script to generate MNIST batch files if not already present:
```bash
python mnist_download.py
```

### 3. Check Port 6100
Verify no processes are using port 6100:
```bash
netstat -a -n -o | find "6100"
```
If a process is listed, terminate it:
```bash
taskkill /PID <pid> /F
```

## Running the pipeline

The pipeline consists of two main scripts, running in 2 different terminals:
- `stream.py`: Streams MNIST batches over TCP (port 6100).
- `main.py`: Runs the Spark Streaming job to process the received batches.

### Step 1: Start the Streaming Server
In the first terminal, run:
```bash
python stream.py --folder mnist_batches --batch-size 10000
```

**Notes:**
- `--folder`: Directory containing the MNIST batches. (Required)
- `--batch-size`: Number of images per batch. (Required)
- `--sleep`: Seconds between batches. (Default is '3')
- `--split`: Train/test split. (Default is 'train')
- `--endless`: Enable/disable endless stream. (Default is 'False')
- The script processes 60,000 training images across 6 batch files.

### Step 2: Run the Spark Job
In the second terminal, run:
```bash
spark-submit main.py
```
