# Network Traffic Anomaly Detection

This project is a web-based tool that helps detect unusual or potentially harmful network activity using machine learning. It is designed to be beginner-friendly and easy to use, even if you're new to machine learning.

##What This App Does

- Lets you input details about a network connection (like protocol type, service, bytes sent/received).
- Uses two models — Isolation Forest and Autoencoder — to analyze the input.
- Shows you whether the input is Normal or an Anomaly.

## Models Used

### Isolation Forest
A tree-based algorithm that isolates anomalies by randomly selecting features and splitting values.

### Autoencoder
A neural network trained to reproduce normal data. If it fails to do so for new input, it likely indicates an anomaly.

## Dataset Used

We use the **KDD Cup 1999 Dataset**, which is widely used for benchmarking intrusion detection systems.

You can download the dataset from this link:  
**https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data**

Please use the file named **kddcup.csv** in the project.

---

## How to Run the Project

### 1. Clone the repository


git clone https://github.com/your-username/network-anomaly-detector.git
cd network-anomaly-detector
