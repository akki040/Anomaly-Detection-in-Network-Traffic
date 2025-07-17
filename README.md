Network Traffic Anomaly Detection
This project is a web-based tool that helps detect unusual or potentially harmful network activity using machine learning. It is designed to be beginner-friendly and easy to use, even if you're new to machine learning.

What This App Does
Lets you input details about a network connection (like protocol type, service, bytes sent/received).

Uses two models — Isolation Forest and Autoencoder — to analyze the input.

Shows you whether the input is Normal or an Anomaly.

Models Used
Isolation Forest
A tree-based algorithm that isolates anomalies by randomly selecting features and splitting values.

Autoencoder
A neural network trained to reproduce normal data. If it fails to do so for new input, it likely indicates an anomaly.

Dataset Used
We use the KDD Cup 1999 Dataset which is widely used for benchmarking intrusion detection systems.

You can download the dataset from this link:
=> https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data

Please use the file named kddcup.csv in the project.

How to Run the Project
1. Clone the repository

git clone https://github.com/your-username/network-anomaly-detector.git
cd network-anomaly-detector
2. Add the dataset
Download kddcup.csv from the Kaggle dataset link and place it in the root directory of the project.

3. Install the required Python libraries
Make sure Python 3.9 or later is installed. Then run:

pip install -r requirements.txt


4. Run the app

streamlit run app.py
Your browser will open with the interactive application.

Project Structure

network-anomaly-detector

app.py               # Streamlit app script
kddcup.csv           # Dataset file
requirements.txt     # All required libraries with versions
README.md            # You're reading this!


Required Python Libraries
Your requirements.txt should include:


streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
tensorflow==2.16.1
You can update these if you use different versions.

Why This Matters
Detecting anomalies in network traffic is important for spotting cyberattacks like DDoS, port scans, or intrusions. This app is a small but powerful example of how machine learning can help automate that detection.

Acknowledgements
- KDD Cup 1999 dataset (via Kaggle)
- scikit-learn for Isolation Forest
- TensorFlow/Keras for Autoencoder

Streamlit for building the interactive web app
