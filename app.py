import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers


# Page config & mappings

st.set_page_config(page_title=" Anomaly Detector", layout="wide")
st.title(" Network Traffic Anomaly Detection")

protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2}
service_map = {'http': 0, 'ftp': 1, 'smtp': 2, 'domain_u': 3}
flag_map = {'SF': 0, 'REJ': 1, 'RSTO': 2}


#  Load & train models

@st.cache_resource
def load_and_train():
    df = pd.read_csv("kddcup.csv")

    # Drop known non-numeric
    X = df.drop(['label', 'true_label', 'attack', 'anomaly'], axis=1, errors='ignore')

    # Map text to int
    X['protocol_type'] = X['protocol_type'].map(protocol_map)
    X['service'] = X['service'].map(service_map)
    X['flag'] = X['flag'].map(flag_map)

    assert all(X.dtypes != 'object'), " Non-numeric columns remain!"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest()
    iso.fit(X_scaled)

    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(14, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(X_scaled, X_scaled, epochs=5, batch_size=32, shuffle=True, verbose=0)

    mse_train = np.mean(np.power(X_scaled - autoencoder.predict(X_scaled), 2), axis=1)
    threshold = np.percentile(mse_train, 80)

    return iso, autoencoder, scaler, threshold, X.columns.tolist()

iso, autoencoder, scaler, threshold, columns = load_and_train()


#  User input with more features

st.write("##  Enter New Network Traffic Data")

protocol_type = st.selectbox("Protocol Type", list(protocol_map.keys()))
service = st.selectbox("Service", list(service_map.keys()))
flag = st.selectbox("Flag", list(flag_map.keys()))

duration = st.number_input("Duration", min_value=0)
src_bytes = st.number_input("Source Bytes", min_value=0)
dst_bytes = st.number_input("Destination Bytes", min_value=0)
count = st.number_input("Connection Count (count)", min_value=0)
srv_count = st.number_input("Service Count (srv_count)", min_value=0)
serror_rate = st.number_input("SYN Error Rate (serror_rate)", min_value=0.0, max_value=1.0, value=0.0)
srv_serror_rate = st.number_input("Service SYN Error Rate (srv_serror_rate)", min_value=0.0, max_value=1.0, value=0.0)


# Predict

if st.button(" Predict"):
    input_data = {
        'protocol_type': protocol_map[protocol_type],
        'service': service_map[service],
        'flag': flag_map[flag],
        'duration': duration,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'count': count,
        'srv_count': srv_count,
        'serror_rate': serror_rate,
        'srv_serror_rate': srv_serror_rate,
    }

    for col in columns:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])[columns]
    input_scaled = scaler.transform(input_df)

    iso_result = iso.predict(input_scaled)[0]
    iso_anomaly = " Anomaly" if iso_result == -1 else " Normal"

    ae_pred = autoencoder.predict(input_scaled)
    mse = np.mean(np.power(input_scaled - ae_pred, 2))
    ae_anomaly = " Anomaly" if mse > threshold else " Normal"

    st.write("##  Results")
    st.write(f"**Isolation Forest:** {iso_anomaly}")
    st.write(f"**Autoencoder:** {ae_anomaly}")
    st.write(f"**MSE:** {mse:.6f} vs Threshold: {threshold:.6f}")

st.info(" Tip: Use realistic values for best results!")
