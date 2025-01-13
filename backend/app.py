from flask import Flask, jsonify, request
from flask_cors import CORS  # Import CORS
import numpy as np
from IDS import IntrusionDetector
from flask_socketio import SocketIO, emit
from threading import Thread
from collections import deque
import socket

# Initialize Flask and SocketIO
app = Flask(__name__)

# Enable CORS for all routes (and allow specific origins)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Allow React app to access API
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")  # Allow WebSocket connections from React

# Configuration
VM_IP = '172.22.51.186'  # Replace with your VM’s IP address
PORT = 60002            # Port that you set in Cooja
MAX_DATA_POINTS = 50    # Max number of data points to store

# Store last 50 data points
data_buffer = deque(maxlen=MAX_DATA_POINTS)

def receive_data():
    """Connects to the VM and receives data continuously."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # Now the socket module is available
        s.connect((VM_IP, PORT))
        print(f"Connected to Cooja on {VM_IP}:{PORT} for live data.")
        
        while True:
            data = s.recv(1024)  # Receive data in chunks of 1024 bytes
            if not data:
                break  # Exit loop if connection is closed
            
            decoded_data = data.decode('utf-8')  # Decode data to string
            data_buffer.append(decoded_data)  # Store data in buffer
            print("Received data:", decoded_data)

            # Emit data to WebSocket clients
            socketio.emit('new_data', {'data': decoded_data})

# Start a separate thread to fetch data from the VM
def start_data_receiver():
    data_thread = Thread(target=receive_data)
    data_thread.daemon = True  # Set thread as daemon so it exits when the main program exits
    data_thread.start()

@app.route('/latest-data', methods=['GET'])
def get_latest_data():
    """HTTP endpoint to get the last 50 data points."""
    return jsonify(list(data_buffer))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    x = np.array(data)
    if x.size == 1:
        x = np.array(x)
    intrusion = IntrusionDetector(model_path="intrusion-detector.pkl")
    pred = intrusion.predict(x, get_class_names=True)
    return jsonify(pred)

if __name__ == "__main__":
    # Start data receiver in a separate thread when the server starts
    start_data_receiver()
    
    # Run the Flask server
    socketio.run(app, debug=True, port=5000)
