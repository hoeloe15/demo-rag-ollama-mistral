#!/bin/bash

# Start the backend
echo "Starting backend..."
nohup python backend.py &

# Wait for the backend to start
sleep 10

# Start the Streamlit frontend
echo "Starting Streamlit frontend..."
streamlit run app.py
