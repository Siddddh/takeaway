# Real-time Fraud Detection System

A real-time fraud detection system that monitors financial transactions and identifies potential fraudulent activities using machine learning.

## Features

- Real-time transaction monitoring
- Anomaly detection using Isolation Forest
- Interactive dashboard with Streamlit
- Transaction simulation
- Fraud alerts and statistics
- Model training and evaluation
- Data export functionality

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Use the sidebar controls to:
   - Adjust simulation speed
   - Set fraud detection threshold
   - Train the model
   - Export transaction data

## Project Structure

```
fraud_detection/
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── models/            # Saved models
└── README.md          # Project documentation
```

## Technical Details

- Uses Isolation Forest for anomaly detection
- Implements real-time transaction simulation
- Provides interactive visualizations
- Supports model training and evaluation
- Includes data export functionality

## Future Improvements

- Add more sophisticated fraud detection algorithms
- Implement real-time data streaming
- Add more transaction features
- Improve visualization capabilities
- Add user authentication 