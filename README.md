# Final Project Machine Learning

## Overview
This project implements explainable AI techniques to make machine learning models more interpretable. The application uses both traditional machine learning (KNN) and deep learning (ResNet50v2) approaches with visualization tools to explain model predictions.

## Project Structure
```
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── data/                       # Dataset directory
│   ├── test.csv                # Test dataset
│   ├── train.csv               # Training dataset
│   ├── valid.csv               # Validation dataset
│   └── cat_dog/                # Image dataset for classification
├── models/                     # Saved model files
│   ├── knn_chp_model.pkl       # KNN model with CHPs
│   ├── knn_chp_norm_model.pkl  # Normalized KNN model with CHPs
│   └── resnet50v2.h5           # ResNet50v2 deep learning model
├── utils/                      # Utility modules
│   ├── explainer.py            # Model explanation utilities
│   └── visualization.py        # Visualization tools
└── website/                    # Web interface components
    ├── static/                 # Static assets (CSS, JS, images)
    └── templates/              # HTML templates
```

## Features
- Interactive web interface for model explanation
- Support for both traditional ML and deep learning models
- Visual explanations of model predictions
- Comparison between different models and techniques

## Installation
1. Clone the repository
    ```bash
    git clone https://github.com/VuTrinhNguyenHoang/Final-Project-Machine-Learning.git
    cd Final-Project-Machine-Learning
    ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    # Trên macOS/Linux:
    source venv/bin/activate
    # Trên Windows:
    venv\Scripts\activate
    ```

3. Install required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage
Start the Flask application:
```
python app.py
```
Then open your web browser and navigate to http://localhost:5000

## Supported Models
- KNN with Custom Hyperplane Projections
- KNN with Normalized Custom Hyperplane Projections
- ResNet50v2 for image classification

## Explanation Methods
This project implements various explanation techniques including:
- Feature importance visualization
- Local interpretable model-agnostic explanations
- Visual explanations for image classification

## Web Interface
The web interface provides:
- Model selection options
- Interactive visualizations
- Comparison of different explanation techniques
- Example-based exploration of model behavior

## Requirements
See `requirements.txt` for a full list of dependencies.