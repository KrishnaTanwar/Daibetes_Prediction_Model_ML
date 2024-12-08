# Diabetes Prediction Model

## Overview
A machine learning model using Support Vector Machine (SVM) to predict diabetes risk. The model achieves 78% accuracy in predicting whether a person has diabetes based on various health metrics.

## Project Structure
- `app.py`: Main application file containing the model implementation
- `diabetes.csv`: Dataset containing health metrics
- `requirements.txt`: List of Python dependencies

## Dataset Features
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
- Outcome (Target Variable)

## Requirements
- Python 3.x
- NumPy
- Pandas
- scikit-learn

## Installation
1. Clone this repository
```bash
git clone [your-repo-link]
```

2. Install required packages
```bash
pip install -r requirements.txt
```

## Usage
Run the main application:
```bash
python app.py
```

The model will:
1. Load and preprocess the data
2. Train the SVM classifier
3. Display model accuracy
4. Make a sample prediction

## Model Performance
- Model Accuracy: 78%
- Algorithm: Support Vector Machine (SVM)
- Kernel: Linear

## License
MIT License

## Acknowledgments
Special thanks to mentor Devendra for guidance and support throughout this project.
