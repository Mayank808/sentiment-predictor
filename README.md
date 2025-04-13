# Sentiment Predictor

## Overview

The Sentiment Predictor project leverages machine learning to analyze and predict sentiment from input data. It includes several components that streamline data analysis, model training, and prediction generation. Developed and fine‑tuned deep learning, ensemble, and random forest models using TensorFlow and NLTK, the system is specifically designed to predict employee Glassdoor review ratings.

## Components

- **Data:**  
  The [`data`](data/) folder contains training and testing datasets (e.g., `PC_small_train_v1.csv` and `PC_test_without_response_v1.csv`) as well as prediction outputs in CSV format.

- **Model Development & Tuning:**  
  The project contains various experiments:
  - The main prediction and evaluation scripts/notebooks (`pcFinal.py`, `pcFinal.ipynb`, and `pcFinal.html`) run the sentiment prediction workflow.
  - The [`model_testing`](model_testing/) directory includes the model architecture visualization (`model_structure.png`), experimental notebooks such as [`testing_different_model.ipynb`](model_testing/testing_different_model.ipynb), and tuner output in the [`tuner_dir/rating_prediction`](model_testing/tuner_dir/rating_prediction/) folder which contains tuning trials and checkpoints.

## How to Use

1. **Data Preparation:**  
   Ensure the datasets in the [`data/`](data/) folder are up-to-date. Modify or replace CSV files as needed.

2. **Model Training & Evaluation:**

   - Run `pcFinal.py` or open `pcFinal.ipynb` to execute the main prediction pipeline.
   - Use `pcFinal.html` to view the model’s performance and outputs in a web/browser-friendly format.
   - Explore experiments in the [`model_testing`](model_testing/) directory to review alternative models and tuning strategies.

3. **Model Tuning:**  
   The tuning process is managed in the [`tuner_dir/rating_prediction`](model_testing/tuner_dir/rating_prediction/) folder where experiment logs, trial checkpoints, and configuration files (e.g., `oracle.json`, `tuner0.json`) are stored.

## Additional Information

- **Architecture:**  
  The overall model structure is visualized in both the root and [`model_testing/model_structure.png`](model_testing/model_structure.png).
