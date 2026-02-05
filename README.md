# Customer Purchase Predictor

**Predict Customer Needs, Boost Sales Instantly**

A **Streamlit web application** that uses machine learning to predict whether an online shopper is likely to complete a purchase based on their browsing behavior.

## Project Overview

This project analyzes online shopping session data (such as page views, time spent, bounce rates, product interactions, etc.) and applies a trained classification model to determine purchase probability in real-time.

Main goal: Help e-commerce businesses identify high-intent visitors and optimize marketing/sales strategies.

### Key Features
- Interactive web interface built with **Streamlit**
- Real-time purchase prediction using pre-trained ML model
- Clean separation of data exploration, model training, and deployment
- Easy to run locally and potentially deploy on cloud platforms

## Tech Stack

| Component              | Technology/Tools used                     |
|------------------------|-------------------------------------------|
| Programming Language   | Python 3.10+                              |
| Web Framework          | Streamlit                                 |
| Data Processing        | pandas, numpy                             |
| Machine Learning       | scikit-learn                              |
| Model Serialization    | joblib                                    |
| Development Environment| Jupyter Notebook (exploration & training) |
| Dependency Management  | requirements.txt                          |

## Project Structure
Customer-Purchase-Predictor/
├── Data/                   # Dataset files (CSV, etc.)
├── Notebooks/              # Jupyter notebooks for EDA, model training & evaluation
├── src/                    # Source code (preprocessing, utils, model helpers)
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
└── .gitignore              # Git ignore file


## Prerequisites

- Python 3.10 or higher
- pip (package installer)
- Git (to clone the repository)

