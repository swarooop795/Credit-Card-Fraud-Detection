# Credit-Card-Fraud-Detection

# Purpose :
To build a web interface where users can input transaction features, and the system predicts if the transaction is fraudulent or legitimate using a trained ML model.

# 🧠 Project Motivation

Credit card fraud poses a significant threat in the digital payment ecosystem. With the increase in online and card-not-present transactions, the frequency and sophistication of fraudulent activities have also risen. Detecting these rare fraudulent transactions quickly and accurately is essential to protect both users and financial institutions.

This project was developed to:

Enable detection of fraudulent transactions using machine learning.

Provide an intuitive web interface for non-technical users to interact with the detection system.

Allow real-time analysis of uploaded transaction datasets.

The goal was to make fraud detection simple, interactive, and usable for small organizations or educational demonstrations.

# 💻 Frontend Technologies Used

The frontend is designed using HTML, Bootstrap, and Flask templating. It includes:

A responsive homepage with a background image and call-to-action button.

An upload page for submitting CSV files.

A results page that displays statistics like total, fraudulent, and genuine transactions with styled components.

All pages are built using render_template_string() and embedded HTML inside the Python script (nairy.py).

# 🛠 Backend Technologies Used
The backend is powered by:

Python (Flask Framework) for handling web routes and logic.

Pandas for data handling and CSV parsing.

Scikit-learn for:

Data preprocessing (StandardScaler)

Model training (LogisticRegression)

# 🔧 Key Components
# 1. Imports

from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

Flask: For web framework.

Pandas: For data handling.

Sklearn modules: For model training, scaling, and evaluation.

# 2. Flask App Setup

app = Flask(_name_)
model = None
scaler = None

Initializes Flask app and global variables for ML model and scaler.

# 3. Web UI (HTML Template)
   
HTML content is stored in Python as a string (HOME_TEMPLATE) with:

Bootstrap CSS for styling

Background image

Likely a form to input transaction values

# 🔍 Overview

This app lets users:

Access a web interface to start the fraud detection process.

Upload a CSV file of transaction data.

Train a Logistic Regression model on uploaded data.

View predictions of fraudulent vs genuine transactions

# 🌐 Web Routes & Functionality

@app.route("/") → Home Page
Renders a beautiful landing page using HOME_TEMPLATE.

Button: "Start Now" → /start

@app.route("/start") → Upload Page
Displays an upload form using UPLOAD_TEMPLATE.

Accepts a .csv file containing transaction data.

@app.route("/predict", methods=["POST"]) → Prediction Logic
When user uploads a CSV:

Loads it using pandas.read_csv().

Checks for required column Class (1 = fraud, 0 = genuine).

Splits data: train_test_split().

Scales features using StandardScaler.

Trains a Logistic Regression model on-the-fly.

Predicts on the test set.

Generates a classification report using sklearn.metrics.

Extracts:

Number of fraudulent and genuine transactions

Fraud % in the test set

Displays results using RESULTS_TEMPLATE.

# ⚙ Challenges Faced

Class Imbalance: Fraudulent transactions are rare (~0.1%), which made training effective models difficult without oversampling or advanced techniques.

Model Efficiency: The system retrains the model on every CSV upload, which is not efficient for real-time use.

User Interface Integration: Building a visually clean and responsive Flask-based web app while ensuring backend predictions worked correctly was a key hurdle.

Error Handling: Ensuring the system gracefully handled invalid inputs, missing columns, or bad file formats was crucial to usability

Evaluation (classification_report)

Logistic Regression model is trained on-the-fly using uploaded data.

To Access this Project : python nairy.py

Classification Report : python report.py 
