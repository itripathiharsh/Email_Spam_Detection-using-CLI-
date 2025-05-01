# Phishing Email Detector

A machine learning-based CLI tool to classify emails as **Phishing** or **Legit** using Logistic Regression and TF-IDF vectorization.

## Features
- Classifies email text as `Phishing` or `Legit`
- Supports both single-line and multi-line input
- Preprocesses text by removing URLs, special characters, and stopwords
- Evaluates model performance (accuracy, precision, recall)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd phishing-email-detector

2. pip install pandas scikit-learn

3. Usage:
       Single-line Input -- python phishing_detector.py -m "Your email text here"
       Multiline Input -- python phishing_detector.py
                          Enter your multi-line email (press Ctrl+D or Ctrl+Z to finish):
                          Dear User,
                          Your account has been locked. Verify now:
                          http://fake-login.com
                          ^D
   
