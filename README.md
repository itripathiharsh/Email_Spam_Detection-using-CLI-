# Phishing Email Detector

A machine learning-based command-line tool to classify emails as legitimate or phishing attempts.

## Features

-   Detect phishing emails with machine learning
-   Train custom models on your own datasets
-   Pre-trained model for immediate use
-   Interactive or command-line operation
-   Confidence scores and risk assessment
-   Hyperparameter optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start Guide

### Classify an Email

```bash
# Using a pre-trained model (if available)
python phishing_detector.py --message "Dear user, click here to verify your account information at http://suspicious-link.com"
```

### Train a New Model

```bash
# Train using default dataset
python phishing_detector.py --train

# Train with hyperparameter optimization
python phishing_detector.py --train --optimize

# Train using a custom dataset
python phishing_detector.py --train --dataset path/to/your/dataset.csv
```

### Interactive Mode

```bash
# Start interactive mode
python phishing_detector.py

# Then paste or type your email content
# Press Ctrl+D (Unix/Mac) or Ctrl+Z followed by Enter (Windows) when finished
```

## Dataset Format

The training dataset should be a CSV file with at least these columns:

-   `Email Text`: The content of the email
-   `Email Type`: Classification label ("Safe Email" or "Phishing Email")

Example:

```csv
Email Text,Email Type
"Dear customer, your account needs verification...",Phishing Email
"Hi John, here are the meeting notes we discussed...",Safe Email
```

## Command Line Arguments

```
usage: phishing_detector.py [-h] [-m MESSAGE] [-t] [-o] [-d DATASET] [-v]

Phishing Email Detector

options:
  -h, --help            show this help message and exit
  -m MESSAGE, --message MESSAGE
                        Single-line email text to classify
  -t, --train           Train a new model
  -o, --optimize        Perform hyperparameter optimization during training
  -d DATASET, --dataset DATASET
                        Path to training dataset

Examples:
  # Train a new model:
  python phishing_detector.py --train

  # Train with hyperparameter optimization:
  python phishing_detector.py --train --optimize

  # Classify a single-line email:
  python phishing_detector.py --message "Dear user, click here to verify your account"

  # Interactive mode (paste multi-line text):
  python phishing_detector.py
```

## How It Works

1. **Text Preprocessing**:

    - Cleans email text by removing URLs, standardizing formatting
    - Replaces emails, mentions, and numbers with placeholders
    - Removes special characters while preserving meaning

2. **Feature Extraction**:

    - Uses TF-IDF vectorization to convert text to numerical features
    - Includes unigrams and bigrams for context
    - Applies English stopword removal

3. **Classification**:
    - Logistic Regression with balanced class weights
    - Provides both classification and confidence scores
    - Risk assessment based on confidence levels

## Dependencies

-   Python 3.7+
-   scikit-learn
-   pandas
-   numpy

## File Structure

```
Email_Spam_Detection-using-CLI-/
├── spam_detector.py         # Main script
├── models/                  # Directory for saved models
│   └── phishing_model.pkl   # Saved model file
├── Dataset/                 # Directory for datasets
│   ├── Phishing_Email.csv   # Default dataset
│   └── Phishing_Email.rar   # Compressed dataset
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
