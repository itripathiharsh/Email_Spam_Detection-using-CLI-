#!/usr/bin/env python3
"""
Phishing Email Detector - A machine learning tool to classify emails as legitimate or phishing.

This script provides functionality to:
1. Train and save a model using a labeled dataset
2. Load an existing model for quick predictions
3. Process and classify email text via command line or interactive input
"""

import argparse
import sys
import re
import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, List, Optional
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "phishing_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
DEFAULT_DATASET_PATH = Path("Dataset") / "Phishing_Email.csv"


def clean_text(text: str) -> str:
    """
    Clean and normalize email text for processing.
    
    Args:
        text: Raw email text
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""  # Handle NaN/empty values
    
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' emailaddress ', text)
    # Replace mentions/hashtags
    text = re.sub(r'\@\w+', ' mention ', text)
    text = re.sub(r'\#\w+', ' hashtag ', text)
    # Replace numbers with placeholder
    text = re.sub(r'\d+', ' number ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text.strip()


def load_and_preprocess_data(file_path: Path) -> Tuple[pd.Series, pd.Series]:
    """
    Load the dataset and preprocess it for training.
    
    Args:
        file_path: Path to the CSV dataset
        
    Returns:
        Tuple of (features, labels)
    """
    logger.info(f"Loading dataset from {file_path}")
    
    try:
        data = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Validate required columns
    required_cols = ["Email Text", "Email Type"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    # Drop rows with missing values in critical columns
    initial_rows = len(data)
    data.dropna(subset=required_cols, inplace=True)
    dropped_rows = initial_rows - len(data)
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with missing values")
    
    # Clean text and create labels
    logger.info("Preprocessing email text")
    data['Cleaned_Email'] = data['Email Text'].apply(clean_text)
    
    # Create binary labels
    label_map = {'Safe Email': 0, 'Phishing Email': 1}
    # Check if all values in Email Type are in our map
    unknown_types = set(data['Email Type'].unique()) - set(label_map.keys())
    if unknown_types:
        logger.warning(f"Found unknown email types: {unknown_types}. These will be excluded.")
        data = data[data['Email Type'].isin(label_map.keys())]
    
    data['Label'] = data['Email Type'].map(label_map)
    
    # Return features and labels
    X = data['Cleaned_Email']
    y = data['Label']
    
    logger.info(f"Dataset processed: {len(X)} samples, {sum(y)} phishing, {len(X) - sum(y)} legitimate")
    return X, y


def create_pipeline() -> Pipeline:
    """
    Create an ML pipeline with TF-IDF vectorizer and classifier.
    
    Returns:
        sklearn Pipeline object
    """
    return Pipeline([
        ('vectorizer', TfidfVectorizer(
            stop_words='english',
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 2),
            max_features=10000
        )),
        ('classifier', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        ))
    ])


def train_model(
    data_path: Path = DEFAULT_DATASET_PATH,
    optimize: bool = False,
    save_model: bool = True
) -> Tuple[Pipeline, dict]:
    """
    Train a phishing detection model.
    
    Args:
        data_path: Path to the training dataset
        optimize: Whether to perform hyperparameter tuning
        save_model: Whether to save the trained model to disk
        
    Returns:
        Tuple of (trained pipeline, performance metrics)
    """
    # Ensure model directory exists
    if save_model:
        MODEL_DIR.mkdir(exist_ok=True, parents=True)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Optimize hyperparameters if requested
    if optimize:
        logger.info("Performing hyperparameter optimization")
        param_grid = {
            'vectorizer__max_df': [0.9, 0.95],
            'vectorizer__min_df': [2, 5],
            'vectorizer__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 1.0, 10.0],
        }
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        pipeline = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
    else:
        # Train the model
        logger.info("Training model")
        pipeline.fit(X_train, y_train)
    
    # Evaluate
    logger.info("Evaluating model")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    # Print evaluation results
    logger.info("\nModel Evaluation:")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    logger.info("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model if requested
    if save_model:
        logger.info(f"Saving model to {MODEL_PATH}")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(pipeline, f)
    
    return pipeline, metrics


def load_model() -> Optional[Pipeline]:
    """
    Load a previously trained model from disk.
    
    Returns:
        Loaded model pipeline or None if file not found
    """
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        logger.warning(f"Could not load model: {e}")
        return None


def predict_message(message: str, pipeline: Optional[Pipeline] = None) -> Tuple[str, float]:
    """
    Predict if a message is phishing or legitimate.
    
    Args:
        message: Email text to classify
        pipeline: Trained pipeline (will load from disk if None)
        
    Returns:
        Tuple of (prediction label, confidence score)
    """
    # Load model if not provided
    if pipeline is None:
        pipeline = load_model()
        if pipeline is None:
            logger.error("No model available. Please train a model first.")
            return "Unknown", 0.0
    
    # Clean and predict
    cleaned = clean_text(message)
    if not cleaned:
        logger.warning("Empty message after cleaning")
        return "Unknown", 0.0
    
    # Get prediction and probability
    prob = pipeline.predict_proba([cleaned])[0, 1]  # Probability of phishing class
    prediction = "Phishing" if prob >= 0.5 else "Legitimate"
    confidence = max(prob, 1-prob)  # Convert to confidence 0.5-1.0
    
    return prediction, confidence


def print_prediction_result(prediction: str, confidence: float) -> None:
    """
    Print prediction results in a user-friendly format.
    
    Args:
        prediction: The predicted class
        confidence: Confidence score (0-1)
    """
    confidence_pct = confidence * 100
    
    if prediction == "Phishing":
        if confidence >= 0.9:
            risk_level = "HIGH RISK"
        elif confidence >= 0.7:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"
        
        print(f"\n⚠️  PREDICTION: {prediction} Email ({risk_level})")
    else:
        if confidence >= 0.9:
            trust_level = "HIGH CONFIDENCE"
        elif confidence >= 0.7:
            trust_level = "MEDIUM CONFIDENCE"
        else:
            trust_level = "LOW CONFIDENCE"
            
        print(f"\n✓ PREDICTION: {prediction} Email ({trust_level})")
    
    print(f"Confidence: {confidence_pct:.1f}%")


def main():
    """Main function to handle command-line arguments and program flow."""
    parser = argparse.ArgumentParser(
        description="Phishing Email Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model:
  python phishing_detector.py --train
  
  # Train with hyperparameter optimization:
  python phishing_detector.py --train --optimize
  
  # Classify a single-line email:
  python phishing_detector.py --message "Dear user, click here to verify your account"
  
  # Interactive mode (paste multi-line text):
  python phishing_detector.py
"""
    )
    
    parser.add_argument(
        "-m", "--message", 
        type=str, 
        help="Single-line email text to classify"
    )
    parser.add_argument(
        "-t", "--train", 
        action="store_true", 
        help="Train a new model"
    )
    parser.add_argument(
        "-o", "--optimize", 
        action="store_true", 
        help="Perform hyperparameter optimization during training"
    )
    parser.add_argument(
        "-d", "--dataset", 
        type=str, 
        help=f"Path to training dataset (default: {DEFAULT_DATASET_PATH})"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Set logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Process according to arguments
    pipeline = None
    
    # Training mode
    if args.train:
        dataset_path = Path(args.dataset) if args.dataset else DEFAULT_DATASET_PATH
        pipeline, _ = train_model(
            data_path=dataset_path,
            optimize=args.optimize,
            save_model=True
        )
    
    # If not training or if also classifying a message
    if not args.train or args.message:
        if not args.train:
            pipeline = load_model()
        
        # Classification mode
        if args.message:
            prediction, confidence = predict_message(args.message, pipeline)
            print_prediction_result(prediction, confidence)
        else:
            # Interactive mode - if no message provided and not in training mode
            if not args.train:
                print("Enter your email text (press Ctrl+D or Ctrl+Z on a new line to finish):")
                try:
                    message = sys.stdin.read()
                    if message.strip():
                        prediction, confidence = predict_message(message, pipeline)
                        print_prediction_result(prediction, confidence)
                    else:
                        print("No input provided.")
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        sys.exit(1)