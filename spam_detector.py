import argparse
import sys
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95)
model = LogisticRegression()


# preprocessing
def clean_text(text):
    if not isinstance(text, str):
        return ""  # Handle NaN/empty values
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions/hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and whitespace
    return text.strip()


# Train Model
def train_model():
    data = pd.read_csv("Dataset/Phishing_Email.csv")  # Update path if needed
    data.dropna(subset=['Email Text', 'Email Type'], inplace=True)  # Drop rows with missing values
    data['Email Text'] = data['Email Text'].apply(clean_text)

    data['Label'] = data['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})

    X = data['Email Text']
    y = data['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model.fit(X_train_tfidf, y_train)

    # Testing
    y_pred = model.predict(X_test_tfidf)
    # print("\nModel Evaluation:\n")
    # print(classification_report(y_test, y_pred))
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))


def predict_message(message):
    cleaned = clean_text(message)
    message_tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(message_tfidf)
    return "Phishing" if prediction[0] == 1 else "Legit"  # Updated output


def main():
    parser = argparse.ArgumentParser(description="Phishing Email Detector CLI")
    parser.add_argument("-m", "--message", type=str, help="Single-line email text to classify")
    args = parser.parse_args()

    train_model()

    if args.message:
        result = predict_message(args.message)
        print(f"\nPrediction: {result}")
    else:
        print("Enter your multi-line email (press Ctrl+D or Ctrl+Z to finish):")
        message = sys.stdin.read()
        result = predict_message(message)
        print(f"\nPrediction: {result}")


if __name__ == "__main__":
    main()