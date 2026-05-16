# My-Fifth-Repository-1
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from utils import preprocess_text

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("dataset.csv")

print("\nDataset Loaded Successfully!\n")
print(df.head())

# ==============================
# Preprocess Text
# ==============================

df['clean_text'] = df['text'].apply(preprocess_text)

# ==============================
# Features & Labels
# ==============================

X = df['clean_text']
y = df['sentiment']

# ==============================
# Convert Text to Numerical Data
# ==============================

vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# ==============================
# Model Building
# ==============================

model = LogisticRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==============================
# Predictions
# ==============================

y_pred = model.predict(X_test)

# ==============================
# Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"\nAccuracy: {accuracy:.2f}")

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ==============================
# Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

# ==============================
# Save Model
# ==============================

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel & Vectorizer Saved!")

# ==============================
# Real-Time Prediction System
# ==============================

while True:

    print("\n==============================")
    print("SENTIMENT PREDICTION SYSTEM")
    print("==============================")

    user_input = input("\nEnter your sentence (or type 'exit'): ")

    if user_input.lower() == 'exit':
        print("\nProgram Exited Successfully!")
        break

    # Preprocess input
    cleaned_input = preprocess_text(user_input)

    # Convert input to vector
    vectorized_input = vectorizer.transform([cleaned_input])

    # Prediction
    prediction = model.predict(vectorized_input)[0]

    # Probability Scores
    probabilities = model.predict_proba(vectorized_input)[0]

    print("\n==============================")
    print(f"Predicted Sentiment: {prediction}")
    print("==============================")

    print("\nPrediction Probabilities:\n")

    for label, prob in zip(model.classes_, probabilities):
        print(f"{label}: {prob:.2f}")