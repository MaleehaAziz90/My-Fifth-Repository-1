# Full Source Code

---

# dataset.csv

```csv
text,sentiment
"I love this product","Positive"
"This is absolutely amazing","Positive"
"Very happy with the service","Positive"
"The experience was fantastic","Positive"
"I enjoyed using this app","Positive"
"Great performance and smooth interface","Positive"
"This made my day","Positive"
"I highly recommend this","Positive"
"The quality is excellent","Positive"
"Wonderful customer support","Positive"
"I hate this application","Negative"
"Worst experience ever","Negative"
"This is terrible","Negative"
"I am very disappointed","Negative"
"The service was awful","Negative"
"Very bad quality","Negative"
"I regret buying this","Negative"
"This app crashes a lot","Negative"
"Not satisfied at all","Negative"
"Extremely frustrating experience","Negative"
"It is okay","Neutral"
"Nothing special about it","Neutral"
"The app works as expected","Neutral"
"I have no strong opinion","Neutral"
"It is average","Neutral"
"The product is fine","Neutral"
"Neither good nor bad","Neutral"
"The experience was normal","Neutral"
"It functions properly","Neutral"
"This is acceptable","Neutral"
"I love the design but hate the speed","Mixed"
"The interface is beautiful but performance is poor","Mixed"
"Good features but terrible support","Mixed"
"I like the app although it crashes sometimes","Mixed"
"The quality is nice but delivery was late","Mixed"
"Excellent concept but bad execution","Mixed"
"I enjoyed the movie but the ending was awful","Mixed"
"The food was tasty but service was slow","Mixed"
"I like the camera but battery life is bad","Mixed"
"The product is useful but expensive","Mixed"
```

---

# utils.py

```python
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)
```

---

# model.py

```python
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

# Load dataset
df = pd.read_csv("dataset.csv")

# Preprocess dataset
df['clean_text'] = df['text'].apply(preprocess_text)

# Features and Labels
X = df['clean_text']
y = df['sentiment']

# Convert text into vectors
vectorizer = TfidfVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Confusion Matrix
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

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully!")

# Prediction System
while True:

    user_input = input(
        "\nEnter your sentence (or type 'exit'): "
    )

    if user_input.lower() == 'exit':
        print("Program exited.")
        break

    cleaned_input = preprocess_text(user_input)

    vectorized_input = vectorizer.transform(
        [cleaned_input]
    )

    prediction = model.predict(
        vectorized_input
    )[0]

    probabilities = model.predict_proba(
        vectorized_input
    )[0]

    print(
        f"\nPredicted Sentiment: {prediction}"
    )

    print("\nPrediction Probabilities:\n")

    for label, prob in zip(
        model.classes_,
        probabilities
    ):
        print(f"{label}: {prob:.2f}")
```

---

# requirements.txt

```txt
pandas
scikit-learn
nltk
matplotlib
seaborn
joblib
```