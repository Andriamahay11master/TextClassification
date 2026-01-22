import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=["label", "text"]
)

# Convert labels to numbers
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# Vectorize text
vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
)

X = vectorizer.fit_transform(data["text"])
y = data["label_num"]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Predict new messages
# -------------------------------

def predict_message(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    if prediction == 1:
        return "SPAM"
    else:
        return "HAM (Not Spam)"


# Test messages
test_messages = [
    "Congratulations! You have won a free ticket",
    "Hey, are we still meeting at 5?",
    "URGENT! Claim your prize now",
    "Can you send me the homework?"
]

for msg in test_messages:
    print(f"Message: {msg}")
    print("Prediction:", predict_message(msg))
    print("-" * 40)

