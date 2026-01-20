import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -------------------------------
# Load dataset
# -------------------------------

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=["label", "text"]
)

data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# -------------------------------
# TF-IDF Vectorization (IMPROVED)
# -------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)  # single words + word pairs
)

X = vectorizer.fit_transform(data["text"])
y = data["label_num"]

# -------------------------------
# Train / Test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train model
# -------------------------------

model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model accuracy with TF-IDF:", accuracy)

# -------------------------------
# Prediction function
# -------------------------------

def predict_message(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    return "SPAM" if prediction == 1 else "HAM (Not Spam)"

# -------------------------------
# Command-line app
# -------------------------------

print("\n📨 Smart Spam Detection AI (TF-IDF)")
print("Type a message and press Enter")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter a message: ")

    if user_input.lower() == "quit":
        print("Goodbye 👋")
        break

    print("Prediction:", predict_message(user_input))
    print("-" * 40)
