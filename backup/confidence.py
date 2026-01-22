import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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
# TF-IDF Vectorization
# -------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(data["text"])
y = data["label_num"]

# -------------------------------
# Train model
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# Prediction with confidence
# -------------------------------

def predict_message_with_confidence(message):
    message_vector = vectorizer.transform([message])
    probabilities = model.predict_proba(message_vector)[0]

    ham_prob = probabilities[0]
    spam_prob = probabilities[1]

    if spam_prob > ham_prob:
        return "SPAM", spam_prob
    return "HAM (Not Spam)", ham_prob

# -------------------------------
# Command-line app
# -------------------------------

print("\n📨 Smart Spam Detection AI (with confidence)")
print("Type a message and press Enter")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter a message: ")

    if user_input.lower() == "quit":
        print("Goodbye 👋")
        break

    label, confidence = predict_message_with_confidence(user_input)
    print(f"Prediction: {label} (confidence: {confidence * 100:.1f}%)")
    print("-" * 40)
