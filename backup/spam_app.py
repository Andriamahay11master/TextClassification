import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# Load and prepare data
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
# Vectorization
# -------------------------------

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
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
# Prediction function
# -------------------------------

def predict_message(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    if prediction == 1:
        return "SPAM"
    return "HAM (Not Spam)"

# -------------------------------
# Command-line app
# -------------------------------

print("📨 Spam Detection AI")
print("Type a message and press Enter")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter a message: ")

    if user_input.lower() == "quit":
        print("Goodbye 👋")
        break

    result = predict_message(user_input)
    print("Prediction:", result)
    print("-" * 40)
