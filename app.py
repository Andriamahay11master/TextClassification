from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

app = Flask(
    __name__,
    static_folder="assets",
    static_url_path="/static"
)

# -------------------------------
# Train the AI model (once at startup)
# -------------------------------

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=["label", "text"]
)

data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(data["text"])
y = data["label_num"]

X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LinearSVC()
model.fit(X_train, y_train)

# -------------------------------
# Utility: convert SVM score to confidence
# -------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------------
# Routes
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        message_vector = vectorizer.transform([message])

        # SVM raw score (distance from hyperplane)
        score = model.decision_function(message_vector)

        # Convert score to pseudo-probability
        prob_spam = sigmoid(score)
        prob_ham = 1 - prob_spam

        pred = model.predict(message_vector)[0]

        if pred == 1:
            prediction = "Spam"
            confidence = prob_spam
        else:
            prediction = "Ham"
            confidence = prob_ham

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
