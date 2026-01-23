from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__, static_folder="assets",
    static_url_path="/static")

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
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

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
        probs = model.predict_proba(message_vector)[0]

        ham_prob, spam_prob = probs

        if spam_prob > ham_prob:
            prediction = "SPAM"
            confidence = spam_prob
        else:
            prediction = "HAM (Not Spam)"
            confidence = ham_prob

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

