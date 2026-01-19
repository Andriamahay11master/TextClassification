import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=["label", "text"]
)

# Convert labels to numbers
# ham -> 0, spam -> 1
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# Create the vectorizer
vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
)

# Fit and transform the text data
X = vectorizer.fit_transform(data["text"])
y = data["label_num"]

# Inspect results
print("Shape of feature matrix:", X.shape)
print("Number of labels:", y.shape)

# Show some learned words
print("\nFirst 10 words in vocabulary:")
print(list(vectorizer.vocabulary_.keys())[:10])
