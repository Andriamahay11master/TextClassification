import pandas as pd

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=["label", "text"]
)

# Show first rows
print(data.head())

# Dataset size
print("\nDataset shape:", data.shape)

# Label distribution
print("\nLabel counts:")
print(data["label"].value_counts())
