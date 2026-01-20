# Text Classification

A machine learning project that builds and trains a text classification model to automatically categorize text into predefined categories. This project demonstrates the complete pipeline for text classification, from data loading through model training and evaluation.

## Project Overview

This project implements a **text classification system** using scikit-learn and machine learning techniques to classify SMS messages as spam or ham (legitimate). The implementation covers the entire ML workflow including data exploration, text vectorization, model training, and evaluation.

## Features

- **Data Loading & Exploration**: Load and analyze SMS message datasets
- **Text Preprocessing**: Vectorization with stop word removal and lowercasing
- **Model Training**: Uses Naive Bayes classifier for spam/ham classification
- **Modular Design**: Separated concerns with dedicated scripts for each pipeline step
- **Web Interface**: Flask-based application for real-time text classification

## Project Structure

```
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── text_classifier.py          # Data exploration and analysis
├── step2_vectorization.py      # Text vectorization implementation
├── step3_model.py              # Model training and evaluation
├── spam_app.py                 # Flask web application for classification
└── [data files]                # Dataset directory (if applicable)
```

### File Descriptions

- **text_classifier.py**: Loads the SMS dataset and performs exploratory data analysis (EDA)
- **step2_vectorization.py**: Implements CountVectorizer for converting text to numerical features
- **step3_model.py**: Trains MultinomialNB model and evaluates performance
- **spam_app.py**: Flask application providing a web interface for real-time predictions

## Requirements

- Python 3.7+
- pandas - Data manipulation and analysis
- scikit-learn - Machine learning library
- nltk - Natural Language Toolkit
- matplotlib & seaborn - Data visualization
- numpy - Numerical computing

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd TextClassification
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Explore the Data

```bash
python text_classifier.py
```

This script loads the SMS dataset and displays:

- First few records
- Dataset dimensions
- Label distribution (spam vs ham)

### Step 2: Vectorize Text

```bash
python step2_vectorization.py
```

Converts raw text into numerical features using CountVectorizer.

### Step 3: Train the Model

```bash
python step3_model.py
```

Trains a Naive Bayes classifier and evaluates its performance.

### Step 4: Run the Web Application

```bash
python spam_app.py
```

Launches the Flask web interface for real-time text classification.

## Dataset

The project uses the SMS Spam Collection dataset from:

- **Source**: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
- **Format**: TSV (Tab-Separated Values)
- **Columns**: label (ham/spam), text (message content)

## Algorithm

- **Vectorizer**: CountVectorizer with English stop word removal
- **Model**: MultinomialNB (Naive Bayes classifier)
- **Train-Test Split**: Standard scikit-learn train_test_split

## Performance Metrics

The model evaluation includes:

- Accuracy
- Precision
- Recall
- F1-Score

## Future Enhancements

- Implement additional classifiers (SVM, Random Forest)
- Add cross-validation for better model evaluation
- Implement feature engineering techniques (TF-IDF)
- Add model persistence (pickle/joblib)
- Improve UI/UX of the web application
- Support for custom datasets

## License

[Add your license here]

## Author

[Add your name/info here]
