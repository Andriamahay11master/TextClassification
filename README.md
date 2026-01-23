# Text Classification

A machine learning project that builds and trains a text classification model to automatically categorize text into predefined categories. This project demonstrates the complete pipeline for text classification, from data loading through model training and evaluation, with a modern, responsive web interface.

## Project Overview

This project implements a **spam detection system** using scikit-learn and machine learning techniques to classify SMS messages as spam or ham (legitimate). The implementation covers the entire ML workflow including data exploration, text vectorization, model training, and evaluation, combined with a beautiful Flask-based web application featuring a responsive frontend built with HTML, CSS, and SASS.

## Features

- **Data Loading & Exploration**: Load and analyze SMS message datasets
- **Text Preprocessing**: TF-IDF vectorization with stop word removal and lowercasing
- **Model Training**: Uses Naive Bayes classifier for spam/ham classification
- **Modular Design**: Separated concerns with dedicated scripts for each pipeline step
- **Modern Web Interface**: Flask-based application with a responsive, user-friendly design
- **Real-time Predictions**: Get instant spam detection with confidence scores
- **Styled Frontend**: Professional CSS/SASS styling for an enhanced user experience
- **Mobile-Responsive**: Optimized interface for desktop and mobile devices

## Project Structure

```
├── app.py                       # Main Flask application
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js dependencies (SASS)
├── templates/
│   └── index.html              # Flask HTML template with responsive interface
├── assets/
│   ├── css/
│   │   └── style.css           # Compiled CSS styles
│   └── sass/
│       ├── _variables.scss     # SASS variables and configuration
│       ├── _mixins.scss        # SASS mixins and utilities
│       └── style.scss          # Main SASS stylesheet
├── backup/                      # Previous implementation files (archived)
└── [data files]                # Dataset directory (if applicable)
```

### File Descriptions

- **app.py**: Main Flask application that trains the model and handles web requests
- **templates/index.html**: Responsive HTML template with real-time prediction interface
- **assets/css/style.css**: Compiled CSS stylesheet for professional styling
- **assets/sass/**: SASS source files for maintainable stylesheet organization
  - `_variables.scss`: Color schemes, fonts, and spacing variables
  - `_mixins.scss`: Reusable SASS mixins for responsive design
  - `style.scss`: Main SASS stylesheet
- **backup/**: Previous implementation files for reference

## Requirements

- Python 3.7+
- Flask - Web framework
- pandas - Data manipulation and analysis
- scikit-learn - Machine learning library (TfidfVectorizer, MultinomialNB)
- numpy - Numerical computing

### Development Dependencies

- Node.js (for SASS compilation)
- sass - SCSS/SASS compiler

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd TextClassification
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Install Node.js dependencies (optional, for SASS compilation):

```bash
npm install
```

## Usage

### Running the Application

Start the Flask web application:

## Running locally (Windows)

```bash
python app.py
```

## Running in production (Linux)

```bash
gunicorn app:app
```

The application will:

- Load the SMS dataset from the online source
- Train the TF-IDF + Naive Bayes model at startup
- Launch a web server at `http://localhost:5000`
- Display the interactive spam detection interface

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:5000`
2. Type or paste a message into the textarea
3. Click "Predict" or press Enter to classify the message
4. View the prediction result with confidence score and progress bar

### Compiling SASS (Optional)

If you modify the SASS files, recompile to CSS:

```bash
npm run sass
# or
sass assets/sass/style.scss assets/css/style.css
```

## Dataset

The project uses the SMS Spam Collection dataset from:

- **Source**: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
- **Format**: TSV (Tab-Separated Values)
- **Columns**: label (ham/spam), text (message content)

## Algorithm

- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency) with English stop word removal and bigrams (1-2 grams)
- **Model**: MultinomialNB (Naive Bayes classifier)
- **Train-Test Split**: 80-20 split using scikit-learn train_test_split

## Performance Metrics

The model provides:

- **Accuracy**: Overall classification accuracy
- **Confidence Score**: Probability score for the prediction (0-100%)
- **Visual Feedback**: Progress bar showing confidence level with color-coded results (red for spam, green for ham)

## Future Enhancements

- Implement additional classifiers (SVM, Random Forest, Neural Networks)
- Add model persistence (pickle/joblib) for faster startup
- Implement cross-validation for better model evaluation
- Advanced feature engineering (custom TF-IDF settings, feature selection)
- User authentication and prediction history tracking
- API endpoints for programmatic access
- Support for custom datasets and model training
- Performance analytics and model accuracy monitoring
- Dark mode toggle for the web interface

## License

[Add your license here]

## Author

Henikaja Andriamahay IRIMANANA
