# 🛡️ AI-Driven Email Security & Spam Filter

This project aims to develop and compare the performance of Artificial Intelligence (AI) versus Traditional Rule-based methods for email filtering. The core system utilizes a Logistic Regression model for text classification and features an interactive Web UI built with Gradio for real-time spam detection and visual analysis.

# 📖 Overview
As phishing attacks and spam emails become increasingly sophisticated, traditional keyword-based filters often fail to provide adequate protection. This project leverages Machine Learning (NLP + ML) to analyze the semantic features of email content (using TF-IDF) and performs a comparative experiment against a traditional keyword filter.

# Key Features

- AI Smart Detection: High-accuracy classification based on a Logistic Regression model.

- raditional Baseline: A built-in keyword and format-based rule filter used for performance benchmarking.

- Interactive UI: A user-friendly Web interface built with Gradio, supporting real-time input and dual-result display.

- Performance Evaluation: Automatic calculation and output of Precision, Recall, F1-Score, and Confusion Matrix.

# 💾 Dataset
The model training and testing are based on the Spam Email Dataset sourced from Kaggle.
- Source: [Kaggle - Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset/data)
- Content: The dataset contains a collection of emails labeled as Spam or Ham, used to train the supervised learning model.

# 📂 File Structure
```plaintext
.
├── datas/                  # Directory for datasets and saved models
│   ├── spam_data.csv       # Raw email dataset (Assumed)
│   ├── lr_model.pkl        # Trained AI model
│   └── tfidf_vectorizer.pkl# Fitted TF-IDF vectorizer
├── app.py                  # Main Web UI application (Gradio) for real-time demo
├── model.py                # Main script for AI model training and performance testing
├── traditional_filter.py   # Main script for testing the performance of the traditional filter
├── data_loader.py          # Module: Loads CSV data and splits into train/test sets
├── model_trainer.py        # Module: Defines feature engineering (TF-IDF) and training logic
├── model_evaluator.py      # Module: Calculates metrics (F1-Score, Confusion Matrix, etc.)
├── filter_func.py          # Module: Defines specific rules for the traditional filter
├── requirements.txt        # List of project dependencies
└── .gitignore              # Git ignore file
```


# Installation
It is recommended to use a Python virtual environment to run this project to ensure a clean environment.

## 1. Clone the Repository
```
git clone https://github.com/nillson-dot/SpamEmailFilter.git
cd SpamEmailFilter
```

## 2. Create a Virtual Environment
```
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies
```
pip install -r requirements.txt
```

## 4. Download NLTK Corpus (Required)
```
python -c "import nltk; nltk.download('stopwords')"
```

# 🚀 Usage
## Step 1: Train AI Model & Test Performance
```
python model.py
```

## Step 2: Test Traditional Filter Performance
```
python traditional_filter.py
```

## Step 3: Launch Web UI Demo
Run app.py to start the Gradio interface.
```
python app.py
```
> Once launched, open the URL displayed in the terminal (usually http://127.0.0.1:7860) in your browser to test email content.

# 📝 Credits
## Course
  「網際網路技術」 in NTPU
## Tech Stack
  Python, Scikit-learn, Gradio, Pandas, NLTK
## AI Assistance
  The documentation preparation and refinement, as well as crucial debugging support, were assisted by *Google Gemini*.
