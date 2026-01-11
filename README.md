📚 Kindle Reviews Sentiment Analysis (NLP Project)
📌 Project Overview

This project focuses on Natural Language Processing (NLP) to analyze Amazon Kindle book reviews and predict the sentiment of a review (Positive or Negative).
The main goal is to understand customer opinions using text preprocessing and machine learning models.

🧠 Problem Statement

Online reviews are written in unstructured text form, which is difficult for machines to understand directly.
This project converts raw text reviews into meaningful numerical features and trains a model to classify sentiments accurately.

🗂 Dataset

Dataset contains Kindle book reviews

Columns include:

reviewText – customer review text

rating – star rating given by the user

⚙️ Technologies Used

Python

Pandas & NumPy – data handling

NLTK – text preprocessing

Scikit-learn – machine learning models

Matplotlib / Seaborn – visualization

🔄 Project Workflow

Data Loading

Text Cleaning

Lowercasing

Removing punctuation & numbers

Stopword removal

Stemming / Lemmatization

Feature Extraction

Bag of Words / TF-IDF

Model Training

Logistic Regression / Naive Bayes

Model Evaluation

Accuracy Score

Confusion Matrix

🧹 Text Preprocessing Techniques

Tokenization

Stopword Removal

Stemming using Porter Stemmer

Lemmatization using WordNet

🤖 Machine Learning Models Used

Logistic Regression

Naive Bayes

Logistic Regression gave better accuracy compared to Naive Bayes in this project.

📊 Model Performance

Accuracy achieved: ~80%+

Model performs well in classifying positive and negative reviews.

🚀 How to Run the Project
# Clone the repository
git clone https://github.com/your-username/kindle-review-nlp.git

# Navigate to the project folder
cd kindle-review-nlp

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook

📁 Project Structure
📦 kindle-review-nlp
 ┣ 📜 kindle_reviews.ipynb
 ┣ 📜 README.md
 ┣ 📜 requirements.txt

🎯 Key Learnings

Real-world NLP preprocessing

Feature extraction from text

Sentiment analysis using ML

Importance of clean text data

🔮 Future Improvements

Use Word2Vec / GloVe

Apply Deep Learning (LSTM / BERT)

Add web app using Flask / Streamlit

👤 Author

Tamana
Aspiring Data Scientist | Machine Learning Enthusiast
