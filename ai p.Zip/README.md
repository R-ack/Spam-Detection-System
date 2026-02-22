SMS Spam Detection

Project Overview

This project is an AI-powered SMS classification system that detects spam messages.
It uses Natural Language Processing (NLP) and a Naive Bayes machine learning model to classify SMS messages as either:
-SPAM
-HAM (Not Spam)

The model is trained on a publicly available dataset of SMS messages and can predict new messages in real-time.

Features
1. Preprocesses SMS text using TF-IDF vectorization
2. Trains a Multinomial Naive Bayes classifier
3. Provides accuracy and performance metrics
4. Predicts spam in new messages
5. Works with Python and standard ML libraries

Installation
1. Make sure Python 3 is installed.
2. Open terminal/command prompt and navigate to the project folder.
3. Install required libraries

Dataset
The project uses the SMS Spam Collection Dataset

-File: spam.csv

Dataset contains two columns:
label → spam or ham
message → the SMS text


How to Run
1. Open a terminal/command prompt in the project folder.
2. Run the Python script:

python sms_spam_ai.py


3. The script will display:
-Model accuracy
-Classification report
-Predictions for sample messages

Test Messages Predictions:
"Congratulations! You won a free iPhone!" → SPAM

"Hey, are we meeting tomorrow?" → HAM (Not Spam)

Libraries Used
1. pandas
2. numpy
3. scikit-learn
4. nltk





