# ==============================================
# SMS Spam Detection Project
# AI Classifier using Naive Bayes
# ==============================================

# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
nltk.download('punkt')  # for text tokenization

# 2. Load dataset
# Make sure 'spam.csv' is in the same folder as this script
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]  # Keep only relevant columns
data.columns = ['label', 'message']

# 3. Encode labels (HAM=0, SPAM=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 4. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# 5. Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Evaluate model performance
predictions = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Optional: confusion matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)

# 8. Function for predicting new messages
def classify_sms(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "SPAM" if pred == 1 else "HAM (Not Spam)"

# 9. Test the AI with sample messages
print("\n--- Test Messages ---")
test_messages = [
    "Congratulations! You won a free iPhone!",
    "Hey, are we meeting tomorrow?",
    "Claim your free cash now!",
    "Can you send me the notes from class?"
]

for msg in test_messages:
    print(f"Message: {msg}")
    print("Prediction:", classify_sms(msg))
    print("-----------------------")

# 10. Optional: real-time user input
while True:
    user_input = input("Enter an SMS to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("Prediction:", classify_sms(user_input))
    print("-----------------------")
