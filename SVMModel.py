import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the preprocessed data
train_tfidf = pd.read_csv('svm_train_tfidf.csv')
dev_tfidf = pd.read_csv('svm_dev_tfidf.csv')
train_labels = pd.read_csv('svm_train_labels.csv')['Note']
dev_labels = pd.read_csv('svm_dev_labels.csv')['Note']

# Create a LinearSVC model with specified parameters
svm_model = LinearSVC(C=1.0, max_iter=1000)

# Fit the model on the training data
svm_model.fit(train_tfidf, train_labels)

# Predict on the validation data
predictions = svm_model.predict(dev_tfidf)

# Evaluate the model
print(classification_report(dev_labels, predictions))
print("Accuracy:", accuracy_score(dev_labels, predictions))