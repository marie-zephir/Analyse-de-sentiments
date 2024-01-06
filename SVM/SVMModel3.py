import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the preprocessed data
train_tfidf = pd.read_csv('svm_train_tfidf.csv')
dev_tfidf = pd.read_csv('svm_dev_tfidf.csv')
test_tfidf = pd.read_csv('svm_test_tfidf.csv')
train_labels = pd.read_csv('svm_train_labels.csv')['Note']
dev_labels = pd.read_csv('svm_dev_labels.csv')['Note']
test_review = pd.read_csv('svm_test_reviews.csv')['ReviewId']

# Create a LinearSVC model with specified parameters
svm_model = LinearSVC(C=0.01, max_iter=1000)

# Fit the model on the training data
svm_model.fit(train_tfidf, train_labels)

# Predict on the validation data
predictions = svm_model.predict(dev_tfidf)

# Evaluate the model
print(classification_report(dev_labels, predictions))
print("Accuracy:", accuracy_score(dev_labels, predictions))

predictions_test = svm_model.predict(test_tfidf)
with open('sortie3.txt', 'w') as f:
    count = 0
    for review in test_review:
        f.write(review + " " + predictions_test[count])
        f.write('\n')
        count = count+1

