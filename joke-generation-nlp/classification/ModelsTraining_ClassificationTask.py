import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

import pickle

jokesBody = []
jokesCategory = []
freq = {}

with open("../preprocessing/jokesAfterPreprocessing.csv", "r") as f:
    reader = csv.DictReader(f)
    jokes = list(reader)


for joke in jokes:
       jokesBody.append(joke.get("Body"))
       jokesCategory.append(joke.get("Category"))


X_train, X_test, y_train, y_test = train_test_split(jokesBody, jokesCategory,test_size=0.2, random_state=0)

# Creation of token counters matrix
vectorizer = CountVectorizer(max_features=1500, max_df=0.85, stop_words=stopwords.words("english"))
X = vectorizer.fit_transform(X_train)


# Transform the token counters matrix to a TF-IDF representation
tfidf_Transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
X_train_tfidf = tfidf_Transformer.fit_transform(X)


# Train the random forest model
R = RandomForestClassifier(n_estimators=50, random_state=0)
R.fit(X_train_tfidf,y_train)

# Train the multinomial Naive Bayes model
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Train the Logistic Regression model
lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
lm.fit(X_train_tfidf, y_train)

# Metrics of all 3 models
print(metrics.classification_report(y_test, lm.predict(vectorizer.transform(X_test))))
print(metrics.classification_report(y_test, R.predict(vectorizer.transform(X_test))))
print(metrics.classification_report(y_test, clf.predict(vectorizer.transform(X_test))))

# Store the Logistic Regreession model
pickle.dump(lm, open("LogisticRegressionModel.sav","wb"))
