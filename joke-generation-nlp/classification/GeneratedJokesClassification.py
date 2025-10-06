import pickle
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords




model = pickle.load(open("LogisticRegressionModel.sav", "rb")) #load the Logistic Regression model
jokesBody = []
jokesCategory = []
GeneratedJokes = []

with open("../preprocessing/jokesAfterPreprocessing.csv", "r") as f:
    reader = csv.DictReader(f)
    jokes = list(reader)

with open("../generation/GeneratedJokes.csv", "r") as f2:
    reader2 = csv.DictReader(f2)
    jokesToClassify = list(reader2)



for joke in jokes:
       jokesBody.append(joke.get("Body"))
       jokesCategory.append(joke.get("Category"))


for joke in jokesToClassify:
       GeneratedJokes.append(joke.get("output"))



vectorizer = CountVectorizer(max_features=1500, max_df=0.85, stop_words=stopwords.words("english"))
X = vectorizer.fit_transform(jokesBody)



cat = model.predict(vectorizer.transform(GeneratedJokes)) #Classify the generated jokes

teliko = open("GeneratedJokesClassification.csv", "w")
teliko.write("Category," + "Body" +"\n")
for joke in zip(GeneratedJokes,cat):
    teliko.write(joke[1] + "," + joke[0] + "\n")
teliko.close()


