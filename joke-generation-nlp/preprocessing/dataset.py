import pandas as pd    
import re
import pickle
from sklearn.feature_extraction import text
from nltk.tokenize import RegexpTokenizer
from scipy.sparse.construct import random
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
# function to split the data for cross-validation
from sklearn.model_selection import train_test_split
# function for transforming documents into counts
from sklearn.feature_extraction.text import CountVectorizer
# function for encoding categories
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import nltk
import os
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")

df = pd.read_json(r'https://raw.githubusercontent.com/taivop/joke-dataset/master/stupidstuff.json')
df5= pd.read_json(r'https://raw.githubusercontent.com/taivop/joke-dataset/master/wocka.json')





#secondly lets find out how many different categories we have in our dataset and the number of jokes that each category has
categories={}
for c in df.category:
    if c not in categories:
        categories[c]=1
    else:
        categories[c]+=1


drop_unique_jokes= [] # lets remove unique jokes, eitherway they are not important for our prediction
for category,num in categories.items():
    if num == 1:
     drop_unique_jokes.append(category)
for c in drop_unique_jokes:
    df=df[df.category != c]


#secondly lets find out how many different categories we have in our dataset and the number of jokes that each category has
categories={}
for c in df5.category:
    if c not in categories:
        categories[c]=1
    else:
        categories[c]+=1


drop_unique_jokes= [] # lets remove unique jokes, eitherway they are not important for our prediction
for category,num in categories.items():
    if num == 1:
     drop_unique_jokes.append(category)
for c in drop_unique_jokes:
    df5=df5[df5.category != c]



def normalize_text(s):
    
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    s = re.sub(r'[^(a-zA-Z)\s]','',s)

    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)

     # specific
    s = re.sub(r"won\'t", "will not", s)
    s = re.sub(r"can\'t", "can not", s)
   

    # general
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'m", " am", s)
    s = re.sub(r"ya", "you", s)
    s = re.sub(r"yo","your",s)
    s = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', s)
    s = re.sub(r'<br />', ' ', s)
    s = re.sub(r'\'', ' ', s)

    return s



df['jokes'] = [normalize_text(s) for s in df['body']]
df=df.drop('body',axis=1)

df5['jokes'] = [normalize_text(s) for s in df5['body']]
df5=df5.drop('body',axis=1)





jokesBody = []
jokesCategory = []
freq = {}




data1 = df

data2 = df5




#----------------------- Wocka arxeio ---------------------



for index,joke in data2.iterrows():
    if joke.get("category") != "one liners" and joke.get("category") != "knock-knock"  and joke.get("category") != "blonde" and joke.get("category") != "yo momma" and joke.get("category") != "yo mama":    ###### Aytes tis kathgories tiw afairw


        if joke.get("category") == "yo momma" or joke.get("category") == "yo mama":
            joke["category"] = "insults"

        if joke.get("category") == "religious" or joke.get("category") == "sports" or joke.get("category") == "college" or joke.get("category") == "at work" or joke.get("category") == "tech" or joke.get("category") == "news / politics" or joke.get("category") == "bar" or joke.get("category") == "gross" or joke.get("category") == "children" or joke.get("category") == "men / women" or joke.get("category") == "puns" or joke.get("category") == "other / misc":
            joke["category"] = "others"

        jokesBody.append(joke.get("jokes"))
        jokesCategory.append(joke.get("category"))



#--------------------- stupidsStuff arxeio -------------------

for index,joke in data1.iterrows():
    
    if joke.get("category") == "light bulbs":
        joke["category"] = "lightbulb"
        jokesBody.append(joke.get("jokes"))
        jokesCategory.append(joke.get("category"))


    if joke.get("category") == "animals":
        joke["category"] = "animal"
        jokesBody.append(joke.get("jokes"))
        jokesCategory.append(joke.get("category"))
        #if joke["category"] in freq.keys():




teliko = open("jokesAfterPreprocessing.csv", "w")
teliko.write("Category," + "Body" +"\n")
for joke in zip(jokesBody,jokesCategory):
    teliko.write(joke[1] + "," + joke[0] + "\n")
teliko.close()

























#testing 

# def nb_model(n):
#     print('\n')
#     print("*************************** "+str(n[1])+" ****************************")
#     print('\n')
#     vectorizer = CountVectorizer(binary=n[0])
#     x = vectorizer.fit_transform(df1['jokes'])

#     encoder = LabelEncoder() #  Encode target labels with value between 0 and n_classes-1.
#     y = encoder.fit_transform(df1['category'])
#     print(y)
#     nb = n[1]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = df1.category, random_state=42)
#     nb.fit(x_train, y_train)
    

#     y_predicted = nb.predict(x_test)
#     print(str(n[1])+' binary='+str(n[0])+ ' accuracy: ', metrics.accuracy_score(y_test, y_predicted)) 
#     print(str(n[1])+' binary='+str(n[0])+ ' recall: ', metrics.recall_score(y_test, y_predicted,average='macro')) 
#     print(str(n[1])+' binary='+str(n[0])+ ' f1: ', metrics.f1_score(y_test, y_predicted,average='macro')) 
#     y_predicted_labels = encoder.inverse_transform(y_predicted)
#     labels_predicted={}
#     for k in y_predicted_labels:
#         if k not in labels_predicted:
#             labels_predicted[k]=1
#         else:
#             labels_predicted[k]+=1
#     print('\n')
#     print('predicted categories of jokes: ',labels_predicted)
    
# li=[[False,MultinomialNB()],[True,MultinomialNB()],[False,BernoulliNB()]]
# for j in li:
#     nb_model(j)

# #second approach/ we will evaluate by score
# #fist we need to create 2 classes for the scores, one will be the jokes above average scores and the other
# df2['Label'] = 0
# ratings=0
# for x in df2.rating:
#     ratings+=x

# average_of_ratings=round(ratings/len(df2))

# df2.loc[df2['rating'] > average_of_ratings, ['Label']] = 1

# len(df2[df2['Label'] == 1])/len(df2)

# def second_approach_init(text):

#     text= normalize_text(text)
#     text = text.lower()
#     text = text.split()
#     stops = set(stopwords.words("english"))
#     text = [w for w in text if not w in stops]
#     text = " ".join(text)

#     # Tokenize each word
#     text =  nltk.WordPunctTokenizer().tokenize(text)
        
#     return text


# df2['joke_words'] = list(map(second_approach_init, df2.body))


# def lemmatized_words(text):
#     lemm = nltk.stem.WordNetLemmatizer()
#     df2['lemmatized_text'] = list(map(lambda word:
#                                      list(map(lemm.lemmatize, word)),
#                                      df2['joke_words']))
# lemmatized_words(df2.joke_words)

# #Bag of Words (BoW)

# ngram=[]
# for i in range(1,4):
#     cv = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[i,i], lowercase=False)
#     x = cv.fit_transform(df2['joke_words'])
#     words = cv.get_feature_names()
#     ngram.append(words)
    
# print('unigram: ', len(ngram[0]),' bigram: ', len(ngram[1]),' trigram: ',len(ngram[2]))


# #bag of words transofrmation

# training_data, test_data = train_test_split(df2, train_size = 0.7, random_state=42)
# bow_transform = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[3,3], lowercase=False)
# X_tr_bow = bow_transform.fit_transform(training_data['joke_words'])
# X_te_bow = bow_transform.transform(test_data['joke_words'])
# y_tr = training_data['Label']
# y_te = test_data['Label']
# print(y_tr,y_te)

# #TF-idf approach   
# # Tf-idf(w, d)= Bow(w, d) * log(Total Number of Documents /(Number of documents in which word w appears))

# tfidf_transform = text.TfidfTransformer(norm=None)
# X_tr_tfidf = tfidf_transform.fit_transform(X_tr_bow)
# X_te_tfidf = tfidf_transform.transform(X_te_bow)

# #classification with logistic regression 
# def simple_logistic_classify(X_tr, y_tr, X_test, y_test, description, _C=1.0):
#     model = LogisticRegression(C=_C).fit(X_tr, y_tr)
#     score = model.score(X_test, y_test)
#     print('Test Score with', description, 'features', score)
#     return model

# model_bow = simple_logistic_classify(X_tr_bow, y_tr, X_te_bow, y_te, 'bow')
# model_tfidf = simple_logistic_classify(X_tr_tfidf, y_tr, X_te_tfidf, y_te, 'tf-idf')

# param_grid_ = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}
# bow_search = sklearn.model_selection.GridSearchCV(LogisticRegression(), cv=5, param_grid=param_grid_)
# tfidf_search = sklearn.model_selection.GridSearchCV(LogisticRegression(), cv=5, param_grid=param_grid_)
# bow_search.fit(X_tr_bow, y_tr)
# print(bow_search.best_score_)
# tfidf_search.fit(X_tr_tfidf, y_tr)
# print(tfidf_search.best_score_)
# print(bow_search.best_params_)
# print(tfidf_search.best_params_)
# print(bow_search.cv_results_)

# results_file = open('tfidf_gridcv_results.pkl', 'wb')
# pickle.dump(bow_search, results_file, -1)
# pickle.dump(tfidf_search, results_file, -1)
# results_file.close()

# pkl_file = open('tfidf_gridcv_results.pkl', 'rb')
# bow_search = pickle.load(pkl_file)
# tfidf_search = pickle.load(pkl_file)
# pkl_file.close()
# search_results = pd.DataFrame.from_dict({'bow': bow_search.cv_results_['mean_test_score'],'tfidf': tfidf_search.cv_results_['mean_test_score']})
# print(search_results)
                        