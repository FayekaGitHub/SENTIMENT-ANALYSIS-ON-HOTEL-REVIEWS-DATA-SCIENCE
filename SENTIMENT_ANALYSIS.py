from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

data = pd.read_csv('/content/drive/My Drive/colab/hotel-reviews.csv')

data.shape

data.sample(5)

data.describe()

data['Is_Response'].value_counts()

"""## Pre Processing"""

#removing columns that are not needed
data.drop(columns = ['User_ID', 'Browser_Used', 'Device_Used', 'Is_Response'], inplace = True)

data.sample(3)

"""## Text cleaning"""

import re
import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import WordPunctTokenizer
stopwords = nltk.corpus.stopwords.words('english')

def text_process(text):
  text_new = "".join([i for i in text if i not in string.punctuation])
  text_new
  words = nltk.tokenize.word_tokenize(text_new)
  clean_text = " ".join([i for i in words if i not in stopwords])
  text_clean = "".join([i.lower() for i in clean_text])
  return text_clean

cleaned_text = []

for text in data.Description:
    cleaned_text.append(text_process(text))

clean_text = pd.DataFrame({'clean_text' : cleaned_text})

data = pd.concat([data, clean_text], axis = 1)

data.sample(10)

from textblob import TextBlob

def getPolarity(sent):
  sentence = TextBlob(sent)
  p = sentence.sentiment.polarity
  if(p<-0.5 and p>=-1):
     return -2
  elif(p<0 and p>=-0.5):
    return -1
  elif(p==0):
     return 0
  elif(p<=0.5 and p>0):
    return 1
  elif(p>0.5 and p<=1):
     return 2
  #return p

Response = []
for text in data.clean_text:
  Response.append(getPolarity(text))
Response = pd.DataFrame({'Response' : Response})

data = pd.concat([data, Response], axis = 1)

data.sample(5)

IV = []
DV = []
x1=0
x2=0
x3=0
x4=0
x5=0
for i, row in data.iterrows():
  if(row['Response']==2 and x1<1000):
    IV.append(row['clean_text'])
    DV.append(2)
    x1=x1+1
  elif(row['Response']==1 and x2<1000):
    IV.append(row['clean_text'])
    DV.append(1)
    x2=x2+1
  elif(row['Response']==0 and x3<1000):
    IV.append(row['clean_text'])
    DV.append(0)
    x3=x3+1
  elif(row['Response']==-1 and x4<1000):
    IV.append(row['clean_text'])
    DV.append(-1)
    x4=x4+1
  elif(row['Response']==1 and x5<1000):
    IV.append(row['clean_text'])
    DV.append(-2)
    x5=x5+1

len(IV)

"""# Vectorization


"""

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(IV)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_train_tfidf, DV, test_size=0.2, random_state=0)

"""# Multinomial Naive Bayes

"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

classifier = MultinomialNB();
classifier.fit(x_train, y_train)

from sklearn.metrics import classification_report, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(cm)
print(classification_report(y_test, y_pred))

results = []
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
r1 = cross_val_score(classifier, x_train, y_train, cv=kfold, scoring='accuracy')
results.append(r1)

r1

"""# Complement Naive Bayes Model"""

from sklearn.naive_bayes import ComplementNB
classifier2 = ComplementNB()
classifier2.fit(x_train, y_train)

from sklearn.metrics import classification_report, accuracy_score
y_pred2 = classifier2.predict(x_test)
cm = confusion_matrix(y_test, y_pred2)
print(accuracy_score(y_test, y_pred2))
print(cm)
print(classification_report(y_test, y_pred2))

kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
r2 = cross_val_score(classifier2, x_train, y_train, cv=kfold, scoring='accuracy')
results.append(r2)

r2

"""# Logistic Regression

"""

from sklearn.linear_model import LogisticRegression
classifier3 = LogisticRegression(C=1.0).fit(x_train, y_train)

from sklearn.metrics import classification_report, accuracy_score
y_pred3 = classifier3.predict(x_test)
cm = confusion_matrix(y_test, y_pred3)
print(accuracy_score(y_test, y_pred3))
print(cm)
print(classification_report(y_test, y_pred3))

kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
r3 = cross_val_score(classifier3, x_train, y_train, cv=kfold, scoring='accuracy')
results.append(r3)

r3

"""# Support vector machine

"""

from sklearn import model_selection, svm
classifier4 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
classifier4.fit(x_train, y_train)

from sklearn.metrics import classification_report, accuracy_score
y_pred4 = classifier4.predict(x_test)
cm = confusion_matrix(y_test, y_pred4)
print(accuracy_score(y_test, y_pred4))
print(cm)
print(classification_report(y_test, y_pred4))
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
r4 = cross_val_score(classifier4, x_train, y_train, cv=kfold, scoring='accuracy')
results.append(r4)

names = []
names.append('MultinomialNB')
names.append('ComplementNB')
names.append('LR')
names.append('svm')
names

"""# Sample Reviews

"""

docs_new = [
    "stayed way home trip south america beautiful spot central everything really would nt hesitate stay quality rooms food drink top class staff excellent minutes times square minutes central park couldnt better anyone thinking break",
    'not nice would never recommend',
    "it was good but it could have been better",
    "That hotel is surely a HELLTEL!"
]
X_new_counts = count_vector.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted1 = classifier.predict(X_new_tfidf)
predicted2 = classifier2.predict(X_new_tfidf)
predicted3 = classifier3.predict(X_new_tfidf)
predicted4 = classifier4.predict(X_new_tfidf)
print("Multinomial Naive Bayes:")
for i in predicted1:
  print(i)
print("Complement Naive Bayes:")
for i in predicted2:
  print(i)
print("Logistic Regression:")
for i in predicted3:
  print(i)
print("SVM:")
for i in predicted4:
  print(i)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
