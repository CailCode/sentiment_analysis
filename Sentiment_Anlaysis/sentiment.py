
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("movie.csv")

X = df['text']
y= df['tag']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

print("--Bernoulli--")
model = BernoulliNB()
model.fit(X_train,y_train)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

acc_tr = accuracy_score(y_train,train_predict)
acc_te = accuracy_score(y_test,test_predict)

print(f"ACC TRAIN::{acc_tr} -- ACC TEST:{acc_te}")

# inizia il vero programma
while True:
    print("---Scrivi la tua recensione!---")
    review = []
    review.append(input())
    review = vectorizer.transform(review)
    print(f"La tua recensione è::{model.predict(review)}")
    esc = input("Vuoi uscire::")
    if esc=="si": break
    




