
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

#dataset
df = pd.read_csv("movie.csv")

X = df['text'] #features
y= df['tag'] #target

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
# let's vectorize the string with Bag of Words method

# split the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

print("--Bernoulli--")
model = BernoulliNB()
model.fit(X_train,y_train) # train the model

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 
acc_tr = accuracy_score(y_train,train_predict)
acc_te = accuracy_score(y_test,test_predict)
#
print(f"ACC TRAIN::{acc_tr} -- ACC TEST:{acc_te}")
# accuracy of test is 70%

# start 
while True:
    print("---Write a Review!---")
    review = []
    review.append(input())
    review = vectorizer.transform(review)
    print(f"La tua recensione è::{model.predict(review)}")
    esc = input("Exit::yes-no::")
    if esc=="yes": break
    




