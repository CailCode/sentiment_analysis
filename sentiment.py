
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re

# turn of the SettingWithCopyWarning warnign
pd.set_option('mode.chained_assignment', None)

#clear the string from special characters
def clear_string(ss):
    temp = ss.strip()
    temp = re.sub('[^a-zA-Z0-9]',' ',temp)
    return temp

#dataset
df = pd.read_csv("movie.csv")

X = df['text'] #features
y= df['tag'] #target

#clear the dataset
for i in range(len(X)):
    X[i] = clear_string(X[i])


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
# let's vectorize the string with Bag of Words method

# split the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

print("--Train--")
model = MultinomialNB()
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
    review[0] = clear_string(review[0])
    review = vectorizer.transform(review)
    print(f"La tua recensione è::{model.predict(review)}")
    esc = input("Exit::yes-no::")
    if esc=="yes": break
    




