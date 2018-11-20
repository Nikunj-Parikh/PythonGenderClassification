import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load our data
df = pd.read_csv('gender_refine-csv.csv')

df_names = df

df_names.sex.replace({'F':0,'M':1},inplace=True)

Xfeatures = df_names['name']
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

cv.get_feature_names()

y = df_names.sex

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")

print("Accuracy of Model",clf.score(X_train,y_train)*100,"%")


# Sample1 Prediction
sample_name = ["Mary"]
vect = cv.transform(sample_name).toarray()

print(vect)

clf.predict(vect)

# A function to do it
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")
    

genderpredictor(input("Enter Name: "))