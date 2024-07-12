import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data
df=pd.read_csv('/Users/kolchmielarz/Desktop/newsproj.py/news.csv')

#Get shape and head
df.shape
df.head()

#Get the labels from docs
labels=df.label
labels.head()

print("Shape of the DataFrame:", df.shape)
print("First five rows of the DataFrame:")
print(df.head())

#Split the dataset for training and test 
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test = tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

#printf not working here?
print('Accuracy: ' + str(round(score * 100, 2)) + '%')

#Build confusion matrix - Print TP, TN, FP, FN
confusion_matrix_values = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']).tolist()

print("Confusion Matrix:")
for row in confusion_matrix_values:
    print(row)



