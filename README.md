This code reads acsv file containing news data into a data farame, extracts the labels (e.g, 'FAKE' or 'Real"), and splits the dataset into a training and test sets. 
It then vectorizes the text data using  afidfVectorizer and trains a PassiveAggressiveClassifier ont teh training data.
The classifier is used to predict the labels for the test set, and the accuracay of the model is calculated and printed. 
Finally, the code builds and prints a confusion matrix to evaluate the classifier's performance in terms of true positives, true negatives, false positives, and false negatives
