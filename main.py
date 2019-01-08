# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 22:06:14 2018

@author: sone_e
"""
import preprocessing_text
import feature_extractor
import classifier
import pandas as pd
from sklearn.model_selection import train_test_split

file = pd.read_csv('training dataset/new_training_dataset.csv',
                   encoding='ISO-8859-1', index_col=None)
'''
print("shape: ", file.shape)
print("describe: ", file.describe())
print("head: ", file.head())

file.dropna(inplace=True)
'''
preprocessing_text.preprocessing(file)
#print(file)

# split dataset into training and testing sets
# ratio 0.8:0.2
X_train, X_test, y_train, y_test = train_test_split(
    file.text, file.label, test_size=0.2, random_state=42)

#X_train, X_test = feature_extractor.count_vectorizer(X_train, X_test)
X_train, X_test = feature_extractor.tf_idf(X_train, X_test)

#classifier.multinomialNB(X_train, X_test, y_train, y_test)
#classifier.logisticReg(X_train, X_test, y_train, y_test)
#classifier.sgdClassifier(X_train, X_test, y_train, y_test)
classifier.svc(X_train, X_test, y_train, y_test)
#classifier.decisionTree(X_train, X_test, y_train, y_test)
#classifier.randomForest(X_train, X_test, y_train, y_test)
#classifier.gradientBoost(X_train, X_test, y_train, y_test)
