import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import nltk
from nltk.corpus import stopwords
import string
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve


path = 'YelpData/Yelp_train.csv'
#path = 'Yelp_test.csv'
data = pd.read_csv(path, encoding = "latin") # todo: will this encoding type affects data?
# SHAPE OF THE DATASET
print("Shape of the dataset:")
print(data.shape)
# COLUMN NAMES
print("Column names:")
print(data.columns)
# DATATYPE OF EACH COLUMN
print("Datatype of each column:")
print(data.dtypes)
# SEEING FEW OF THE ENTRIES
print("Few dataset entries:")
print(data.head())
# DATASET SUMMARY
data.describe(include='all')


# graph = sns.FacetGrid(data=data,col='stars')
# # graph.map(plt.hist,'funny',color='blue')
# graph.map(plt.hist,'funny')
# plt.show()


# note: for pandas, use & | instead of and or
# note: data['name'][() & ()] the brackets are necessary
# note: data['funny'] in str!!


# print('funny column sample',type(data['useful'][0]))


# tmp=stats.f_oneway(data['stars'][(data['funny'] >= 0) & (data['funny'] < 1)],
#                data['stars'][(data['funny'] >= 1) & (data['funny'] < 2)],
#                data['stars'][(data['funny'] >= 2) & (data['funny'] < 3)],
#                data['stars'][(data['funny'] >= 3 ) & (data['funny'] < 4)])
#
# print('f=%f, p=%f ' %(tmp[0],tmp[1]))


x = data['text']
y = data['stars']
print(x.head())
print(y.head())
print('finish checking. start vectorizing')


def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# CONVERTING THE WORDS INTO A VECTOR
vocab = CountVectorizer(analyzer=text_process).fit(x)
print(len(vocab.vocabulary_))
r0 = x[0]
print(r0)
vocab0 = vocab.transform([r0])
print(vocab0)
"""
    Now the words in the review number 78 have been converted into a vector.
    The data that we can see is the transformed words.
    If we now get the feature's name - we can get the word back!
"""
print("Getting the words back:")
print(vocab.get_feature_names()[1])


x = vocab.transform(x)
#Shape of the matrix:
print("Shape of the sparse matrix: ", x.shape)
#Non-zero occurences:
print("Non-Zero occurences: ",x.nnz)

# DENSITY OF THE MATRIX
density = (x.nnz/(x.shape[0]*x.shape[1]))*100
print("Density of the matrix = ",density)

# SPLITTING THE DATASET INTO TRAINING SET AND TESTING SET
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=101)

import pickle



# Saving the objects:
with open('objs_full.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x_train, x_test, y_train, y_test], f)
print('data saved to file')
# # Getting back the objects:
# with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
#     obj0, obj1, obj2 = pickle.load(f)

# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
predmnb = mnb.predict(x_test)
print("Confusion Matrix for Multinomial Naive Bayes:")
print(confusion_matrix(y_test,predmnb))
print("Score:",round(accuracy_score(y_test,predmnb)*100,2))
print("Classification Report:",classification_report(y_test,predmnb))