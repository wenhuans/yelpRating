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
print('num of rows before purging = ', data.shape)
valid_index = []
tmp = data['cool'].values

for i in range(data.shape[0]):
    #print('*** type check', type(tmp[i]))
    if float(tmp[i]) <= 100:
        valid_index.append(i)

data = data.loc[valid_index,:]
print('num of rows after purging = ', data.shape)
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

target = 'disgusting'
r = (1,3)


graph = sns.FacetGrid(data=data,col='stars')
# graph.map(plt.hist,'funny',color='blue')
graph.map(plt.hist,target,range = r)

# draw = data['stars'][(data[target] ==2) ]
# plt.hist(draw)
# plt.xlabel('stars')
# plt.ylabel('Counts')
# plt.title([target,'[3,4)'])

plt.show()


# note: for pandas, use & | instead of and or
# note: data['name'][() & ()] the brackets are necessary
# note: data['funny'] in str!!


# print('funny column sample',type(data['useful'][0]))


tmp=stats.f_oneway(data['stars'][(data[target] >= 2) & (data[target] < 3)],
                   data['stars'][(data[target] >= 1) & (data[target] < 2)],
                   data['stars'][(data[target] >= 0) & (data[target] < 1)],
               data['stars'][(data[target] >= 3) & (data[target] < 4)])

print('f=%f, p=%f ' %(tmp[0],tmp[1]))


