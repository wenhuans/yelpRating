import numpy as np
import pandas as pd
from scipy import sparse

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, GRU
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, metrics
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from keras.models import load_model
import pickle

model = load_model('yelp_1M_text_only.h5')

import numpy as np
import pandas as pd
from scipy import sparse

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, GRU
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

path = 'yelp_review.csv'
data = pd.read_csv(path, encoding = "latin",nrows = 100000) # todo: will this encoding type affects data?
for col in data.columns:
    print(col)
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
print('data peek finished...')
print('num of rows before purging = ', data.shape)
valid_index = []
tmp = data['cool'].values

for i in range(data.shape[0]):
    #print('*** type check', type(tmp[i]))
    if float(tmp[i]) <= 100:
        valid_index.append(i)

data = data.loc[valid_index,:]
print('num of rows after purging = ', data.shape)

data_all = data[['text','stars']]
# data_all['stars'].hist()
# data_all.head()
# plt.show()




data_all_one_hot = pd.get_dummies(data_all, columns = ['stars'])
print('size of the data set, text only with one-hot encoding of stars',data_all_one_hot.shape)

num_review = data_all_one_hot.shape[0]
split_index = int(num_review * 0.8)
data_train_one_hot = data_all_one_hot[:split_index]
data_test_one_hot = data_all_one_hot[split_index:]
num_train = data_train_one_hot.shape[0]
num_test = data_test_one_hot.shape[0]
# print('size of all reviews = %d, train = %d, test = %d' % (num_review, num_train, num_test))
# I'm using GLoVe word vectors to get pretrained word embeddings
embed_size = 200
# max number of unique words
max_features = 20000
# max number of words from review to use
maxlen = 200

# File path
embedding_file = 'glove.twitter.27B.200d.txt'

# read in embeddings
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
print('computing embeddings matrix...')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file))
print('embeddings matrix populated')

class_names = ['stars_1', 'stars_2', 'stars_3', 'stars_4', 'stars_5']
# Splitting off my y variable
y = data_train_one_hot[class_names].values

print('start tokenizing texts...')
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(data_train_one_hot['text'].values))
X_train = tokenizer.texts_to_sequences(data_train_one_hot['text'].values)
X_test = tokenizer.texts_to_sequences(data_test_one_hot['text'].values)
x_train = pad_sequences(X_train, maxlen = maxlen)
x_test = pad_sequences(X_test, maxlen = maxlen)
print('tokenization finished. ')
print('start handling missed word...')
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))
# create a zeros matrix of the correct dimensions
embedding_matrix = np.zeros((nb_words, embed_size))
missed = []
for word, i in word_index.items():
    if i >= max_features: break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        missed.append(word)
print('matrix updated. Num of words not present in the reference vocabulary:%d' % len(missed))
print('saving the tokeninzed data...')


print('model evaluation...')
y_test = model.predict([x_test], batch_size=1024, verbose = 1)
score, acc = model.evaluate(x_test, data_test_one_hot[class_names].values, verbose = 1, batch_size=1024)
print('from Keras accuracy = ', acc)

v = metrics.classification_report(np.argmax(data_test_one_hot[class_names].values, axis = 1),np.argmax(y_test, axis = 1))
print(v)
v = metrics.confusion_matrix(np.argmax(data_test_one_hot[class_names].values, axis = 1),np.argmax(y_test, axis = 1))
print('confusion matrix = \n',v)

plt.imshow(v, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(np.arange(5), ('1','2','3','4','5'))
plt.yticks(np.arange(5), ('1','2','3','4','5'))
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()






