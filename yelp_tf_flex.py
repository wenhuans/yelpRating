import numpy as np
import pandas as pd
from scipy import sparse
import keras
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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import string


path = 'YelpData/Yelp_train.csv'

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
print('data peek finished...')
print('num of rows before purging = ', data.shape)
valid_index = []
tmp = data['cool'].values
tmp1 = data['sentiment_score'].values
tmp2 = data['favorites'].values
tmp3 = data['horrible'].values

# feature_names = ['sentiment_score']
# feature_names = ['sentiment_score', 'useful','funny','cool','nchar','nword','gem','incredible','perfection','phenomenal'
#     ,'divine','die','highly','superb','heaven','amazing','favorites','perfect','gross','poorly','flavorless','waste'
#     ,'terrible','tasteless','rude','awful','inedible','horrible','apology','disgusting','worst']
feature_names = ['sentiment_score', 'useful','funny','cool','gem','incredible','perfection','phenomenal'
    ,'divine','die','highly','superb','heaven','amazing','favorites','perfect','gross','poorly','flavorless','waste'
    ,'terrible','tasteless','rude','awful','inedible','horrible','apology','disgusting','worst']
feature_len = len(feature_names)


for i in range(data.shape[0]):
    #print('*** type check', type(tmp[i]))
    if float(tmp1[i]) <= 100 and float(tmp2[i]) <= 100 and float(tmp3[i]) <= 100:
        valid_index.append(i)

data = data.loc[valid_index,:]
data.fillna(0) # replace nan (in sentiment_score ,etc
print('num of rows after purging = ', data.shape)

# # Text cleaning
# print('text cleaning ...')
#
# stop_words = set(stopwords.words('english'))
# counter = -1
# for row in data['text']:
#     counter += 1
#     if (counter % 200 == 0):
#         print('@ row %d' %counter)
#     line = word_tokenize(row)  #tokenize
#     line = [s.translate(str.maketrans('', '', string.punctuation)) for s in line] #remove punctuation
#     while('' in line) :
#         line.remove('')    #remove empty elements
#     line = [w.lower() for w in line]    #to lower case
#     line = [w for w in line if not w in stop_words] #remove stop words
#     line = TreebankWordDetokenizer().detokenize(line)   #de-tokenize
#
#     # data['text'][counter] = line
#     data.at[counter, 'text']=line

# counter = 0
# for i in range(data.shape[0]):
#     if not (data['sentiment_score'].values[i] >= - 5.0):
#         print('before',data['sentiment_score'][i])
#         data['sentiment_score'][i] = 0
#         print('after',data['sentiment_score'][i])
#         counter +=1
# print('************ num of NA sentiment score = ', counter)
import copy
compiler = copy.deepcopy(feature_names)
compiler.extend(['text','stars'])
print('compiler = ',compiler)
# data_all = data[['text','stars','sentiment_score','favorites','horrible']]
data_all = data[compiler]
# data_all['stars'].hist()
# data_all.head()
# plt.show()

data_all_one_hot = pd.get_dummies(data_all, columns = ['stars'])
print('size of the data set, text only with one-hot encoding of stars',data_all_one_hot.shape)
# print('before shuffle: ', data_all_one_hot[:5])
# pre split shuffle to further randomize data entries
data_all_one_hot = data_all_one_hot.sample(frac=1).reset_index(drop=True)
# print('after shuffle: ', data_all_one_hot[:5])
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
print(data_train_one_hot.columns)
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
train_len = x_train.shape[0]
test_len = x_test.shape[0]
print('******************size check sentiment --> x_train',data_train_one_hot[feature_names].values.reshape((train_len,feature_len)).shape, x_train.shape)
print('******************************* check here y labels', y[0:4])
print(' senti check',data_train_one_hot[feature_names].values.reshape((train_len,feature_len))[0:4])


# x_train = np.concatenate((x_train,data_train_one_hot['sentiment_score'].values.reshape((train_len,1))), axis = 1)
# x_test = np.concatenate((x_test,data_test_one_hot['sentiment_score'].values.reshape((test_len,1))), axis = 1)

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


print('model setup')
inp = Input(shape = (maxlen,))
x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = True)(inp)
x = SpatialDropout1D(0.5)(x)
x = Bidirectional(LSTM(40, return_sequences=True))(x)
x = Bidirectional(GRU(40, return_sequences=True))(x)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
conc = concatenate([avg_pool, max_pool])
# auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(x)
auxiliary_input = Input(shape=(feature_len,), name='aux_input') # todo
x = keras.layers.concatenate([conc, auxiliary_input]) # todo

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outp = Dense(5, activation = 'sigmoid')(x)
# data_train_one_hot['sentiment_score'].values.reshape((train_len,1))
# data_test_one_hot['sentiment_score'].values.reshape((test_len,1))
model = Model(inputs = [inp, auxiliary_input], outputs = outp)
# patience is how many epochs to wait to see if val_loss will improve again.
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3)
checkpoint = ModelCheckpoint(monitor = 'val_loss', save_best_only = True, filepath = 'yelp_lstm_gru_weights.hdf5')
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) #todo
print('fitting the model...')
history = model.fit([x_train,data_train_one_hot[feature_names].values.reshape((train_len,feature_len))], y, batch_size = 512, epochs = 20, validation_split = .1,
          callbacks=[earlystop, checkpoint])

model.save('yelp_nn_mode_sentiment_cleaned.h5')
print('model saved')

import pickle
# Saving the objects:
with open('objs_tf_sentiment_cleaned.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x_train, x_test, y, data_test_one_hot], f)
print('data saved to file')

print('model evaluation...')
y_test = model.predict([x_test,data_test_one_hot[feature_names].values.reshape((test_len,feature_len))], batch_size=1024, verbose = 1)
tmp= model.evaluate([x_test, data_test_one_hot[feature_names].values.reshape((test_len,feature_len))], data_test_one_hot[class_names].values, verbose = 1, batch_size=1024)
print('from Keras accuracy = ', tmp)
print(model.metrics_names)
print('double check, type of data_test_one_hot = ', type(data_test_one_hot))
v = metrics.classification_report(np.argmax(data_test_one_hot[class_names].values, axis = 1),np.argmax(y_test, axis = 1))
print(v)
v = metrics.confusion_matrix(np.argmax(data_test_one_hot[class_names].values, axis = 1),np.argmax(y_test, axis = 1))
print('confusion matrix = \n',v)
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])  # RAISE ERROR
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig ("LSTM_Acc.png",dpi=600)
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss']) #RAISE ERROR
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig ("LSTM_Loss.png",dpi=600)
plt.show()


# or
#cm = np.array([[1401,    0],[1112, 0]])

plt.imshow(v, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks(np.arange(5), ('1','2','3','4','5'))
plt.yticks(np.arange(5), ('1','2','3','4','5'))
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()


