# =============================================================================
#Task: Classifier which takes in a job description and gives the department name for it
#
#1: Read files from docs and convert to pandas Dataframe
#2: Merge with document_departments on Doc ID
#3: Clean Descriptions attribute 
#5: Tokenize the Descriptions and Load Glove vector for embedding 
#6: Split dataset train,test set with 10 percent test data
#7: Out OF Cross Validation with Stratified KFold used
#8: Used simple LSTM with glove embeddings and two dense layers 

#Assumptions:
#1: Removed samples with no descriptions
#2: Removing samples of minority class ( class less than 10 count )


#Requirements
#Download GloVe from the NLP lab at Stanford and add in folder data
#http://nlp.stanford.edu/data/glove.42B.300d.zip
# =============================================================================



import pandas as pd
import numpy as np
import os
import json 
from tqdm import tqdm
from pandas.io.json import json_normalize #package for flattening json in pandas df
from string import punctuation
import re
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)

#deep learning library
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.layers import SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
#sklearn library
from sklearn import preprocessing,metrics
from sklearn.model_selection import train_test_split,StratifiedKFold
# nltk library
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Initialize NLTK function
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin,tsec = divmod((datetime.now()-start_time).total_seconds(),60)
        print('\n Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))
        
# function to read docs folder convert into csv file by extracting description of each doc
def read_file():    
    files = os.listdir('./data/docs')
    docs = pd.DataFrame()
    
    for fileName in files:
        with open('./data/docs/'+ fileName) as f:
            d = json.load(f)
        doc = json_normalize(d)
        docs = docs.append(doc)
    
    docs = docs[['jd_information.description','_id']]
    docs.columns = ['Descriptions','Document ID']
    docs['Document ID'] = docs['Document ID'].astype('int64')
    
    return docs
 
# Remove Stopwords    
def remove_stopwords(text):
    stopwords_nltk_en = set(stopwords.words('english'))
    stopwords_punct = set(punctuation)
    stoplist_combined = set.union(stopwords_nltk_en, stopwords_punct)
    tokens = text.split()
    for word in tokens:
        if word in stoplist_combined:
            tokens.remove(word)
            
    return ' '.join(tokens)


#Lemmatization
def lemmatize_word(w):
    try:
        x = lemma.lemmatize(w).lower()
        return x
    except Exception as e:
        return w


def lemmatize_sentence(text):
    x = [lemmatize_word(t) for t in text.split()]
    return " ".join(x)

# Stemming
def stemming_word(w):
    return stemmer.stem(w)


def stemming_sentence(text):
    x = [stemming_word(t) for t in text.split()]
    return " ".join(x)
    
# remove digit and punctuation
def cleaner1(text):
    text = text.lower()
    reg = r'[' + punctuation + ']' + "+"
    text = re.sub(reg,' ',text)
    text = re.sub("\d+", "", text)
    x = [t for t in text.split() if len(t)>=3]
    return " ".join(x)

# Preprocess Texts
def clean_text(text):
    text = cleaner1(text)
    text = remove_stopwords(text)
    text = stemming_sentence(text)
    text = lemmatize_sentence(text)
    return text

def clean_docs(docs):
#    remove rows which have empty descriptions
    docs.loc[docs['Descriptions']=='','Descriptions'] = np.nan
    docs.dropna(inplace=True)
    
#    remove Department which have counts less than 10 
    counts = docs['Department'].value_counts()
    docs.loc[docs['Department'].isin(counts[counts<10].index),'Department'] = np.nan
    docs.dropna(inplace=True)
    

#   cleaning descriptions texts 
    docs['Descriptions'] = docs['Descriptions'].apply(lambda x: clean_text(x))

#   remove most frequent words 
    most_freq = pd.Series(' '.join(docs['Descriptions']).split()).value_counts()[:16]
    most_freq = list(most_freq.index)
    
    docs['Descriptions'] = docs['Descriptions'].apply(lambda x: " ".join(x for x in x.split() if x not in most_freq))

#   remove least frequent words
    least_freq = pd.Series(' '.join(docs['Descriptions']).split()).value_counts()[-500:]
    least_freq = list(least_freq.index)
    
    docs['Descriptions'] = docs['Descriptions'].apply(lambda x: " ".join(x for x in x.split() if x not in least_freq))
    
    return docs

def get_embedding_index():
    # load the GloVe vectors in a dictionary:
    embeddings_index = {}
    f = open('./data/glove.42B.300d.txt')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()  
    
    return embeddings_index

# create an embedding matrix for the words we have in the dataset
def get_glove_vector(token,embeddings_index):
    word_index = token.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return word_index,embedding_matrix

 # A simple LSTM with glove embeddings and two dense layers        
def build_model(word_index,embedding_matrix,max_len):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                         300,
                         weights=[embedding_matrix],
                         input_length=max_len,
                         trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.85))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.85))
    
    model.add(Dense(12))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

def get_oof_prediction(x_train,y_train,x_test,word_index,embedding_matrix,max_len=100):
    ntrain = len(x_train)
    ntest = len(x_test)
    x_test = token.texts_to_sequences(x_test)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
    skf = StratifiedKFold(n_splits=4,random_state=512)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.zeros((ntest,12))
    oof_train_skf = np.zeros((ntrain,12))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(xtrain,ytrain)):
        print ('fold number = '+str(i))
        x_tr = token.texts_to_sequences(x_train[train_ind])
        x_ts = token.texts_to_sequences(x_train[test_ind])
        # zero pad the sequences
        x_tr = sequence.pad_sequences(x_tr, maxlen=max_len)
        x_ts = sequence.pad_sequences(x_ts, maxlen=max_len)
        # we need to binarize the labels for the neural net
        y_tr = np_utils.to_categorical(y_train[train_ind])
        y_ts = np_utils.to_categorical(y_train[test_ind])
        model = build_model(word_index,embedding_matrix,max_len)
        model.fit(x_tr, y=y_tr, batch_size=64, epochs=100, 
                  verbose=1, validation_data=(x_ts, y_ts), callbacks=[earlystop])

        oof_train_skf[test_ind] = model.predict(x_ts)
        oof_test_skf = oof_test_skf + model.predict(x_test)
        
        print("accuracy score for fold : ",i," is:",metrics.accuracy_score(y_train[test_ind],np.argmax(oof_train_skf[test_ind], axis=1)))
  
    oof_test = np.argmax(oof_test_skf, axis=1)
    oof_train = np.argmax(oof_train_skf, axis=1)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1),


if __name__ == '__main__':    
    
    print("Reading data folder ..... ")
    document_departments = pd.read_csv("./data/document_departments.csv")    
    docs = read_file()
    docs = docs.merge(document_departments, left_on='Document ID',right_on='Document ID',how='inner')

    print("Preprocessing descriptions ..... ")
    docs = clean_docs(docs)
    
    lbl_enc = preprocessing.LabelEncoder()
    train_y = lbl_enc.fit_transform(docs['Department'].values)
    train_x = docs['Descriptions'].values
    
   # using keras tokenizer here

    token = text.Tokenizer(num_words = None)
    token.fit_on_texts(list(train_x))

    print("Reading glove file  ..... ")    
    embeddings_index = get_embedding_index()
    word_index,embedding_matrix = get_glove_vector(token,embeddings_index)        
    
    xtrain, xvalid, ytrain, yvalid = train_test_split(train_x, train_y, 
                                                      stratify=train_y, 
                                                      random_state=42, 
                                                      test_size=0.1, shuffle=True)
    
    
    max_len = 150
    print("Running Out of Fold Cross Validation ..... ")
    start_time = timer(None)
    oof_train,oof_test = get_oof_prediction(xtrain,ytrain,xvalid,word_index,embedding_matrix,max_len)
    timer(start_time)
    print('accuracy after out of fold cv : ',metrics.accuracy_score(yvalid,oof_test))
