# -*- coding: utf-8 -*-
import matplotlib as mpl
from collections import Counter
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
import re
import sys
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_similarity_score
cv = CountVectorizer()
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency,boxcox,skew
from scipy.stats import stats
from sklearn.preprocessing import LabelEncoder ,StandardScaler,OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_validate,GridSearchCV,RandomizedSearchCV
from sklearn.decomposition import PCA,NMF,FactorAnalysis
from sklearn.metrics import accuracy_score,log_loss,f1_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import RFECV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import catboost as catb
from catboost import Pool
from sklearn.linear_model import LogisticRegression,SGDClassifier,Ridge,Lasso,RandomizedLasso
from sklearn.svm import SVC
from nltk.corpus import brown
from nltk.corpus import webtext
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
porter = PorterStemmer()
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk import pos_tag
from bs4 import BeautifulSoup
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from bs4 import Comment

path = './dataset/'
test = pd.DataFrame(pd.read_csv(path+'test.csv'))
train = pd.DataFrame(pd.read_csv(path+'train/train.csv'))
html_data = pd.DataFrame(pd.read_csv(path+'train/html_data.csv',usecols = ['Webpage_id']))
sample = pd.DataFrame(pd.read_csv(path+'sample_submission.csv'))

html_data_head = html_data.head(5)

def plot_numerical_feature(data,col,title='',bins=50,figsize = (8,6)):
    plt.figure(figsize = figsize)
    sns.distplot(data[col].dropna(),kde=True)
    plt.show()
    

def plot_categorical_feature(data,col,title='',xlabel_angle=0,figsize = (8,6)):
    plot_data = data[col].value_counts()
    plt.figure(figsize=figsize)
    sns.barplot(x=plot_data.values,y=plot_data.index)
    if (xlabel_angle > 0):
        plt.yticks(rotation = xlabel_angle)
    plt.show()
    
def plot_numerical_feature_bylabel(data,col,title='',bins=50,figsize = (6,20)):
    plot_data= []
    for i in range(1,6):
        plot_data.append(data.loc[data.damage_grade=='Grade '+str(i),col])

    fig , ax_arr = plt.subplots(5,1,figsize = figsize)
    
    for i in range(1,6):
        sns.distplot(plot_data[i-1].dropna(),kde=True,ax = ax_arr[i-1])
        ax_arr[i-1].set_title('Distribution of '+col+'Grade'+str(i))

#    fig.suptitle('Distribution of '+col)
    plt.show()

def plot_categorical_feature_bylabel(data,col,title='',xlabel_angle=0,figsize = (6,20)):
    plot_data = []
    for i in range(1,6):
        plot_data.append(data[data.damage_grade=='Grade '+str(i)][col].value_counts())
    
    fig , ax_arr = plt.subplots(5,1,figsize = figsize)
#    plt.subplot_tool()
    for i in range(1,6):
        sns.barplot(y = plot_data[i-1].index,x = plot_data[i-1].values,ax = ax_arr[i-1])
        ax_arr[i-1].set_title('Distribution of '+col+'Grade'+str(i))
        for tick in ax_arr[i-1].get_yticklabels():
            if (xlabel_angle > 0):
                tick.set_rotation(xlabel_angle)

#    fig.suptitle('Distribution of '+col)
    plt.show()

def count_missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percentage = (data.isnull().sum()*100/data.isnull().count()).sort_values(ascending=False)
    return pd.concat([total,percentage],keys = ['count','percentage'],axis=1)

def label_encode(data):
    columns = get_cat_feature(data)
    lb = LabelEncoder()
    for col in columns:
        data[col] = lb.fit_transform(data[col])


data = train.append(test)

def cleaner2(html):
    try:         
        soup = BeautifulSoup(html, "html.parser")    
        [x.extract() for x in soup.find_all('script')]
        [x.extract() for x in soup.find_all('style')]
        [x.extract() for x in soup.find_all('meta') ]
        [x.extract() for x in soup.find_all('noscript')]
        [x.extract() for x in soup.find_all(text=lambda text:isinstance(text, Comment))]
        # get text
        text = soup.get_text()
        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        text = cleaner1(text)
        
        text = tokenize(text)
    except TypeError as e: print(html,e)

    return text


        
def cleaner1(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace("http","")
    s = s.replace("www","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 
  
wnl = WordNetLemmatizer()
    
def lemmatize_sent(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]
           

def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    return [word for word in lemmatize_sent(text) 
            if word not in stoplist_combined
            and not word.isdigit()]

def tokenize(text):
    """
    sent_tokenize(): segment text into sentences
    word_tokenize(): break sentences into words
    """
    try:         
        tokens_ = [word_tokenize(s) for s in sent_tokenize(text)]
        tokens = []
        for token_by_sent in tokens_:
            tokens += token_by_sent
        tokens = list(filter(lambda t: t.lower() not in stoplist_combined and not t.isdigit(), tokens))
        filtered_tokens = [porter.stem(word) for word in tokens]
        filtered_tokens = [t for t in lemmatize_sent(str(filtered_tokens))]
        filtered_tokens = [w.lower() for w in filtered_tokens if len(w)>=3]
        return filtered_tokens
            
    except TypeError as e: print(text,e)

def get_stopwords():
    # Stopwords from stopwords-json
    stopwords_json = {"en":["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}
    stopwords_json_en = set(stopwords_json['en'])
    stopwords_nltk_en = set(stopwords.words('english'))
    stopwords_punct = set(punctuation)
    # Combine the stopwords. Its a lot longer so I'm not printing it out...
    stoplist_combined = set.union(stopwords_json_en, stopwords_nltk_en, stopwords_punct)
    return stoplist_combined


stoplist_combined = get_stopwords()

datas = html_data.merge(data,left_on='Webpage_id',right_on='Webpage_id',how='one_to_one')

s_data = []
for i in range(79345):
    print(i)
    s_data.append(cleaner2("%s" % (html_data['Html'][i]) )) 

np.save('html_res.npy',np.array(s_data))

html_res = np.load('html_res.npy')

import collections

cv = CountVectorizer(tokenizer=lambda doc: doc,max_features=500, lowercase=False)
cv = cv.fit(html_res)

html_vector = pd.DataFrame({'Webpage_id':html_data['Webpage_id'],'Html':html_res})
train_vect  = train.merge(html_vector,left_on='Webpage_id',right_on='Webpage_id',how='inner')
test_vect  = test.merge(html_vector,left_on='Webpage_id',right_on='Webpage_id',how='inner')
lbl_enc = LabelEncoder()
train_vect['Tag'] = lbl_enc.fit_transform(train_vect['Tag'].values)
trainDf = pd.DataFrame(cv.transform(train_vect['Html']).toarray())
testDf = pd.DataFrame(cv.transform(test_vect['Html']).toarray())
trainDf.columns = ['col' + str(x) for x in trainDf.columns]
testDf.columns = ['col' + str(x) for x in testDf.columns]


xtrain , xtest , ytrain ,ytest = train_test_split(trainDf,train_vect['Tag'],test_size=0.2,random_state=512,stratify=train_vect['Tag'])


def get_oof_predictionprob(clf,x_train,y_train,x_test,SEED,NFOLDS,clf_type=0):
    skf = StratifiedKFold(n_splits=NFOLDS,random_state=SEED)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.zeros((ntest,9))
    oof_train_skf = np.zeros((ntrain,9))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        print ('fold numer = '+str(i))
        x_tr = x_train[train_ind]
        y_tr = y_train[train_ind]
        x_ts = x_train[test_ind]
        y_ts = y_train[test_ind]
        if clf_type==0:
            clf.train(x_tr,y_tr)
            oof_train_skf[test_ind] = clf.predict_prob(x_ts)
            oof_test_skf =oof_test_skf + clf.predict_prob(x_test)
        else:    
            clf.train(x_tr,y_tr,x_ts,y_ts)
            oof_train_skf[test_ind] = clf.predict(x_ts)
            oof_test_skf = oof_test_skf + clf.predict(x_test)
        
        print("f1 score for fold : ",f1_score(y_ts,np.argmax(oof_train_skf[test_ind], axis=1),average='weighted'))
  
    oof_test = np.argmax(oof_test_skf, axis=1)
    oof_train = np.argmax(oof_train_skf, axis=1)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1),oof_test_skf


#lgbm parameters

lgb_params = {
    'boosting_type':'gbdt',
    'objective':'multiclass',
    'metric':'multi_logloss',
    'learning_rate':0.1,  
    'n_estimators': 300,
    'max_depth': 7,
    'min_samples_leaf': 100,
    'colsample_bytree':0.6,
    'subsample':0.8,
#    'gamma':0.0468,
#    'reg_lambda':0.4640,
#    'reg_alpha':0.8571,
    'verbose': 0,
    'nthread':-1,
    'num_leaves':48,
    'early_stopping_round': 20,
    'num_class':9
}     


class LgbWrapper(object):
    def __init__(self, seed=512, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 300)

    def train(self, xtra, ytra, xte, yte):
        ytra = ytra.ravel()
        yte = yte.ravel()
        dtrain = lgb.Dataset(xtra, label=ytra)
        dvalid = lgb.Dataset(xte, label=yte)
        watchlist = [dvalid]
        self.gbdt = lgb.train(self.param, dtrain, self.nrounds,watchlist,verbose_eval=10)

    def predict(self, x):
        return self.gbdt.predict(x)



from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin,tsec = divmod((datetime.now()-start_time).total_seconds(),60)
        print('\n Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))
        
# stacking ensemble tree models
ntrain = trainDf.shape[0]
ntest = testDf.shape[0]
SEED=512
predictors = list(trainDf.columns)
ntrain_x = trainDf[predictors].values
ntest_x = testDf[predictors].values
lgbm = LgbWrapper(seed=SEED,params=lgb_params)
catb = CatWrapper(cat_index=None,seed=SEED,params=cat_params)

start = timer(None)
lgb_oof_train,lgb_oof_test,lgb_oof_test_prob = get_oof_predictionprob(lgbm,ntrain_x,train_vect['Tag'],ntest_x,512,5,clf_type=1)
timer(start)


def get_predictionfromcv(predict):
    tags = ['clinicalTrials','conferences','forum','guidelines','news','others','profile','publication','thesis']
    submission = pd.DataFrame({'Webpage_id':test_vect['Webpage_id'],'Tag':predict})
    for i in range(9):
        submission.loc[submission.Tag==i,'Tag'] = str(tags[i])
    
    return submission


sub = get_predictionfromcv(list(lgb_oof_test))
sub.Tag.value_counts()
sub.to_csv('lgbm.csv',index=None)
