# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency,boxcox,skew
from scipy.stats import stats
from sklearn.preprocessing import LabelEncoder ,StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_validate,GridSearchCV,RandomizedSearchCV
from sklearn.decomposition import PCA,NMF,FactorAnalysis,TruncatedSVD,FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import FeatureAgglomeration
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.metrics import accuracy_score,log_loss,f1_score,roc_auc_score,confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import RFECV
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from catboost import Pool
from imblearn.over_sampling import smote
sample_submission = pd.read_csv('./Dataset/sample_submission.csv')
train = pd.read_csv('./Dataset/train.csv')
test = pd.read_csv('./Dataset/test.csv')
data = pd.read_csv('./Dataset/useful_data0.csv')
data_head = data.head(25)

def find_remove_skew(data,cols):
    skew_feat = data[cols].apply(lambda x : skew(x))
    print(skew_feat)
    skew_feat =skew_feat[skew_feat>0.3].index
    for feat in skew_feat:
        if(data[feat].min() < 0 ):
            data[feat],_ = boxcox(data[feat]+1 - data[feat].min())
        else:
            data[feat],_ = boxcox(data[feat]+1)

def break_num_features(data):
    cat_feature = ['district_id','building_type','family_category','ward_id','vdcmun_id']
    num_feature = [x for x in data.columns if x not in cat_feature and x not in ['building_id', 'damage_grade']]
    binary_feature = [x for x in num_feature if data[x].nunique()==2]
    continues_feature = ['plinth_area_sq_ft','building_volume','height_ft_diff','height_ft_pre_eq','height_ft_post_eq']
    discrete_feature = [x for x in num_feature if x not in binary_feature and 
                   x not in continues_feature ]
    return cat_feature,binary_feature,discrete_feature,continues_feature


def calc_multiclass(preds,num_data):
    #    or multi-class task, the y_pred is group by class_id first, then group by row_id.
    #    If you want to get i-th row y_pred in j-th class, the access way is
    #    y_pred[j * num_data + i].
    df = pd.DataFrame()
    for i in range(1,6):
        df[i] = preds[(i-1)*num_data:(i)*num_data]
    
    pred = df.idxmax(axis=1).astype('int8')
    return pred.values


#TODO custom function
def custom_f1_weighted(preds,dtrain):
    y = dtrain.get_label() + 1
    preds = calc_multiclass(preds,len(y))
    score = f1_score(y,preds,average='weighted')
    return 'f1', score, True

from scipy.stats import ks_2samp

def koglomorov_test(train,test):
    threshold_p = 0.01
    threshold_stat = 0.3
    diff_col = []
    columns = train.columns
    for col in columns:
        stat,pvalue = ks_2samp(train[col].values,test[col].values)
        print(pvalue)
        if pvalue <= threshold_p and np.abs(stat) > threshold_stat:
            diff_col.append(col)
        
    print(diff_col)
    
    
# break feature into cat,discrete,binary,continues
cat_feature,binary_feature,discrete_feature,continues_feature = break_num_features(data)

#remove skewness from continues_feature feature
find_remove_skew(data,continues_feature)
#remove skewness from descrete feature
find_remove_skew(data,discrete_feature)

# [with descrete skewness]   train's multi_logloss: 0.398705 valid's multi_logloss: 0.485917
# train's multi_logloss: 0.407632 valid's multi_logloss: 0.485661

# features to remove
#remove_features = remove_correlated_feature(data)
#remove_features = remove_features +[c for c in num_feature if c.startswith('has_secondary_use_')]
#data.drop(labels=remove_features,inplace=True,axis=1)
#sns.heatmap(xtest[continues_feature].corr(),annot=True)

train_x = data[:len(train)]
train_y = train_x.damage_grade.values
dtrain_y = data[:len(train)]['damage_grade']
train_x.drop(['building_id','damage_grade'],inplace=True,axis=1)
test_x = data[len(train):]
test_x.drop(['building_id','damage_grade'],inplace=True,axis=1)
train_test_head = train_x.head(12)

#This is a two-sided test for the null hypothesis that whether 2 independent samples are drawn from the same continuous distribution
#koglomorov_test(train_x,test_x)


xtrain , xtest , ytrain ,ytest = train_test_split(train_x,train_y,test_size=0.2,random_state=512,stratify=train_y)


# scaling 
#continues_feature = continues_feature + ['district_count','ward_count','vdcmun_count']
continues_feature = continues_feature + discrete_feature
scaler = StandardScaler()
scaler.fit(xtrain[continues_feature])
xtrain_s = scaler.transform(xtrain[continues_feature])
xtest_s = scaler.transform(xtest[continues_feature])

#
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# stacking ensemble tree models
#class to extend the Sklearn Classifier

class SklearnHelper(object):
    def __init__(self,clf,seed=512,params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self,train_x,train_y):
        self.clf.fit(train_x,train_y)
        
    def predict(self,x):
        return self.clf.predict(x)
    
    def predict_prob(self,x):
        return self.clf.predict_proba(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        return (self.clf.fit(x,y).feature_importances_)


class XgbWrapper(object):
    def __init__(self, seed=512, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 400)

    def train(self, xtra, ytra, xte, yte):
        dtrain = xgb.DMatrix(xtra, label=ytra)
        dvalid = xgb.DMatrix(xte, label=yte)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds,
            watchlist, early_stopping_rounds=10)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

class CatWrapper(object):
    def __init__(self,cat_index=None, seed=512,params=None):
        self.param = params
        self.cat_index = cat_index
        self.param['seed'] = seed

    def train(self, xtra, ytra, xte, yte):
        dtrain = Pool(xtra,ytra,cat_features=self.cat_index)
        dvalid = Pool(xte,yte,cat_features=self.cat_index)
        self.catb = ctb.CatBoostClassifier(learning_rate=0.5,depth=8,iterations= 500,loss_function='MultiClass') 
        self.catb.fit(dtrain,use_best_model=True, eval_set=dvalid)

    def predict(self, x):
        return self.catb.predict_proba(Pool(x,cat_features=self.cat_index))


class LgbWrapper(object):
    def __init__(self, seed=512, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 2000)

    def train(self, xtra, ytra, xte, yte):
        ytra = ytra.ravel()
        yte = yte.ravel()
        dtrain = lgb.Dataset(xtra, label=ytra,categorical_feature=cat_index)
        dvalid = lgb.Dataset(xte, label=yte,categorical_feature=cat_index)
        self.gbdt = lgb.train(self.param, dtrain,valid_sets=[dtrain,dvalid],valid_names=['train','valid'],num_boost_round=self.nrounds,early_stopping_rounds=20,verbose_eval=10)

    def predict(self, x):
        return self.gbdt.predict(x)


def get_oof_prediction(clf,x_train,y_train,x_test,clf_type=0):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        print ('fold numer = '+str(i))
        x_tr = x_train[train_ind]
        y_tr = y_train[train_ind]
        x_ts = x_train[test_ind]
        y_ts = y_train[test_ind]
        if clf_type==0:
            clf.train(x_tr,y_tr)
            oof_train[test_ind] = clf.predict(x_ts)
            oof_test_skf[i,:] = clf.predict(x_test)
        else:    
            clf.train(x_tr,y_tr,x_ts,y_ts)
            oof_train[test_ind] = np.argmax(clf.predict(x_ts), axis=1)
            oof_test_skf[i,:] = np.argmax(clf.predict(x_test), axis=1)
        
    oof_test = stats.mode(oof_test_skf,axis=0)[0]
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)


def get_oof_predictionprob(clf,x_train,y_train,x_test,SEED,NFOLDS,clf_type=0):
    skf = StratifiedKFold(n_splits=NFOLDS,random_state=SEED)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.zeros((ntest,5))
    oof_train_skf = np.zeros((ntrain,5))
    
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
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1),oof_test_skf,oof_train_skf


# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 150,
    'criterion': 'entropy',
#    'warm_start': True, 
    'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 100,
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':200,
    'criterion': 'entropy',
    'max_features': 0.5,
    'max_depth': 11,
    'min_samples_leaf': 100,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 50,
    'learning_rate' : 0.01
}

#Extreme Gradient Boosting parameters
xgb_params = {
    'booster':'gbtree',
    'objective':'multi:softprob',
    'learning_rate':0.3,   
    'num_class':5,
    'n_estimators': 200,
    'max_depth': 7,
    'min_samples_leaf': 100,
    'colsample_bytree':0.55,
    'subsample':0.9,
#    'gamma':0.0468,
#    'reg_lambda':0.4640,
#    'reg_alpha':0.8571,
    'eval_metric':'mlogloss',
    'verbose': 0,
    'nthread':-1,    
}        

#lgbm parameters

lgb_params = {
    'boosting_type':'gbdt',
    'objective':'multiclass',
    'metric':'multi_logloss',
    'learning_rate':0.24,  
    'n_estimators': 2000,
    'max_depth': 8,
    'min_samples_leaf': 100,
    'colsample_bytree':0.55,
    'subsample':0.9,
#    'gamma':0.0468,
#    'reg_lambda':0.4640,
#    'reg_alpha':0.8571,
    'verbose': 0,
    'nthread':-1,
    'num_leaves':55,
    'num_class':5
}     

cat_params ={
    'learning_rate':.5, 
    'depth':8, 
    'n_estimators': 300,
#    'iterations': 500,
    'early_stopping_round': 20,
    'loss_function':'MultiClass'
    }   


from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin,tsec = divmod((datetime.now()-start_time).total_seconds(),60)
        print('\n Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))
        

SEED=512
ranking = pd.read_csv('RFECV-ranking-final.csv')
not_sel_features = ranking[ranking.Rank>2]['Features'].values
predictors = list([x for x in train_x.columns if x not in list(not_sel_features)])
cat_feature = ['district_id','ward_id','vdcmun_id']
cat_feature = cat_feature + [x for x in predictors if train_x[x].nunique()==2]
cat_index =[train_x[predictors].columns.get_loc(c) for c in cat_feature ]
ntrain_x = train_x[predictors].values
ntest_x = test_x[predictors].values
ntrain = train_x.shape[0]
ntest = test_x.shape[0]

rfc = SklearnHelper(RandomForestClassifier,seed=SEED,params=rf_params)
etc = SklearnHelper(ExtraTreesClassifier,seed=SEED,params=et_params)
xgbm = XgbWrapper(seed=SEED,params=xgb_params)

#gini warmstart false
rf_oof_train,rf_oof_test,rf_oof_test_prob = get_oof_predictionprob(rfc,ntrain_x,train_y,ntest_x,512,5,clf_type=0)

#entropy warmstart false
rf_oof_train_wstart,rf_oof_test_wstart,rf_oof_test_prob_wstart = get_oof_predictionprob(rfc,ntrain_x,train_y,ntest_x,512,5,clf_type=0)

#entropy warmstart true
rf_oof_train_entropy,rf_oof_test_entropy,rf_oof_test_prob_entropy = get_oof_predictionprob(rfc,ntrain_x,train_y,ntest_x,512,5,clf_type=0)

avg_rf_prob = rf_oof_test_prob_entropy+rf_oof_test_prob_wstart+rf_oof_test_prob


et_oof_train,et_oof_test,et_oof_test_prob = get_oof_predictionprob(etc,ntrain_x,train_y,ntest_x,512,5,clf_type=0)

start = timer(None)
catb = CatWrapper(cat_index=cat_index,seed=SEED,params=cat_params)
cat_oof_train,cat_oof_test,cat_oof_test_prob,cat_oof_train_prob = get_oof_predictionprob(catb,ntrain_x,train_y,ntest_x,512,5,clf_type=1)

xgbm = XgbWrapper(seed=SEED,params=xgb_params)
xgb_oof_train,xgb_oof_test,xgb_oof_test_prob,xgb_oof_train_prob = get_oof_predictionprob(xgbm,ntrain_x,train_y,ntest_x,512,3,clf_type=1)
timer(start)

start = timer(None)
lgbm = LgbWrapper(seed=SEED,params=lgb_params)
lgb_oof_train,lgb_oof_test,lgb_oof_test_prob,lgb_oof_train_prob = get_oof_predictionprob(lgbm,ntrain_x,train_y,ntest_x,512,7,clf_type=1)
timer(start)


res = xgb_oof_test_prob+cat_oof_test_prob+lgb_oof_test_prob
pre = np.argmax(res,axis=1)

np.save('avg_rf_prob.npy',avg_rf_prob)
np.save('rf_oof_test_prob.npy',rf_oof_test_prob)
np.save('rf_oof_test_entropy.npy',rf_oof_test_prob_wstart)
np.save('rf_oof_test_entropy_wstart.npy',rf_oof_test_prob_entropy)
np.save('et_oof_test_prob.npy',et_oof_test_prob)
np.save('./result/lightgbm/lgb_oof_test_prob.npy',lgb_oof_test_prob)
np.save('./result/lightgbm/lgb_oof_train_prob.npy',lgb_oof_train_prob)
np.save('./result/lightgbm/lgb_oof_train.npy',lgb_oof_train)
np.save('./result/lightgbm/lgb_oof_test.npy',lgb_oof_test)

avg_rf_prob = np.load('avg_rf_prob.npy')
et_oof_test_prob = np.load('et_oof_test_prob.npy')
lgb_oof_test_prob = np.load('lgb_oof_test_prob.npy')
xgb_oof_test_prob = np.load('xgb_oof_test_prob.npy')
cat_oof_test_prob = np.load('cat_oof_test_prob.npy')
xgb_oof_train_prob = np.load('xgb_oof_train_prob.npy')
cat_oof_train_prob = np.load('cat_oof_train_prob.npy')
xgb_oof_test3cv = np.load('xgb_oof_test.npy')

np.save('xgb_oof_train.npy',xgb_oof_train)
np.save('xgb_oof_test.npy',xgb_oof_test)
np.save('xgb_oof_test_prob.npy',xgb_oof_test_prob)
np.save('xgb_oof_train_prob.npy',xgb_oof_train_prob)

np.save('./result/lightgbm/cat_oof_train.npy',cat_oof_train)
np.save('./result/lightgbm/cat_oof_test.npy',cat_oof_test)
np.save('./result/lightgbm/cat_oof_test_prob.npy',cat_oof_test_prob)
np.save('./result/lightgbm/cat_oof_train_prob.npy',cat_oof_train_prob)


np.save('rf_oof_train10cv.npy',rf_oof_train)
np.save('rf_oof_test10cv.npy',rf_oof_test)
np.save('et_oof_train10cv.npy',et_oof_train)
np.save('et_oof_test10cv.npy',et_oof_test)
np.save('lgb_oof_train10cv.npy',lgb_oof_train)
np.save('lgb_oof_test10cv.npy',lgb_oof_test)
np.save('xgb_oof_train10cv.npy',xgb_oof_train)
np.save('xgb_oof_test10cv.npy',xgb_oof_test)
np.save('cat_oof_train10cv.npy',cat_oof_train)
np.save('cat_oof_test10cv.npy',cat_oof_test)


rf_oof_train10cv = np.load('rf_oof_train10cv.npy')
rf_oof_test10cv = np.load('rf_oof_test10cv.npy')
et_oof_train10cv = np.load('et_oof_train10cv.npy')
et_oof_test10cv = np.load('et_oof_test10cv.npy')
lgb_oof_train10cv = np.load('lgb_oof_train10cv.npy')
lgb_oof_test10cv = np.load('lgb_oof_test10cv.npy')
xgb_oof_train10cv = np.load('xgb_oof_train10cv.npy')
xgb_oof_test10cv = np.load('xgb_oof_test10cv.npy')


xgb_oof_test = np.load('./ensemble/xgb_oof_test.npy')


knn_oof_train = np.load('knn_oof_train.npy')
knn_oof_test = np.load('knn_oof_test.npy')
sgd_oof_train = np.load('sgd_oof_train.npy')
sgd_oof_test = np.load('sgd_oof_test.npy')
nb_oof_train = np.load('nb_oof_train.npy')
nb_oof_test = np.load('nb_oof_test.npy')

base_predictions_train = pd.DataFrame( {'RandomForest1': xgb_oof_test.ravel(),
                                        'RandomForest2': xgb_oof_test10cv.ravel(),
                                        'RandomForest3': xgb_oof_test3cv.ravel(),
#     'ExtraTrees': et_oof_train.ravel(),
#     'LightGbM': lgb_oof_train.ravel(),
#      'XGBoost': xgb_oof_train.ravel(),
#      'KNN':knn_oof_train.ravel(),
#      'SGD':sgd_oof_train.ravel(),
#      'NB':nb_oof_train.ravel()
    })

base_predictions_train.head()

sns.heatmap(base_predictions_train.astype(float).corr(),annot=True)

ens_trainx = np.concatenate((rf_oof_train,et_oof_train,xgb_oof_train,lgb_oof_train),axis=1)
ens_testx = np.concatenate((rf_oof_test,et_oof_test,xgb_oof_test,lgb_oof_test),axis=1)


ens_trainx = np.concatenate((xgb_oof_train_prob,cat_oof_train_prob),axis=1)
ens_testx = np.concatenate((xgb_oof_test_prob,cat_oof_test_prob),axis=1)



df = pd.DataFrame(data=ens_trainx,columns=['x1','x2','x3','x4','x5','c1','c2','c3','c4','c5'])



xtrain,xtest,ytrain,ytest = train_test_split(df,train_y,stratify=train_y,test_size=0.2,random_state=SEED)

scaler = StandardScaler()
xtrain_s = scaler.fit_transform(xtrain)
xtest_s = scaler.transform(xtest)

pca = PCA()
xtrain_s = pca.fit_transform(xtrain_s)
xtest_s = pca.transform(xtest_s)
evr = pca.explained_variance_ratio_
xtrain['pca1'] = xtrain_s[:,0]
xtrain['pca2'] = xtrain_s[:,1]
xtrain['pca3'] = xtrain_s[:,2]
xtest['pca1'] = xtest_s[:,0]
xtest['pca2'] = xtest_s[:,1]
xtest['pca3'] = xtest_s[:,2]


predictors = ['pca1','pca2','pca3']

def fit_lgbm_model(dtrain_x, dtest_x, dtrain_y, dtest_y,params,predictors,
                   categorical_feature=None,objective = 'binary',
                   num_boost_round=300,early_stopping_rounds=10,
                   metrics = 'auc', feval=None,verbose_eval=10):
    
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 600,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0.001,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 4,
        'verbose': 0,
        'num_class':5
    }

    lgb_params.update(params)
    
    lgbTrain = lgb.Dataset(dtrain_x[predictors].values,label = dtrain_y,
                           feature_name=predictors,categorical_feature=categorical_feature)
    
    lgbValid = lgb.Dataset(dtest_x[predictors].values,label = dtest_y,
                           feature_name=predictors,categorical_feature=categorical_feature)

    evals_results = {}
    
    lgbm = lgb.train(lgb_params,
              lgbTrain,
              valid_sets=[lgbTrain,lgbValid],
              valid_names=['train','valid'],
              evals_result=evals_results, 
              num_boost_round=num_boost_round,
              early_stopping_rounds=early_stopping_rounds,
              verbose_eval=10,
              feval=feval)
    
    n_estimators = lgbm.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])
    return lgbm

params = {
        'learning_rate': 0.02,
        'num_leaves': 31,  # 2^max_depth - 1
        'max_depth': 5,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'subsample': .6,  # Subsample ratio of the training instance.
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'reg_alpha': 0.01,
        'seed':512
        }


model = fit_lgbm_model(xtrain, xtest,
                       ytrain, ytest,
                       lgb_params,predictors,
                       categorical_feature=None,
                       objective='multiclass', 
                       num_boost_round=2000,
                       early_stopping_rounds=100,
                       metrics = 'multi_logloss',
                       verbose_eval=True
                       )

#XGBOOST
dtrain = xgb.DMatrix(xtrain, label=ytrain)
dvalid = xgb.DMatrix(xtest, label=ytest)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Try different parameters! My favorite is random search :)
xgb_params = {
    'eta': 0.025,
    'max_depth': 7,
    'subsample': 0.80,
    'booster':'gbtree',
    'objective':'multi:softprob',
    'eval_metric': 'mlogloss',
    'lambda': 0.8,   
    'alpha': 0.4,
    'num_class':5,
    'silent': 1
}

model_xgb = xgb.train(xgb_params, dtrain, 2000, watchlist, early_stopping_rounds=300,
                  maximize=False, verbose_eval=15)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(xtrain,ytrain)
print("Train score {} ".format(mnb.score(xtrain, ytrain)))
print("Test score {} ".format(mnb.score(xtest, ytest)))

sgd  = SGDClassifier(max_iter=50,random_state=512)
sgd.fit(xtrain,ytrain)
print("Train score {} ".format(sgd.score(xtrain, ytrain)))
print("Test score {} ".format(sgd.score(xtest, ytest)))


from sklearn.linear_model import LogisticRegression,SGDClassifier
logReg  = LogisticRegression(max_iter=50,random_state=512,C=0.46723737976886515,penalty='l2',n_jobs=-1,verbose=2)
logReg.fit(xtrain,ytrain)
print("Train score {} ".format(logReg.score(xtrain, ytrain)))
print("Test score {} ".format(logReg.score(xtest, ytest)))


print('\Blending results with a Logistic Regression ... ')

blendParams = {'C':[1000],'tol':[0.01]} # test more values in your local machine
clf = GridSearchCV(LogisticRegression(solver='newton-cg', multi_class='multinomial'), blendParams, scoring='log_loss',
                   refit='True', n_jobs=-1, cv=5)
clf.fit(base_predictions_train,train_y)
print('The Best parameters of the blending model\n{}'.format(clf.best_params_))
print('The best score:{}'.format(clf.best_score_))




# prediction from crossvalidation
def get_predictioncv(cv_predictions,cv):
    res=pd.DataFrame()
    for i in range(10):
        for j in range(5):
            col  = 'Grade '+str(j+1)
            columns = res.columns
            if col not in columns:
                res[col] = cv_predictions[i][:,j]
            else:
                res[col] = res[col] + cv_predictions[i][:,j]
    
    damage = res.idxmax(axis=1)
    sub = pd.DataFrame({'building_id':data[len(train):]['building_id'],'damage_grade':damage.values}) 
    return sub

#prediction from model
def get_predictionfrommodel(predict):
    predictions = []
    for x in predict:
        predictions.append(np.argmax(x))
    
    submission = pd.DataFrame({'building_id':data[ntrain:]['building_id'],'damage_grade':predictions})
    for i in range(5):
        submission.loc[submission.damage_grade==i,'damage_grade'] = 'Grade '+str(i+1)
    
    return submission

#prediction from model
def get_predictionknn(predict):
    submission = pd.DataFrame({'building_id':data[ntrain:]['building_id'],'damage_grade':predict})
    for i in range(5):
        submission.loc[submission.damage_grade==i,'damage_grade'] = 'Grade '+str(i+1)
    
    return submission

submission = get_predictionknn(list(lgb_oof_test))
submission = get_predictioncv(cv_predictions,10)
submission.to_csv('lgb_oof_test7cv.csv',index=None)

sub = pd.read_csv('cv_ldaknnmodel1.csv')
sub1 = pd.read_csv('./result/lightgbm_with_feature_extraction/cv_model.csv')
sub2 = pd.read_csv('./result/score76091/cv_model2.csv')



df = pd.DataFrame()
lbl = LabelEncoder()
df['knn'] = lbl.fit_transform(sub.damage_grade.values)
df['lgbm'] = lbl.fit_transform(sub1.damage_grade.values)
df['lgbm_wm'] = lbl.fit_transform(sub2.damage_grade.values)

sns.heatmap(df.corr(),annot=True)

sub.damage_grade.value_counts()


