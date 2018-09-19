# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency,boxcox,skew
from scipy.stats import stats
from sklearn.preprocessing import LabelEncoder ,StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_validate,GridSearchCV,RandomizedSearchCV
from sklearn.decomposition import PCA,NMF,FactorAnalysis
from sklearn.metrics import accuracy_score,log_loss,f1_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import RFECV
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import smote
from catboost import CatBoostClassifier
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

#Principal Component Analysis (PCA) identifies the combination of components 
#(directions in the feature space) that account for the most variance in the data.

#We initialise pca choosing Minka’s MLE to guess the minimum number of 
#output components necessary
pca = PCA(random_state=512, whiten=True)
pca.fit(xtrain_s)
xtrain_s = pca.transform(xtrain_s)
xtest_s = pca.transform(xtest_s)
# new features as 90% varaince of descrete and continues feature represented by first 4 components
xtrain['pca1']=xtrain_s[:,0]
xtrain['pca2']=xtrain_s[:,1]
xtrain['pca3']=xtrain_s[:,2]
xtrain['pca4']=xtrain_s[:,3]

xtest['pca1']=xtest_s[:,0]
xtest['pca2']=xtest_s[:,1]
xtest['pca3']=xtest_s[:,2]
xtest['pca4']=xtest_s[:,3]

# plot of pca
plt.scatter(xtrain_s[:,0], xtrain_s[:,3], c=ytrain,  cmap='prism', alpha=0.4)
plt.xlabel('Component 1')
plt.ylabel('Componene 4')

er = pca.explained_variance_ratio_
exp_var_cum=np.cumsum(pca.explained_variance_ratio_)
plt.step(range(exp_var_cum.size), exp_var_cum)


#
seed = 512
ranking = pd.read_csv('RFECV-ranking-01.csv')
sel_features = ranking[ranking.Rank==1]['Features'].values

predictors = list(xtrain.columns)
cat_feature = cat_feature + [x for x in predictors if xtrain[x].nunique()==2]

# scaled features standard 
scaler = StandardScaler()
scaler.fit(xtrain[predictors])
xtrain_s = scaler.transform(xtrain[predictors])
xtest_s = scaler.transform(xtest[predictors])


# catboost classifier
cat_index =[xtrain[predictors].columns.get_loc(c) for c in cat_feature ]

from catboost import Pool
train_pool = Pool(xtrain,ytrain,cat_features=cat_index)
test_pool = Pool(xtest,ytest,cat_features=cat_index)
# Initialize CatBoostClassifier
model = CatBoostClassifier(thread_count=4,learning_rate=0.2,iterations=500,early_stopping_rounds=10, random_seed=512, loss_function="MultiClass")
# Fit model 
model.fit(train_pool, use_best_model=True, eval_set=test_pool)
# Get predicted classes
preds_class = model.predict(test_pool)
f1_score(ytest,preds_class,average='weighted')
5048083334,7820758297687722

ntest = len(xtest)
oof_test_skf = np.zeros((ntest,5))
oof_test_skf = model.predict_proba(test_pool)
oof_test_skf = oof_test_skf + model.predict_proba(test_pool)

params = {'learning_rate':[0.25,0.3,0.35,0.4,0.45,0.5]}
clf = CatBoostClassifier(iterations=1000,early_stopping_rounds=10, random_seed=512, loss_function="MultiClass")
randsearch = RandomizedSearchCV(clf,param_distributions=params,scoring='f1_weighted',n_jobs=-1,verbose=2,cv=3,random_state=512)
randsearch.fit(xtrain)



#QDA with predictors and selected features
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(xtrain_s,ytrain)
print("Train score {} ".format(qda.score(xtrain_s, ytrain)))
print("Test score {} ".format(qda.score(xtest_s, ytest)))

#Train score 0.6353698398125871 
#Test score 0.6336533362880185 
#with feature selection Train score 0.6161813821704445 
#Test score 0.6129810926531226 

# LDA  , KNN ,  KNN+LDA  
#The objective of LDA is to preserve the class separation information 
#whilst still reducing the dimensions of the dataset

#KNN too slow better to use with lda
knn = KNeighborsClassifier(n_neighbors=55,n_jobs=-1)
knn.fit(xtrain_s,ytrain)
print("Train score {} ".format(knn.score(xtrain_s, ytrain)))
print("Test score {} ".format(knn.score(xtest_s, ytest)))


#LDA performs better than qda
lda = LinearDiscriminantAnalysis()
lda.fit(xtrain_s,ytrain)
print("Train score {} ".format(lda.score(xtrain_s, ytrain)))
print("Test score {} ".format(lda.score(xtest_s, ytest)))

#Train score 0.6947357382550335 
#Test score 0.6933511669687304

#plot
exp_var_cum=np.cumsum(lda.explained_variance_ratio_)
plt.step(range(exp_var_cum.size), exp_var_cum)
plt.scatter(xtrain_s[:,0],xtrain_s[:,1],c=ytrain,cmap='prism',alpha=0.4)

#KNN+LDA
xtrain_s = lda.transform(xtrain_s)
xtest_s = lda.transform(xtest_s)
knnlda = KNeighborsClassifier(n_neighbors=55,n_jobs=-1)
knnlda.fit(xtrain_s,ytrain)
print("Train score {} ".format(knnlda.score(xtrain_s, ytrain)))
print("Test score {} ".format(knnlda.score(xtest_s, ytest)))

#usf0:Train score 0.7190448113207547 
#Test score 0.703315315030114 
#uf2:Train score 0.7188766303659617 
#Test score 0.7030224846264038 
#with feature selection Train score 0.7178715018361403 
#Test score 0.7020094497162711 

#LogisticRegression
from sklearn.linear_model import LogisticRegression,SGDClassifier
logReg  = LogisticRegression(random_state=seed,C=0.46723737976886515,penalty='l2',multi_class='ovr',solver='lbfgs',n_jobs=-1,verbose=2)
logReg.fit(xtrain,ytrain)
print("Train score {} ".format(logReg.score(xtrain, ytrain)))
print("Test score {} ".format(logReg.score(xtest, ytest)))
 
from scipy.stats import uniform as sp_uniform

params = {'penalty': ['l2'],
          'C': sp_uniform(0,1),
          'solver': ['sag','newton-cg','lbfgs'],
          'n_jobs': [-1]
}
classifier = LogisticRegression(random_state=512)
search_results = RandomizedSearchCV(estimator = classifier,
                                        param_distributions = params,
                                        n_iter = 10, n_jobs = 1,
                                        cv = 3, verbose = 3 )
search_results.fit(xtrain[predictors], ytrain)
search_results.best_score_

params2 = {'penalty': ['l1'],
          'C': sp_uniform(0,1),
          'solver': ['saga'],
          'n_jobs': [-1]
}
classifier1 = LogisticRegression(random_state=512)
search_results1 = RandomizedSearchCV(estimator = classifier,
                                        param_distributions = params,
                                        n_iter = 10, n_jobs = 1,
                                        cv = 3, verbose = 3 )
search_results1.fit(xtrain[predictors], ytrain)
search_results1.best_score_


#C=0.46723737976886515,penalty='l1',multi_class='ovr'

#uf0:Train score 0.7052935450170951 
#Test score 0.7027454828931644 
#with feature selection Train score 0.7039184183867291 
#Test score 0.7021519077505085
#lda+ Train score 0.6957745029758136 
#Test score 0.6951398067319335 
#SGD

SGD = SGDClassifier(max_iter=10,random_state=512)
SGD.fit(xtrain_s,ytrain)
print("Train score {} ".format(SGD.score(xtrain_s, ytrain)))
print("Test score {} ".format(SGD.score(xtest_s, ytest)))

#uf0: Train score 0.6898921267569963 
#Test score 0.688460107793246 
#uf2 :Train score 0.68713593769786 
#Test score 0.6856663474551455 
#with feature selection Train score 0.6893183329112321 
#Test score 0.6886104801627188 
#lda+ Train score 0.6777989268076485 
#Test score 0.67691309268478 
#hyperparameter tuning takes too much time
params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
#    "alpha" : [0.0001, 0.001, 0.01, 0.1],
#    "penalty" : ["l2", "l1", "none"],
}
SGD = SGDClassifier(max_iter=50,random_state=512,alpha=0.1,penalty='l2')
clf = GridSearchCV(SGD,param_grid=params,n_jobs=-1,scoring='f1_weighted')
clf.fit(xtrain_s,ytrain)

#SVM takes too much time we have to use with feature selection

from sklearn.svm import LinearSVC
svc = LinearSVC(random_state=seed)
svc.fit(xtrain_s,ytrain)
print("Train score {} ".format(svc.score(xtrain_s, ytrain)))
print("Test score {} ".format(svc.score(xtest_s, ytest)))

#Train score 0.6975235849056604 
#Test score 0.6955434378289396 
#with feature selection Train score 0.703267459161707 
#Test score 0.7010518151527863

#Naive Bayes performing very poor but better if all categorical features are onehotencoded
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
nb = MultinomialNB()
nb.fit(xtrain,ytrain)
print("Train score {} ".format(nb.score(xtrain, ytrain)))
print("Test score {} ".format(nb.score(xtest, ytest)))

#on featureset 0 : Train score 0.5992109345321008 
#Test score 0.599131005991151

#on featureset 2 : Train score 0.596605119032544 
#with feature selection Test score 0.5966142473862908 
#Train score 0.5067569171837406 
#Test score 0.5063908257025951



def remove_correlated_feature(data):
    # Threshold for removing correlated variables
    threshold = 0.85
    
    # Absolute value correlation matrix
    corr_matrix = data.corr().abs()
    
    # Upper triangle of correlations
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    # Select columns with correlations above threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Remove the columns
    print(to_drop)
    return to_drop

# lightgbm

# knnlda best score at 53->7041,55->.7042,61->7038
# knnlda removing colllinear best score at 53->7045,51->.7040,35->70309   
# logreg rem collinear best score 71->702832540580754,73->7029433412740497,75->7029275126035789

scores=[]              
for i in range(2,33):
    sel_features = ranking[ranking.Rank<i]['Features'].values
    predictors = list(sel_features)
    remove_features= remove_correlated_feature(xtrain[predictors])
    predictors = [ x for x in predictors if x not in remove_features]
    scaler = StandardScaler()
    scaler.fit(xtrain[predictors])
    xtrain_s = scaler.transform(xtrain[predictors])
    xtest_s = scaler.transform(xtest[predictors])
    logReg  = LogisticRegression(random_state=seed,multi_class='ovr',n_jobs=-1,verbose=2)
    logReg.fit(xtrain_s,ytrain)
#    lda = LinearDiscriminantAnalysis()
#    lda.fit(xtrain_s,ytrain)
#    xtrain_s = lda.transform(xtrain_s)
#    xtest_s = lda.transform(xtest_s)
#    knnlda = KNeighborsClassifier(n_neighbors=55,n_jobs=-1)
#    knnlda.fit(xtrain_s,ytrain)
    print('number features used: ',len(sel_features))
    print("Train score {} ".format(logReg.score(xtrain_s, ytrain)))
    score = logReg.score(xtest_s, ytest)
    print("Test score {} ".format(score))
    scores.append(score)


ntrain = train_x.shape[0]
ntest = test_x.shape[0]
SEED = 512
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS,random_state=SEED)

def get_knn_oof_prediction(x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        model = KNeighborsClassifier(n_neighbors=55)
        y_tr = y_train[train_ind]
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train[train_ind])
        x_ts = scaler.transform(x_train[test_ind])
        x_test_s = scaler.transform(x_test)
        lda = LinearDiscriminantAnalysis()
        x_tr = lda.fit_transform(x_tr,y_tr)
        x_ts = lda.transform(x_ts)
        x_test_s = lda.transform(x_test_s)           
        model.fit(x_tr,y_tr)
        oof_train[test_ind] = model.predict(x_ts)
        oof_test_skf[i,:] = model.predict(x_test_s)
        print("Test score {} ".format(model.score(x_ts,y_train[test_ind])))
        
    oof_test = stats.mode(oof_test_skf,axis=0)[0]
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)


def get_sgd_oof_prediction(SEED,x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        model = CatBoostClassifier(learning_rate=0.02,depth=6, l2_leaf_reg = 14, iterations = 500,loss_function='Logloss')
        y_tr = y_train[train_ind]
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train[train_ind])
        x_ts = scaler.transform(x_train[test_ind])
        x_test_s = scaler.transform(x_test)
        model.fit(x_tr,y_tr)
        oof_train[test_ind] = model.predict(x_ts)
        oof_test_skf[i,:] = model.predict(x_test_s)
        print("Test score {} ".format(model.score(x_ts, y_train[test_ind])))
        
    oof_test = stats.mode(oof_test_skf,axis=0)[0]
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

def get_naivebayes_oof_prediction(x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        model = MultinomialNB()
        y_tr = y_train[train_ind]
        x_tr = x_train[train_ind]
        x_ts = x_train[test_ind]
        model.fit(x_tr,y_tr)
        oof_train[test_ind] = model.predict(x_ts)
        oof_test_skf[i,:] = model.predict(x_test)
        print("Test score {} ".format(model.score(x_ts, y_train[test_ind])))
        
    oof_test = stats.mode(oof_test_skf,axis=0)[0]
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)


predictors = list(train_x.columns)
ranking = pd.read_csv('RFECV-ranking-01.csv')

logReg  = LogisticRegression(max_iter=50,random_state=512,C=0.46,penalty='l2',multi_class='multinomial',solver='newton-cg',n_jobs=-1,verbose=2)
logReg.fit(xtrain_s,ytrain)
print("Train score {} ".format(logReg.score(xtrain_s, ytrain)))
print("Test score {} ".format(logReg.score(xtest_s, ytest)))

remove_features= remove_correlated_feature(xtrain)

sel_features = ranking[ranking.Rank<=4]['Features'].values
remove_features  = ['pca1','pca2','pca3','pca4']
predictors = list(sel_features)
predictors = [ x for x in predictors if x not in remove_features]
ntrain_x = train_x[predictors].values
ntest_x = test_x[predictors].values

knn_oof_train,knn_oof_test = get_knn_oof_prediction(ntrain_x,train_y,ntest_x)
np.save('knn_oof_train.npy',knn_oof_train)
np.save('knn_oof_test.npy',knn_oof_test)

nb_oof_train,nb_oof_test = get_naivebayes_oof_prediction(ntrain_x,train_y,ntest_x)
np.save('nb_oof_train.npy',nb_oof_train)
np.save('nb_oof_test.npy',nb_oof_test)

sgd_oof_train,sgd_oof_test = get_sgd_oof_prediction(SEED,ntrain_x,train_y,ntest_x)
np.save('sgd_oof_train.npy',sgd_oof_train)
np.save('sgd_oof_test.npy',sgd_oof_test)


np.save('knn_oof_train.npy',knn_oof_train)
np.save('knn_oof_test.npy',knn_oof_test)

knn_oof_train = np.load('knn_oof_train.npy')
knn_oof_test = np.load('knn_oof_test.npy')
sgd_oof_train = np.load('sgd_oof_train.npy')
sgd_oof_test = np.load('sgd_oof_test.npy')
nb_oof_train = np.load('nb_oof_train.npy')
nb_oof_test = np.load('nb_oof_test.npy')
knn_oof_train = np.load('knn_oof_train.npy')
knn_oof_test = np.load('knn_oof_test.npy')




    
#TODO Feature selecture using target permutation


imp_df = pd.DataFrame()
imp_df["feature"] = list(predictors)
imp_df["importance_gain"] = xgbm.feature_importance(importance_type='gain')
imp_df["importance_split"] = model.feature_importance(importance_type='split')

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
    
    submission = pd.DataFrame({'building_id':data[len(train):]['building_id'],'damage_grade':predictions})
    for i in range(5):
        submission.loc[submission.damage_grade==i,'damage_grade'] = 'Grade '+str(i+1)
    
    return submission

#prediction from model
def get_predictionknn(predict):
    submission = pd.DataFrame({'building_id':data[len(train):]['building_id'],'damage_grade':predict})
    for i in range(5):
        submission.loc[submission.damage_grade==i,'damage_grade'] = 'Grade '+str(i+1)
    
    return submission


submission = get_predictionknn(list(model.predict(Pool(test_x,cat_features=cat_index))))
submission = get_predictioncv(cv_predictions,10)
submission.to_csv('catboost_base.csv',index=None)

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

