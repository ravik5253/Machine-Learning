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
sample_submission = pd.read_csv('./Dataset/sample_submission.csv')
train = pd.read_csv('./Dataset/train.csv')
test = pd.read_csv('./Dataset/test.csv')
data = pd.read_csv('./Dataset/useful_data2.csv')
data_head = data.head(25)

def label_encode(data,en=1):
    columns = ['area_assesed', 'roof_type','other_floor_type','secondary_use', 
               'plan_configuration','condition_post_eq']
    if en ==1:
        lb = LabelEncoder()
        for col in columns:
            data[col] = lb.fit_transform(data[col])
    else:
        ndf = data[columns]
        for x in columns:
            ndf[x] = ndf[x].astype('category')
        
        ndf = pd.get_dummies(ndf,prefix=columns,columns=columns)
        ndfcolumns = ndf.columns.tolist()
        
        for x in ndfcolumns:
            data[x] = ndf[x].astype('int8')
        data.drop(labels=columns,axis=1,inplace=True)    

def find_remove_skew(data,cols):
    skew_feat = data[cols].apply(lambda x : skew(x))
    print(skew_feat)
    skew_feat =skew_feat[skew_feat>0.3].index
    for feat in skew_feat:
        if(data[feat].min() < 0 ):
            data[feat],_ = boxcox(data[feat]+1 - data[feat].min())
        else:
            data[feat],_ = boxcox(data[feat]+1)

def break_num_features(data,num_feature):
    binary_feature = [x for x in num_feature if data[x].nunique()==2]
    continues_feature = ['plinth_area_sq_ft','building_volume','height_ft_diff','height_ft_pre_eq','height_ft_post_eq']
    discrete_feature = [x for x in num_feature if x not in binary_feature and 
                   x not in continues_feature ]
    return binary_feature,discrete_feature,continues_feature


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
    
    

cat_feature = ['area_assesed', 'land_surface_condition', 'foundation_type', 'roof_type',
               'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration',
               'condition_post_eq', 'legal_ownership_status','district_id','secondary_use',
               'building_type','family_category','ward_id','vdcmun_id']

num_feature = [x for x in data.columns if x not in cat_feature and x not in ['building_id', 'damage_grade']]
# break numerical feature into discrete,binary,continues
binary_feature,discrete_feature,continues_feature = break_num_features(data,num_feature)

#remove skewness from continues_feature feature
find_remove_skew(data,continues_feature)
#remove skewness from descrete feature
find_remove_skew(data,discrete_feature)

# [with descrete skewness]   train's multi_logloss: 0.398705 valid's multi_logloss: 0.485917
# train's multi_logloss: 0.407632 valid's multi_logloss: 0.485661

label_encode(data,en=2)


# features to remove
#remove_correlated_feature(data)
remove_features =  ['secondary_use_count']
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

knnpca = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
knnpca.fit(xtrain_s,ytrain)
print("Train score {} ".format(knnpca.score(xtrain_s, ytrain)))
print("Test score {} ".format(knnpca.score(xtest_s, ytest)))

#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(xtrain_s,ytrain)
print("Train score {} ".format(qda.score(xtrain_s, ytrain)))
print("Test score {} ".format(qda.score(xtest_s, ytest)))

# Factoranalysis
from sklearn.decomposition import FactorAnalysis
factor = FactorAnalysis(random_state=512)
factor.fit(train_x[continues_feature])
fxtrain = factor.transform(train_x[continues_feature])
plt.scatter(fxtrain[:,0], fxtrain[:,1], c=train_y,  cmap='prism', alpha=0.4)

dd = pd.DataFrame(factor.components_,columns=continues_feature)
#TODO
#Non-Negative Matrix Factorization (NMF) and TSNE

tsne = TSNE(n_components=2, random_state=0)
xtrain_s = tsne.fit_transform(xtrain_s, train_y)

plt.scatter(xtrain_s[:,0], xtrain_s[:,1], c=ytrain,  cmap='prism', alpha=0.4)


# LDA
#The objective of LDA is to preserve the class separation information 
#whilst still reducing the dimensions of the dataset
scaler = StandardScaler()
scaler.fit(xtrain[predictors])
xtrain_s = scaler.transform(xtrain[predictors])
xtest_s = scaler.transform(xtest[predictors])

lda = LinearDiscriminantAnalysis()
lda.fit(xtrain_s,ytrain)

exp_var_cum=np.cumsum(lda.explained_variance_ratio_)
plt.step(range(exp_var_cum.size), exp_var_cum)

xtrain_s = lda.transform(xtrain_s)
xtest_s = lda.transform(xtest_s)
plt.scatter(xtrain_s[:,0],xtrain_s[:,1],c=ytrain,cmap='prism',alpha=0.4)

knnlda = KNeighborsClassifier(n_neighbors=55,n_jobs=-1)
knnlda.fit(xtrain_s,ytrain)


print("Train score {} ".format(knnlda.score(xtrain_s, ytrain)))
print("Test score {} ".format(knnlda.score(xtest_s, ytest)))
test_x_s = scaler.transform(test_x)
#test_x_s = pca.transform(test_x_s)
test_x_s = lda.transform(test_x_s)
predic = knnlda.predict(test_x_s)

#pca lda knn
#Train score 0.7147215714828415 
#Test score 0.7015425039373818 

#lda knn
#Train score 0.7170879764467519 
#Test score 0.70371894612712 


sel_features = ranking[ranking.Rank==1]['Features'].values
predictors = list(sel_features)
predictors = [ x for x in predictors if x not in remove_features]
num_feature = [x for x in data.columns if x not in cat_feature and x not in ['building_id', 'damage_grade']]
# break numerical feature into discrete,binary,continues
binary_feature,discrete_feature,continues_feature = break_num_features(data,num_feature)
#predictors = [col for col in predictors if col not in col_su]
categorical_feature=[col for col in predictors if col in cat_feature or col in binary_feature]
# lightgbm

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
        'learning_rate': 0.01,
        'num_leaves': 55,  # 2^max_depth - 1
        'max_depth': 8,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 800,  # Number of bucketed bin for feature values
        'subsample': .9,  # Subsample ratio of the training instance.
#        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.55,  # Subsample ratio of columns when constructing each tree.
#        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'reg_alpha': 0.01,
        'seed':512
        }

#multi_logloss: 0.4827483737298364 , 4823440735271875,4822726250409628
#0.7886267141183974 ,0.7892270149175348
# with ohe 48879647987418107, 48505386646501514
#          4883163637005777 , 4858900024133575                 
model = fit_lgbm_model(xtrain, xtest,
                       ytrain, ytest,
                       params,predictors,
                       categorical_feature=categorical_feature,
                       objective='multiclass', 
                       num_boost_round=2000,
                       early_stopping_rounds=500,
                       metrics = 'multi_logloss',
#                       feval = custom_f1_weighted,
                       verbose_eval=True
                       )




true = ytest+1
predict = model.predict(xtest[predictors])
predictions = []
for x in predict:
    predictions.append(np.argmax(x)+1)
score = f1_score(true,predictions,average='weighted')




# gridsearch
params = {
    'learning_rate': [0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24],
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': [55],  # 2^max_depth - 1
    'max_depth': [8],  # -1 means no limit
#    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': [400,800],  # Number of bucketed bin for feature values
    'subsample': [0.9],  # Subsample ratio of the training instance.
#    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': [0.55],  # Subsample ratio of columns when constructing each tree.
#    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
#    'scale_pos_weight':99 # because training data is extremely unbalanced [0.01,0.05,0.1]
    'reg_alpha': [0.01],
    'min_child_samples':[100]
}
lgbm = lgb.LGBMClassifier(boosting_type = "gbdt",objective ='multiclass',nthread = -1,verbose = 0)
gridsearch = RandomizedSearchCV(lgbm,params,cv=5,n_jobs=-1,scoring='f1_weighted')
gridsearch.fit(xtrain[predictors].values,ytrain)             

#train score - 0.7705867867676561
#test score - 0.7580332385368846
print("train score - " + str(gridsearch.score(xtrain[predictors].values,ytrain)))
print("test score - " + str(gridsearch.score(xtest[predictors].values, ytest)))
gridsearch.best_estimator_



#RFECV using lightgbm

dtrain_x = train_x
scaler = StandardScaler()
dxtrain_s = scaler.fit_transform(dtrain_x[continues_feature])
pca = PCA(random_state=512, whiten=True)
dxtrain_s = pca.fit_transform(dxtrain_s)
# new features as 90% varaince of descrete and continues feature represented by first 4 components
dtrain_x['pca1']=dxtrain_s[:,0]
dtrain_x['pca2']=dxtrain_s[:,1]
dtrain_x['pca3']=dxtrain_s[:,2]
dtrain_x['pca4']=dxtrain_s[:,3]


from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print('\n Time taken: %i minutes and %s seconds.' % (tmin, round(tsec, 2)))


rlgbm = lgb.LGBMClassifier(random_state=512,subsample=0.9,colsample_bytree=0.55,reg_alpha=0.1,min_child_samples=100,num_leaves=125,max_bin=400,max_depth=8,learning_rate=0.2,boosting_type = "gbdt",objective ='multiclass',nthread = -1,verbose = 0)
rflgbm = RFECV(estimator=rlgbm,cv=StratifiedKFold(n_splits=5,random_state=512).split(dtrain_x,train_y),step=3,scoring='f1_weighted',n_jobs=-1,verbose=2)

start_time = timer(None)
rflgbm.fit(dtrain_x,train_y)
timer(start_time)



print('\n Optimal number of features: %d' % rflgbm.n_features_)
sel_features = [f for f, s in zip(dtrain_x.columns, rflgbm.support_) if s]
print('\n The selected features are {}:'.format(sel_features))

plt.figure(figsize=(12, 9))
plt.xlabel('Number of features tested x 3')
plt.ylabel('Cross-validation score (AUC)')
plt.plot(range(1, len(rflgbm.grid_scores_) + 1), rflgbm.grid_scores_)
plt.savefig('Porto-RFECV-01.png', dpi=150)
plt.show()


ranking = pd.DataFrame({'Features': dtrain_x.columns})
ranking['Rank'] = np.asarray(rflgbm.ranking_)
ranking.sort_values('Rank', inplace=True)
ranking.to_csv('RFECV-ranking-01.csv', index=False)

ranking = pd.read_csv('RFECV-ranking-01.csv')





def get_prediction(y,predict):
    predictions = []
    for x in predict:
        predictions.append(np.argmax(x))
    
    return f1_score(y,predictions,average='weighted')




# feature selection target permutation




def get_feature_importances(dtrain_x ,dtrain_y ,predictors,categorical_feats, shuffle, seed=None):
    # Go over fold and keep track of CV score (train and valid) and feature importances
    
    # Shuffle target if required
    y = dtrain_y.copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = dtrain_y.copy().sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    dtrain = lgb.Dataset(dtrain_x[predictors], y, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'multiclass',
        'boosting_type': 'gbdt',
        'subsample': 0.9,
        'colsample_bytree': 0.56,
        'max_bin': 200,  # Number of bucketed bin for feature values
        'num_leaves': 67,
        'max_depth': 8,
        'reg_alpha': 0.01,
        'seed': seed,
        'n_jobs': 4,
        'num_class':5
    }
    
    # Fit the model
    clf = lgb.train(params=lgb_params,verbose_eval=True, train_set=dtrain, num_boost_round=200, categorical_feature=categorical_feats)
    
    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(predictors)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = get_prediction(y,clf.predict(dtrain_x[predictors]))
    
    return imp_df


actual_imp_df = get_feature_importances(xtrain,pd.Series(ytrain),predictors,categorical_feature, shuffle=True)

null_imp_df = pd.DataFrame()
nb_runs = 5
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(xtrain,pd.Series(ytrain),predictors,categorical_feature, shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)


correlation_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
    gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
    split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    correlation_scores.append((_f, split_score, gain_score))

corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])










def score_feature_selection(df=None, train_features=None, cat_feats=None, target=None):
    # Fit LightGBM 
    dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
    lgb_params = {
        'objective': 'multiclass',
        'boosting_type': 'gbdt',
        'learning_rate': .02,
        'subsample': 0.9,
        'colsample_bytree': 0.56,
        'num_leaves': 67,
        'max_depth': 8,
        'seed': 13,
        'n_jobs': 4,
        'num_class':5,
        'metric': 'multi_logloss'
    }
    
    # Fit the model
    hist = lgb.cv(
        params=lgb_params, 
        train_set=dtrain, 
        num_boost_round=2000,
        categorical_feature=cat_feats,
        nfold=5,
        stratified=True,
        shuffle=True,
        early_stopping_rounds=50,
        verbose_eval=0,
        seed=17
    )
    # Return the last mean / std values 
    return hist['multi_logloss-mean'][-1], hist['multi_logloss-stdv'][-1]

# features = [f for f in data.columns if f not in ['SK_ID_CURR', 'TARGET']]
# score_feature_selection(df=data[features], train_features=features, target=data['TARGET'])

for threshold in [0, 10, 20, 30 , 40, 50 ,60 , 70, 80 , 90, 95, 99]:
    split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
    split_cat_feats = [_f for _f, _score, _ in correlation_scores if (_score >= threshold) & (_f in categorical_feature)]
    gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]
    gain_cat_feats = [_f for _f, _, _score in correlation_scores if (_score >= threshold) & (_f in categorical_feature)]
                                                                                             
    print('Results for threshold %3d' % threshold)
    split_results = score_feature_selection(df=xtrain, train_features=split_feats, cat_feats=split_cat_feats, target=ytrain)
    print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
    gain_results = score_feature_selection(df=xtrain, train_features=gain_feats, cat_feats=gain_cat_feats, target=ytrain)
    print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))








not_important_feature = imp_df[imp_df.importance_split<90]['feature'].values




imp_df = pd.DataFrame()
imp_df["feature"] = list(predictors)
imp_df["importance_gain"] = model.feature_importance(importance_type='gain')
imp_df["importance_split"] = model.feature_importance(importance_type='split')
imp_df = imp_df.sort_values(by='importance_split',ascending=False)
imp_df[['feature','importance_split']].plot(kind='barh', x='feature', y='importance_split', legend=False, figsize=(10, 20))
    
    

# feature selection
lgbm = lgb.LGBMClassifier(max_depth=7,num_leaves=35,reg_alpha=0.1,boosting_type = "gbdt",
                          max_bin= 200,min_child_samples= 300,subsample=0.8,colsample_bytree=0.56,objective ='multiclass',nthread = -1,verbose = 0)
selector = RFECV(lgbm,cv=5,scoring='f1_weighted',n_jobs=-1)
selector.fit(xtrain[predictors].values,ytrain)
print("The number of selected features is: {}".format(selector.n_features_))
features_kept = xtrain.columns.values[selector.support_] 
features_kept = list(features_kept)


def fit_lgbm_modelnew(dtrain_x, dtest_x, dtrain_y, dtest_y,params,predictors,
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
    
    lgbTrain = lgb.Dataset(dtrain_x,label = dtrain_y,
                           feature_name=predictors,categorical_feature=categorical_feature)
    
    lgbValid = lgb.Dataset(dtest_x,label = dtest_y,
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



def kfold(train_x,train_y,test_x,predictors,categorical_feature):
    kf = KFold(n_splits=10,random_state=101)
    cv_predictions = []
    cv_scores=[]

    for fold,(train_in,test_in) in enumerate(kf.split(train_x,train_y)):
        print(train_in,test_in)
        xtrain,xvalid,ytrain,yvalid = train_x[train_in,:],train_x[test_in,:],train_y[train_in],train_y[test_in]
        params = {
        'learning_rate': 0.2,
        'num_leaves': 35,  # 2^max_depth - 1
        'max_depth': 7,  # -1 means no limit
        'max_bin': 200,  # Number of bucketed bin for feature values
        'subsample': .9,  # Subsample ratio of the training instance.
        'colsample_bytree': 0.56,  # Subsample ratio of columns when constructing each tree.
        'reg_alpha': 0.01,
        'min_child_samples':300
        }

        model = fit_lgbm_modelnew(xtrain, xvalid,
                       ytrain, yvalid,
                       params,predictors,
                       categorical_feature=categorical_feature,
                       objective='multiclass', 
                       num_boost_round=2000,
                       early_stopping_rounds=30,
                       metrics = 'multi_logloss',
                       verbose_eval=True
                       )
        
        true = yvalid+1
        predict = model.predict(xvalid)
        predictions = []
        for x in predict:
            predictions.append(np.argmax(x)+1)
        score = f1_score(true,predictions,average='weighted')

#        score = model.best_score_['valid_0']['multi_logloss']
        cv_scores.append(score)
#        print(model.)
        cv_predictions.append(model.predict(test_x))
    return cv_scores,cv_predictions



#def kfold(train_x,train_y,test_x):
#    kf = KFold(n_splits=5,random_state=1000)
#    cv_predictions = []
#    cv_scores = []
#
#    for fold,(train_in,test_in) in enumerate(kf.split(train_x,train_y)):
#        print(train_in,test_in)
#        xtrain,xvalid,ytrain,yvalid = train_x[train_in,:],train_x[test_in,:],train_y[train_in],train_y[test_in]
#        model = lgb.LGBMClassifier(boosting_type = "gbdt",objective ='multiclass',nthread = -1,
#                                   verbose = 0,n_estimators=1000,learning_rate=0.2,num_leaves=59,
#                                   max_depth=13,reg_alpha=0.01,subsample=0.9,colsample_bytree=0.6)
#        model.fit(xtrain,ytrain,eval_set=[(xvalid,yvalid)],eval_metric='multi_logloss',verbose=1,early_stopping_rounds=30)
#        
#        score = model.best_score_['valid_0']['multi_logloss']
#        cv_scores.append(score)
##        print(model.)
#        cv_predictions.append(model.predict(test_x))
#    return cv_scores ,cv_predictions
        
        
cv_scores,cv_predictions = kfold(train_x[predictors].values,train_y,test_x[predictors].values,predictors,categorical_feature)

    
#TODO Feature selecture using target permutation


imp_df = pd.DataFrame()
imp_df["feature"] = list(predictors)
imp_df["importance_gain"] = model.feature_importance(importance_type='gain')
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


submission = get_predictionknn(predic)
submission = get_predictioncv(cv_predictions,10)
submission.to_csv('cv_ldaknnmodel1.csv',index=None)

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





def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


trn, sub = target_encode(pd.Series(train_x["district_id"]), 
                         pd.Series(test_x["district_id"]), 
                         target=pd.Series(train_y), 
                         min_samples_leaf=100,
                         smoothing=10,
                         noise_level=0.01)
trn.head(10)
target=pd.Series(train_y)
print(target.name)
