# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency,boxcox,skew
from scipy.stats import stats
from sklearn.preprocessing import LabelEncoder ,StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,cross_validate,GridSearchCV,RandomizedSearchCV
from sklearn.decomposition import PCA,NMF
from sklearn.metrics import accuracy_score,log_loss,f1_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import xgboost as xgb
import lightgbm as lgb
sample_submission = pd.read_csv('./Dataset/sample_submission.csv')
train = pd.read_csv('./Dataset/train.csv')
test = pd.read_csv('./Dataset/test.csv')
data = pd.read_csv('./Dataset/useful_data1.csv')
data_head = data.head(25)

def label_encode(data,en=1):
    columns = ['area_assesed', 'roof_type','other_floor_type','secondary_use', 
               'plan_configuration','condition_post_eq']
    data.loc[data.position!='Not attached','position']='Attached'
    data['position']=data['position'].map({'Attached':0,'Not attached':1}).astype('int8')
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
    skew_feat =skew_feat[skew_feat>0.3].index
    for feat in skew_feat:
        if(data[feat].min() < 0 ):
            data[feat],_ = boxcox(data[feat]+1 - data[feat].min())
        else:
            data[feat],_ = boxcox(data[feat]+1)

def break_num_features(data,num_feature):
    binary_feature = [x for x in num_feature if data[x].nunique()==2]
    discrete_feature = [x for x in num_feature if data[x].nunique()<20 and 
                        x not in binary_feature]
    continues_feature = [x for x in num_feature if x not in binary_feature and 
                   x not in discrete_feature ]
    return binary_feature,discrete_feature,continues_feature

cat_feature = ['area_assesed', 'land_surface_condition', 'foundation_type', 'roof_type',
               'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration',
               'condition_post_eq', 'legal_ownership_status','district_id',
               'building_type','family_category','ward_id','vdcmun_id']

num_feature = [x for x in data.columns if x not in cat_feature and x not in ['building_id', 'damage_grade']]
# break numerical feature into discrete,binary,continues
binary_feature,discrete_feature,continues_feature = break_num_features(data,num_feature)

find_remove_skew(data,continues_feature)

label_encode(data,en=2)

train_x = data[:len(train)]
train_y = train_x.damage_grade.values
train_x.drop(['building_id','damage_grade'],inplace=True,axis=1)
test_x = data[len(train):]
test_x.drop(['building_id','damage_grade'],inplace=True,axis=1)
train_test_head = train_x.head(12)

xtrain , xtest , ytrain ,ytest = train_test_split(train_x,train_y,test_size=0.2,random_state=101,stratify=train_y)


# scaling 
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain_s = scaler.transform(xtrain)
xtest_s = scaler.transform(xtest)



predictors = xtrain.columns.tolist()
categorical_feature=[col for col in predictors if col in cat_feature or col in binary_feature]

# lightgbm
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

# Base model xgboost

params = {
        'learning_rate':0.02,
        'max_depth': 6,
        'subsample':0.8,
        'n_estimator':500,
        'colsample_bytree':[0.56],
        'gamma': 0.5,
#        'eval_metric':'mlogloss',
        'num_class':5
        }

dtrain = xgb.DMatrix(xtrain,ytrain)
dvalid = xgb.DMatrix(xtest,ytest)
watchlist = [(dtrain,'train'),(dvalid,'valid')]
model = xgb.train(params,dtrain,1500,watchlist,early_stopping_rounds=30,verbose_eval=10)

xgbc = xgb.XGBClassifier(n_estimators=500,eval_metric='mlogloss',booster='gbtree',objective='multi:softmax',random_state=101)

RandomizedSearchCV(xgbc,params,scoring='f1_weighted',cv=5,random_state=101)


true = ytest+1
predict = model.predict(xtest)
predictions = []
for x in predict:
    predictions.append(np.argmax(x)+1)
score = f1_score(true,predictions,average='weighted')


imp_df = pd.DataFrame()
imp_df["feature"] = list(predictors)
imp_df["importance_gain"] = model.feature_importance(importance_type='gain')
imp_df["importance_split"] = model.feature_importance(importance_type='split')
imp_df = imp_df.sort_values(by='importance_split',ascending=False)
imp_df[['feature','importance_split']].plot(kind='barh', x='feature', y='importance_split', legend=False, figsize=(10, 20))
    
    
# gridsearch
params = {
    'learning_rate': [0.2],
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': [35],  # 2^max_depth - 1
    'max_depth': [6,7],  # -1 means no limit
#    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': [25,50,100,128,150,200,250],  # Number of bucketed bin for feature values
    'subsample': [0.8,0.85,0.9],  # Subsample ratio of the training instance.
#    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': [0.4,0.45,0.5,0.55,0.6],  # Subsample ratio of columns when constructing each tree.
#    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
#    'scale_pos_weight':99 # because training data is extremely unbalanced [0.01,0.05,0.1]
    'reg_alpha': [0.01],
    'min_child_samples':[100,200,300,400,500,600]
}
lgbm = lgb.LGBMClassifier(boosting_type = "gbdt",objective ='multiclass',nthread = -1,verbose = 0)
gridsearch = RandomizedSearchCV(lgbm,params,cv=5,n_jobs=-1,scoring='f1_weighted')
gridsearch.fit(xtrain[predictors].values,ytrain)

print("train score - " + str(gridsearch.score(xtrain[predictors].values,ytrain)))
print("test score - " + str(gridsearch.score(xtest[predictors].values, ytest)))
gridsearch.best_estimator_


def kfold(train_x,train_y,test_x,predictors,categorical_feature):
    skf = StratifiedKFold(n_splits=10,random_state=101)
    cv_predictions = []
    cv_scores=[]
    d_test = xgb.DMatrix(test_x)
    params = {
        'learning_rate':0.02,
        'max_depth': 7,
        'subsample':0.8,
        'n_estimator':500,
        'colsample_bytree':0.56,
        'gamma': 0.5,
#        'eval_metric':'mlogloss',
        'num_class':5
    }
    
    for fold,(train_in,test_in) in enumerate(skf.split(train_x,train_y)):
        xtrain,ytrain = train_x[train_in],train_y[train_in]
        xtest,ytest = train_x[test_in],train_y[test_in]
        dtrain = xgb.DMatrix(xtrain,ytrain)
        dvalid = xgb.DMatrix(xtest,ytest)
        watchlist = [(dtrain,'train'),(dvalid,'valid')]
        model = xgb.train(params,dtrain,1500,watchlist,early_stopping_rounds=30,verbose_eval=10)
        score = model.predict(d_test, ntree_limit=model.best_ntree_limit)
        cv_predictions.append(score)
    return cv_predictions

    
cv_predictions = kfold(train_x[predictors].values,train_y,test_x[predictors].values,predictors,categorical_feature)


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
def get_prediction(predict):
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


submission = get_predictioncv(cv_predictions,10)
submission.to_csv('cv_ldaknnmodel1.csv',index=None)
sub = pd.read_csv('cv_ldaknnmodel1.csv')
sub.damage_grade.value_counts()









