# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import stats
from keras.activations import relu,softmax,sigmoid,tanh
from keras.regularizers import l1_l2,l1,l2
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense,Flatten,Dropout,Activation
from keras.optimizers import SGD,Adadelta,Adagrad,Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import chi2_contingency,boxcox,skew
from sklearn.preprocessing import LabelEncoder ,StandardScaler,OneHotEncoder
from sklearn.model_selection import cross_val_score,train_test_split,KFold,StratifiedKFold,cross_validate,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,log_loss,f1_score,roc_auc_score,confusion_matrix
from sklearn.decomposition import PCA,NMF
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

#TODO removing collinear feature 
remove_correlated_feature(data)
col_su = [c for c in num_feature if c.startswith('has_secondary_use_')]
col_feature = ['has_superstructure_mean','geotechnical_risk_mean']
col_feature = col_feature + col_su
xtrain.drop(col_feature,axis=1,inplace=True)
xtest.drop(col_feature,axis=1,inplace=True)


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


def precision(y_true, y_pred): 
    """Precision metric. Only computes a batch-wise average of precision.  
-    Computes the precision, a metric for multi-label classification of 
-    how many selected items are relevant. 
-    """ 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    precision = true_positives / (predicted_positives + K.epsilon()) 
    return precision 

def recall(y_true, y_pred): 
    """Recall metric. 
-    Only computes a batch-wise average of recall. 
-    Computes the recall, a metric for multi-label classification of 
-    how many relevant items are selected. 
-    """ 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall 

def fbeta_score(y_true, y_pred, beta=1): 

    """Computes the F score.  
-    The F score is the weighted harmonic mean of precision and recall. 
-    Here it is only computed as a batch-wise average, not globally. 
-    This is useful for multi-label classification, where input samples can be 
-    classified as sets of labels. By only using accuracy (precision) a model 
-    would achieve a perfect score by simply assigning every class to every 
-    input. In order to avoid this, a metric should penalize incorrect class 
-    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0) 
-    computes this, as a weighted mean of the proportion of correct class 
-    assignments vs. the proportion of incorrect class assignments.  
-    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning 
-    correct classes becomes more important, and with beta > 1 the metric is 
-    instead weighted towards penalizing incorrect class assignments. 
-    """ 
    if beta < 0: 
        raise ValueError('The lowest choosable beta is zero (only precision).') 

    # If there are no true positives, fix the F score at 0 like sklearn. 
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0: 

        return 0 
    p = precision(y_true, y_pred) 
    r = recall(y_true, y_pred) 
    bb = beta ** 2 
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon()) 
    return fbeta_score 

def fmeasure(y_true, y_pred): 
    """Computes the f-measure, the harmonic mean of precision and recall. 
    Here it is only computed as a batch-wise average, not globally. 
    """ 
    print(y_pred)
    return fbeta_score(y_true, y_pred, beta=1) 


def base_model():
    model = Sequential()
    model.add(Dense(64,input_dim=79))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense(64))
#    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense(32))
#    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense(32))
#    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense(5,activation='softmax'))
    opt = Adam()
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=[fmeasure,recall,precision])
    return model



model = base_model()
scaler = StandardScaler()
nxtrain = scaler.fit_transform(xtrain)
nxtest = scaler.transform(xtest)
nytrain = np_utils.to_categorical(ytrain)
nytest = np_utils.to_categorical(ytest)


annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)    
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
seqmodel = model.fit(nxtrain,nytrain,epochs=200, batch_size=128,validation_data=(nxtest,nytest),callbacks=[early_stopping,annealer])


res = model.predict_classes(nxtest)
score = f1_score(ytest,res,average='weighted')


#gridseaarch neural network
def create_model(layers,activation):
    model = Sequential()
    for i,nodes in enumerate(layers):
        if(i==0):
            model.add(Dense(nodes,input_dim=xtrain.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            
    model.add(Dense(5,activation='softmax'))
    opt = Adam()
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=[fmeasure,recall,precision])        
    return model

model = KerasClassifier(build_fn=create_model,verbose=0)

layers = [[80],[80,40],[80,40,40],[80,40,20],[80,80,40,20],[80,80,40,40],[80,80,40,40,20],[80,80,40,40,20,20]]
activation = ['sigmoid','relu','tanh']
paramgrid = dict(layers=layers,activation=activation,batch_size=[128,256],epochs=[100,150,200])
grid = GridSearchCV(estimator=model,param_grid=paramgrid,cv=3,scoring='f1_weighted')
grid.fit(nxtrain,nytrain)

# neural network oof for ensemble
ntrain = train_x.shape[0]
ntest = test_x.shape[0]
SEED = 512
NFOLDS = 3
skf = StratifiedKFold(n_splits=NFOLDS,random_state=SEED)

def get_oof_neural_network_prediction(x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    
    for i,(train_ind,test_ind) in enumerate(skf.split(x_train,y_train)):
        print
        model = base_model()
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_train[train_ind])
        x_ts = scaler.transform(x_train[test_ind])
        y_tr = np_utils.to_categorical(y_train[train_ind])
        y_ts = np_utils.to_categorical(y_train[test_ind])
        x_test_s = scaler.transform(x_test)
        annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)    
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        model.fit(x_tr,y_tr,epochs=200, batch_size=128,validation_data=(x_ts,y_ts),callbacks=[early_stopping,annealer])
        oof_train[test_ind] = model.predict_classes(x_ts)
        oof_test_skf[i,:] = model.predict_classes(x_test_s)
        
    oof_test = stats.mode(oof_test_skf,axis=0)[0]
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)

ranking = pd.read_csv('RFECV-ranking-01.csv')
sel_features = ranking[ranking.Rank<=4]['Features'].values
predictors = list(sel_features)
predictors = [x for x in predictors if x not in ['pca3' ,'pca1', 'pca2', 'pca4']]
ntrain_x = train_x[predictors].values
ntest_x = test_x[predictors].values

ANN_FS_oof_train,ANN_FS4_oof_test = get_oof_neural_network_prediction(ntrain_x,train_y,ntest_x)

np.save('ANN_FS_oof_train.npy',ANN_FS_oof_train)
np.save('ANN_FS4_oof_test.npy',ANN_FS4_oof_test)

#cross validation

def kfold(train_x,train_y,test_x):
    stkf = StratifiedKFold(n_splits=5,random_state=101)
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    predictions = []
    cv_score = []
    models = []
    model = base_model()
    for fold,(train_ind,test_ind) in enumerate(stkf.split(train_x,train_y)):
        print('fold ================================= ',fold+1 )
        trainx,testx = train_x[train_ind],train_x[test_ind]
        trainy,testy = train_y[train_ind],train_y[test_ind]
        scaler = StandardScaler()
        scaler.fit(trainx)
        trainy = np_utils.to_categorical(trainy)
        testy = np_utils.to_categorical(testy)
        trainx = scaler.transform(trainx)
        testx = scaler.transform(testx)
        resultx = scaler.transform(test_x)
        seqmodel = model.fit(nxtrain,nytrain,epochs=40, batch_size=128,validation_data=(testx,testy),callbacks=[early_stopping,annealer])
        cv_score.append(model.predict_classes(testx))
        predictions.append(model.predict_classes(resultx))
        models.append(model)
    
    return predictions,cv_score



predictions,cv_score = kfold(train_x.values,train_y,test_x.values)


scores = []
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
batch_size=[32,64,128,256,512,1024]
for opt in batch_size:
    model = base_modelc()
    seqmodel = model.fit(nxtrain,nytrain,epochs=10, batch_size=opt,validation_data=(nxtest,nytest),callbacks=[early_stopping,annealer])
    scores.append(seqmodel.history)
    
#Adam batch_size 128  perform best
i=0
for history in scores:
    plt.figure(figsize=[8,6])
    plt.plot(history['fmeasure'],'r',linewidth=3.0)
    plt.plot(history['val_fmeasure'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs '+str(batch_size[i]),fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    i += 1

plt.figure(figsize=[8,6])
plt.plot(scores[0]['val_fmeasure'],'b',linewidth=3.0)
plt.plot(scores[1]['val_fmeasure'],'g',linewidth=3.0)
plt.plot(scores[2]['val_fmeasure'],'r',linewidth=3.0)
plt.plot(scores[3]['val_fmeasure'],'y',linewidth=3.0)
plt.plot(scores[4]['val_fmeasure'],'black',linewidth=3.0)
plt.plot(scores[5]['val_fmeasure'],'pink',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs '+str(batch_size[i]),fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
i += 1


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
def get_predictionNN(predict,cv):
    df = pd.DataFrame()
    for x in range(cv):
        df['col'+str(x)] = predict[x]
    
    predictions = df.mode(axis=1)
    
    submission = pd.DataFrame({'building_id':data[len(train):]['building_id'],'damage_grade':predictions})
    for i in range(5):
        submission.loc[submission.damage_grade==i,'damage_grade'] = 'Grade '+str(i+1)
    
    return submission


#prediction from model
def get_prediction(predict):
    counts = np.bincount(predict)
    predictions = []
    for x in predict:
        predictions.append(np.argmax(counts))
    
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


submission = get_predictionknn(res[0].values)
#submission = get_predictioncv(cv_predictions,10)
submission.to_csv('NNmodel.csv',index=None)



sub = pd.read_csv('cv_ldaknnmodel1.csv')
sub1 = pd.read_csv('./result/lightgbm_with_feature_extraction/cv_model.csv')
sub2 = pd.read_csv('./result/score76091/cv_model2.csv')
sub3 = pd.read_csv('NNmodel.csv')
df = pd.DataFrame()
lbl = LabelEncoder()

df['knn'] = lbl.fit_transform(sub.damage_grade.values)
df['lgbm'] = lbl.fit_transform(sub1.damage_grade.values)
df['lgbm_wm'] = lbl.fit_transform(sub2.damage_grade.values)
df['CNN_cv'] = lbl.fit_transform(sub3.damage_grade.values)



sns.heatmap(df.corr(),annot=True)

sub.damage_grade.value_counts()



#nn
df = pd.DataFrame()
for x in range(5):
    df['col'+str(x)] = predictions[x]

predicts = df.mode(axis=1)
df.to_csv('NNmodel_pred.csv',index=None)
predicts = pd.read_csv('NNmodel_pred.csv')
res = predicts.mode(axis=1)





