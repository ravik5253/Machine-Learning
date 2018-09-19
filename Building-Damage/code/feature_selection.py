# -*- coding: utf-8 -*-
from sklearn.datasets import load_boston
from sklearn.linear_model import (LogisticRegression, Ridge, 
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_val_score

X = train_x
Y = train_y  
# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

colnames = X.columns

rlasso = RandomizedLasso(alpha=0.04,random_state=512)
rlasso.fit(X, Y)
ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)
print('finished')


lr = LogisticRegression(multi_class='ovr',random_state=512)
lr.fit(X,Y)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(X,Y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)

#lr = LogisticRegression(normalize=True,random_state=512)
#lr.fit(X,Y)
ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

lr.coef_
#stability selection
from sklearn.linear_model import RandomizedLogisticRegression

clf = RandomizedLogisticRegression()
clf.fit(X,Y)
zero_feat = []
nonzero_feat = []
# type(clf.coef_)
for i in range(colnames):
    coef = clf.scores_[i]
    if coef == 0:
        zero_feat.append(X.columns[i])
    else:
        nonzero_feat.append((coef, X.columns[i]))
        
print ('Features that have coeffcient of 0 are: ', zero_feat)
print ('Features that have non-zero coefficients are:')
print (sorted(nonzero_feat, reverse = True))


# Using Ridge 
ridge = Ridge(alpha = 7,random_state=512)
ridge.fit(X,Y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

# Using Lasso
lasso = Lasso(alpha=.05,random_state=512)
lasso.fit(X, Y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)


rf = RandomForestClassifier(n_jobs=-1, n_estimators=50, verbose=3,random_state=512)
rf.fit(X,Y)
ranks["RF"] = ranking(rf.feature_importances_, colnames)


scores = []
clf = RandomForestClassifier(n_jobs=-1, n_estimators=50, verbose=3,random_state=512)
score_normal = np.mean(cross_val_score(clf, X, Y, cv = 10))

# X_shuffled.meanfreq
for i in range(colnames):
    X_shuffled = X.copy()
    scores_shuffle = []
    for j in range(3):
        np.random.seed(j*3)
        np.random.shuffle(X_shuffled[X.columns[i]])
        score = np.mean(cross_val_score(clf, X_shuffled, Y, cv = 10))
        scores_shuffle.append(score)
        
    scores.append((score_normal - np.mean(scores_shuffle), X.columns[i]))
    



r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
 
methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")
 
print("\t%s" % "\t".join(methods))
for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str, 
                         [ranks[method][name] for method in methods]))))
    
    
    
 # Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)

sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar", 
               size=14, aspect=1.9, palette='coolwarm')   