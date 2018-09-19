# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency,boxcox,skew
from scipy.stats import stats
from sklearn.preprocessing import LabelEncoder ,StandardScaler,OneHotEncoder
building_Ownership_Use = pd.read_csv('./Dataset/Building_Ownership_Use.csv')
building_Structure = pd.read_csv('./Dataset/Building_Structure.csv')
sample_submission = pd.read_csv('./Dataset/sample_submission.csv')
train = pd.read_csv('./Dataset/train.csv')
test = pd.read_csv('./Dataset/test.csv')


#building_Ownership_Use_head = building_Ownership_Use.head(10)
#building_Structure_head = building_Structure.head(10)
# merging tables
lb = LabelEncoder()
train['damage_grade'].value_counts()
train['damage_grade'] = lb.fit_transform(train['damage_grade'])
train_test = train.append(test)
train_test = train_test.merge(building_Structure,left_on=['building_id','district_id','vdcmun_id'],right_on=['building_id','district_id','vdcmun_id'],how='inner')
train_test = train_test.merge(building_Ownership_Use,left_on=['building_id','district_id','vdcmun_id','ward_id'],right_on=['building_id','district_id','vdcmun_id','ward_id'],how='inner')
train_test_head = train_test.head(25)


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

def plot_twocategorical_features(data,col1,col2,title='',xlabel_angle=0,figsize = (6,20)):
    plot_data = []
    col1_d = data[col1].value_counts()
    for i in col1_d.index:
        plot_data.append(data[data[col1]==i][col2].value_counts())
    
    fig , ax_arr = plt.subplots(col1_d.shape[0],1,figsize = figsize)
    
    for i in range(col1_d.shape[0]):
        sns.barplot(x = plot_data[i].index,y = plot_data[i].values,ax = ax_arr[i])
        ax_arr[i].set_title('Distribution of '+ str(col1_d.index[i]) )
        for tick in ax_arr[i].get_xticklabels():
            if (xlabel_angle > 0):
                tick.set_rotation(xlabel_angle)

#    fig.suptitle('Distribution of '+col)
    plt.show()

def count_missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percentage = (data.isnull().sum()*100/data.isnull().count()).sort_values(ascending=False)
    return pd.concat([total,percentage],keys = ['count','percentage'],axis=1)

def plot_distribution_num_feature(data,columns,size=(20,40),bins=50,title=''):
    l = len(columns)
    fig,ax_arr = plt.subplots(int(l/4+1),4,figsize=size)
    j=0
    fig.subplots_adjust(bottom = .001,top = 1 )
    for column in columns:
        sns.distplot(data[column].dropna(),ax=ax_arr[int(j/4)][j%4])
#        ax_arr[int(j/4)][j%4].set_title(column)
        j += 1
    plt.show()    

def plot_heatmap(data,num_feature):
    fig,ax = plt.subplots(figsize=(30,30))
    sns.heatmap(data[num_feature].dropna().corr(),annot=True,ax=ax)
    plt.savefig('./histogram.png')

# not use as this removes too many indexs
def remove_outliers(data,num_feat):
#    num_feat = get_num_feature(data)
    for feat in num_feat :
        q1 = data[feat].quantile(.25)
        q3 = data[feat].quantile(.75)
        iqr = q3-q1
        lower_bound = q1-iqr*1.5
        upper_bound = q3+iqr*1.5
        ind = data[ (data[feat] < lower_bound) | (data[feat] > upper_bound)].index
#        data.loc[ind,col] = data[col].median() 
        print(feat,' ',len(ind))
        
def label_encode(data):
    columns = get_cat_feature(data)
    lb = LabelEncoder()
    for col in columns:
        data[col] = lb.fit_transform(data[col])


def OHE_encode(df):
    columns = get_cat_feature(data)
    ndf = data[columns]
    for x in columns:
        ndf[x] = ndf[x].astype('category')
    
    ndf = pd.get_dummies(ndf,prefix=columns,columns=columns)
    ndfcolumns = ndf.columns.tolist()
    
#    print(columns,ndf.head(5))
    for x in ndfcolumns:
        df[x] = ndf[x].astype('int8')
    df.drop(labels=columns,axis=1,inplace=True)    



def get_cat_feature(data):
    return [x for x in data.columns if data[x].dtype == 'object' and x not in ['building_id', 'damage_grade']]

def get_num_feature(data):
    return [x for x in data.columns if data[x].dtype != 'object' and x not in ['building_id', 'damage_grade']]

def break_num_features(data):
    all_num_feature = get_num_feature(data)
    binary_feature = [x for x in all_num_feature if data[x].nunique()==2]
    discrete_feature = [x for x in all_num_feature if data[x].nunique()<32 and 
                        x not in binary_feature]
    continues_feature = [x for x in all_num_feature if x not in binary_feature and 
                   x not in discrete_feature ]
    return binary_feature,discrete_feature,continues_feature

def get_categories_stats(data,col):
    print(data[data.damage_grade.notnull()][col].nunique())
    print(data[col].unique())
    print(data[col].value_counts())
    
#Chi-squared test for independence for each pair of categorical variables    
def chi2_test(data,col1,col2):
    cross = pd.crosstab(data[col1],data[col2])
    chi2, p, dof, expected = chi2_contingency(cross)
    return p

#ANOVA null hypothesis that the mean value of feature/col is the same among all 
#five grade classes.
#Null Hypothesis : Means of all the groups are same
#Alternate Hypothesis : Mean of at least one group is different    
def anova_test(data,col):
    grade = []
    for i in range(5):
        grade.append(data[data.damage_grade==i][col])
    
    st,p = stats.f_oneway(grade[0],grade[1],grade[2],grade[3],grade[4])
    
    return p
  
  
# features that are skewed and applies the boxcox transformation to all of them
# if negative data is there  Back Transformation  
def find_remove_skew(data,cols):
    skew_feat = data[cols].apply(lambda x : skew(x))
    skew_feat =skew_feat[skew_feat>0.3].index
    print(skew_feat)
    for feat in skew_feat:
        if(data[feat].min() < 0 ):
            data[feat],_ = boxcox(data[feat]+1 - data[feat].min())
        else:
            data[feat],_ = boxcox(data[feat]+1)
   
# count_floors_post_eq must be less than count_floors_pre_eq but there is samples
# in test and train where it is not true 
# height_ft_post_eq must be less than height_ft_pre_eq but there is samples
# in test and train where it is not true 
# to do handle above case extract feature like difference in ht and countoffloors
# pre and post earthquake
# ht of floors to calculate correct values of floor count and vice versa as 
# intersection is there for one sample

# we has_repair_started value it is not true for all values ,hence we can say 
# that data contain bug

#TODO
def correcting_floor_cnt(data):
    # sample for which floorcount and height is different for pre and post
    ind = train_test[(train_test.count_floors_post_eq > train_test.count_floors_pre_eq) 
                    & (train_test.height_ft_post_eq > train_test.height_ft_pre_eq)][[x for x in num_feature if x in discrete_feature or x in continues_feature]]

    # sample for which floorcount is different for pre and post
    ind = train_test[(train_test.count_floors_post_eq > train_test.count_floors_pre_eq)][[x for x in num_feature if x in discrete_feature or x in continues_feature]]
    
    # sample for which height is different for pre and post
    ind = train_test[(train_test.height_ft_post_eq > train_test.height_ft_pre_eq)][continues_feature]
    
    return ind    


# remove columns with no standard deviation    
def remove_columns_with_no_variation(dataset):
    columns_to_remove = [col for col in dataset.columns if dataset[col].std()==0]
    return columns_to_remove

    
def extracting_useful_features(data):
    # high cardianlity column vdcmun_id,ward_id ,district_id converted to count feature
    res = data[['vdcmun_id','building_id']].groupby(['vdcmun_id'])[['building_id']].count().reset_index().rename(index=str,columns={'building_id':'vdcmun_count'})
    data = data.merge(res,on='vdcmun_id',how='left')
    res = data[['ward_id','building_id']].groupby(['ward_id'])[['building_id']].count().reset_index().rename(index=str,columns={'building_id':'ward_count'})
    data = data.merge(res,on='ward_id',how='left')
    res = data[['district_id','building_id']].groupby(['district_id'])[['building_id']].count().reset_index().rename(index=str,columns={'building_id':'district_count'})
    data = data.merge(res,on='district_id',how='left')
    data['vdcmun_count'] = data['vdcmun_count'].astype('uint16')
    data['ward_count'] = data['ward_count'].astype('uint16')
    data['district_count'] = data['district_count'].astype('uint16')
    
    # volume of building 
    data['building_volume'] = data.apply(lambda r : r.plinth_area_sq_ft*r.height_ft_pre_eq,axis=1)
    

    #    The majority of buildings with severe damage were those with 6–15 storeys
    data['isTall'] = 0
    ind = data[ (data.count_floors_pre_eq > 6) | (data.height_ft_pre_eq > 36)].index
    data.loc[ind,'isTall'] = 1
    data['isTall'] = data['isTall'].astype('int8')
    
    data['count_floors_diff'] = data.apply(lambda r : r.count_floors_pre_eq - r.count_floors_post_eq ,axis=1)
    data['height_ft_diff'] = data.apply(lambda r : r.height_ft_pre_eq - r.height_ft_post_eq ,axis=1)
# division by zero
#    data['floors_ht_pre'] = data.apply(lambda r : (r.height_ft_pre_eq/r.count_floors_pre_eq) ,axis=1)
#    data['floors_ht_post'] = data.apply(lambda r : (r.height_ft_post_eq/r.count_floors_post_eq) ,axis=1)
    # 1 condition_post_eq 
    # Damaged-Rubble clear,'Damaged-Rubble unclear','Damaged-Rubble Clear-New building built'
    # 'Covered by landslide' are all belong to same damage_grade Grade5 can be replaced by single value
    # labelencode the categories
    condition_post_eq_new = ['Damaged-Rubble clear','Damaged-Rubble unclear',
                             'Damaged-Rubble Clear-New building built','Covered by landslide']
    ind = data[data.condition_post_eq.isin(condition_post_eq_new)].index
    data.loc[ind,'condition_post_eq'] = 'Damaged-Rubble'
    
    #count families rare converting to categorical feature
    
    data.loc[data.count_families==0.0,'family_category'] = 0
    data.loc[data.count_families==1.0,'family_category'] = 1
    data.loc[data.count_families>=2.0,'family_category'] = 2
    data['family_category'] = data['family_category'].astype('int8')
    data['isSmallFamily'] = data['family_category'].map({1:1,0:1,2:0}).astype('int8')

    # plan_configuration has lot of rare category    
    data.loc[((data.plan_configuration!='Rectangular')&(data.plan_configuration!='Square')),'plan_configuration'] = 'rare'   
    
    data.loc[data.plan_configuration!='rare','isRectangular'] = 1
    data.loc[data.plan_configuration=='rare','isRectangular'] = 0
    data['isRectangular'] = data['isRectangular'].astype('int8')
    
    # land_surface_condition moderate: slope and steep slope are rare   
    data.loc[data.land_surface_condition!='Flat','land_surface_condition'] = 'slope'    
    data['land_surface_condition'] = data['land_surface_condition'].map({'Flat':1,'slope':0}).astype('int8')

    # foundation_type: except Mud mortar-Stone/Brick all are rare    
    data.loc[data.foundation_type!='Mud mortar-Stone/Brick','foundation_type'] = 'other'    
    data['foundation_type'] = data['foundation_type'].map({'Mud mortar-Stone/Brick':1,'other':0}).astype('int8')
    
    # ground_floor_type: except mud all are rare
    data.loc[data.ground_floor_type!='Mud','ground_floor_type'] = 'other' 
    data['ground_floor_type'] = data['ground_floor_type'].map({'Mud':1,'other':0}).astype('int8')
    
    # legal_ownership_status: except private all are rare
    data.loc[data.legal_ownership_status!='Private','legal_ownership_status'] = 'not_private'     
    data['legal_ownership_status'] = data['legal_ownership_status'].map({'Private':1,'not_private':0}).astype('int8')
         
    # position
    data.loc[data.position!='Not attached','position']='Attached'
    data['position']=data['position'].map({'Attached':0,'Not attached':1}).astype('int8')

    
    
    # age_building dividing into old(>30yrs) new(<10) middle(10-30)
    ind = data[data.age_building<=10].index
    data.loc[ind,'building_type'] = 0
    ind = data[ (data.age_building>10) & (data.age_building<30)].index
    data.loc[ind,'building_type'] = 1
    ind = data[(data.age_building>=30)].index
    data.loc[ind,'building_type'] = 2
    data['building_type'] = data['building_type'].astype('int8')
    data['isOld'] = data['building_type'].map({1:0,0:0,2:1}).astype('int8')
    data['isOld'] = data['isOld'].astype('int8')
    
    # Sparse features analysis     
    num_feature = get_num_feature(data)
    col_su = [c for c in num_feature if c.startswith('has_secondary_use_')]
    col_gr = [c for c in num_feature if c.startswith('has_geotechnical_risk_')]
    col_ss = [c for c in num_feature if c.startswith('has_superstructure_')]    
    
    # col with has_secondary_use_ prefix
    # this col is onehotencoded values with 118099 house has_secondary_use
    # houses which has_secondary_use_ more than  1 ,
    # correcting that rows by changing has_secondary_use_other to 0
    ind = data[(data[col_su].sum(axis=1)>1) & (data.has_secondary_use==1)& (data.has_secondary_use_other==1)].index
    data.loc[ind,'has_secondary_use_other'] = 0
    data['secondary_use'] = data[col_su].idxmax(axis=1)
    data['secondary_use_count'] = data[col_su].sum(axis=1)
    data['secondary_use_count'] = data['secondary_use_count'].astype('int8')
    # index of row has_secondary_use 0
    ind = data[data.has_secondary_use==0].index
    data.loc[ind,'secondary_use'] = 'no_secondary_use'
    
    ind = data[(data.secondary_use!='has_secondary_use_hotel')&(data.secondary_use!='has_secondary_use_agriculture')].index
    data.loc[ind,'secondary_use'] = 'has_secondary_use_other'
    
    data['geotechnical_risk_count'] = data[col_gr].sum(axis=1)
    data['geotechnical_risk_count'] = data['geotechnical_risk_count'].astype('int8')

    # col with has_geotechnical_risk_ prefix
    # col has_geotechnical_risk for 130958 house
    # for 54427 houses has more than one geotechnical_risk
    # this is not one hot encoded we can find sum as new feature
    # found mean as not useful
    data['geotechnical_risk_count'] = data[col_gr].sum(axis=1)
#    data['geotechnical_risk_mean'] = data[col_gr].mean(axis=1)
    data['geotechnical_risk_count'] = data['geotechnical_risk_count'].astype('int8')
#    data['geotechnical_risk_mean'] = data['geotechnical_risk_mean'].astype('int8')

    # col with has_superstructure prefix
    # for 1052936 houses has more than one has_superstructure
    # this is not one hot encoded we can find sum as new feature
    # found mean as not useful
    data['has_superstructure_count'] = data[col_ss].sum(axis=1)
#    data['has_superstructure_mean'] = data[col_ss].mean(axis=1)
    data['has_superstructure_count'] = data['has_superstructure_count'].astype('int8')
#    data['has_superstructure_mean'] = data['has_superstructure_mean'].astype('int8')    
#    data.drop(col_gr,axis=1,inplace=True)
#    data.drop(col_ss,axis=1,inplace=True)
#    data.drop(remove_features,axis=1,inplace=True)
    return data    
    

def handle_missing_values(data):    
    # missing values building_Ownership_Use
    # count_families , has_secondary_use has missing values 
    # replace with mode of respective missing attributes
    
    #plot_categorical_feature(building_Ownership_Use,'has_secondary_use')
    #plot_twocategorical_features(building_Ownership_Use,'has_secondary_use','legal_ownership_status')
    
    data.loc[data.count_families.isnull(),'count_families'] = 1
    data.loc[data.has_secondary_use.isnull(),'has_secondary_use'] = 0.0

    # missing values train , test
    # has_repair_started has missing values 
    # Not damaged has 0.0 as mode and Damaged-Repaired and used has 1.0 as mode  
    
    ind = data[ (data.condition_post_eq == 'Not damaged') & (data.has_repair_started.isnull())].index
    data.loc[ind,'has_repair_started'] = 0.0
    ind = data[(data.condition_post_eq == 'Damaged-Repaired and used' )& (data.has_repair_started.isnull())].index
    data.loc[ind,'has_repair_started'] = 1.0

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


data = train_test   
handle_missing_values(data)
data  = extracting_useful_features(data)
data_head = data.head(30)
data.to_csv('./Dataset/useful_data2.csv',index=None)


num_feature = get_num_feature(data)
cat_feature = get_cat_feature(data)
# break numerical feature into discrete,binary,continues
binary_feature,discrete_feature,continues_feature = break_num_features(data)
# continues_feature 
#find_remove_skew(data,continues_feature)
remove_outliers(data,continues_feature)
# let's  check if the transformation created infinite values
for var in continues_feature:
    if np.isinf(data[var]).sum()>1:
        print(var)

# Bivariate analysis
# continues vs continues 
# plotting correlataion matrix and pairplot
sns.pairplot(data,
             x_vars=['age_building','height_ft_pre_eq','height_ft_post_eq'],
             y_vars=['age_building','height_ft_pre_eq','height_ft_post_eq'],
             hue='damage_grade',
             diag_kind='kde')
corr = data[['age_building','height_ft_diff','height_ft_pre_eq','height_ft_post_eq','damage_grade']].corr()
print(corr)

# category vs category
# We can see that except plan_configuration each of the all categorical variables has a 
# statistically significant relationship with damage_grade, 
# as they all have p-values that are orders of magnitude below 0.05
# dependency bteween independent var and target
for feat in cat_feature:
    print(feat," ",chi2_test(data,feat,'damage_grade'))

# dependency bteween independent var
for feat1 in cat_feature:
    for feat2 in cat_feature:
        print(feat1,"  and  ",feat2,"  = ",chi2_test(data,feat1,feat2))

# binary_feature
# except has_secondary_use_use_police all have significant relationship
for feat in binary_feature:
    print(feat," ",chi2_test(data,feat,'damage_grade'))

# dependency bteween independent var    
for feat1 in binary_feature:
    for feat2 in binary_feature:
        p = chi2_test(data,feat1,feat2)
        if(p>0.5):
            print(feat1,"  and  ",feat2,"  = ",p)    

# continues vs category
#We can do this using T-Tests and ANOVA (depending on how many levels there are 
#in the categorical variables). This will tell us if there is a statistically 
#significant difference between the mean values of the continuous variable for 
#each level in the categorical variable.
#One of the assumptions of the t-test is normally distributed data
# all have p < 0.5 hence we reject null hypothesis that mean are same for
# damage_grade category
for feat in continues_feature:
    p = anova_test(data,feat)
    if(p<0.5):
        print(feat,"  and  damage_grade ",p)    


#categorical feature encoding try other also
remove_correlated_feature(data)

print(cat_feature)



get_categories_stats(data,'legal_ownership_status')



plot_categorical_feature(data[data.condition_post_eq == 'Not damaged'],'has_repair_started',xlabel_angle=30)
plot_categorical_feature(train_test,'count_families')
plot_numerical_feature(train_test,)

print(train_test.has_repair_started.isnull().sum()) 

print(data.condition_post_eq.value_counts())  
print(data.count_families.value_counts())  
print(data.vdcmun_id.value_counts())  



data_head=data.head(15)
uf_head = uf.head(15) 

plot_numerical_feature(uf,'has_superstructure_count')

uf['avg'] = uf.groupby(['district_id','vdcmun_id'])['count_floors_pre_eq'].transform(lambda x: (x.mode())).astype('int8')


data['avg_geo_risk_d_v_w'] = data.groupby(['district_id','vdcmun_id','ward_id'])['geotechnical_risk_count'].transform(lambda x: (x.mean())).astype('int8')
data['avg_geo_risk_d'] = data.groupby(['district_id'])['geotechnical_risk_count'].transform(lambda x: (x.mean())).astype('int8')
data['std_geo_risk_d'] = data.groupby(['district_id'])['geotechnical_risk_count'].transform(lambda x: (x.std())).astype('int8')


ind = uf.groupby(['district_id'])['count_floors_pre_eq'].agg(lambda x:x.mean())

ind = uf.groupby(['district_id'])['count_floors_post_eq'].agg(lambda x:x.mean())

ind = data.groupby(['district_id','vdcmun_id'])['district_id'].agg(lambda x:x.count())











ind = data[data.ward_id]

print(building_Ownership_Use[building_Ownership_Use.has_secondary_use.isnull()]['legal_ownership_status'].value_counts())

plot_twocategorical_features(train_test[train_test.has_repair_started.isnull()],'condition_post_eq','has_repair_started')
plot_numerical_feature(train_test,'')
plot_numerical_feature_bylabel(train,'has_repair_started')    


# data cleaning and transformation


# cat feature analysis

['area_assesed', 'building_id', 'damage_grade', 'land_surface_condition',
 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type',
 'position', 'plan_configuration', 'condition_post_eq', 'legal_ownership_status']

get_categories_stats(train_test,'area_assesed')
plot_categorical_feature(train_test,'position')    
plot_categorical_feature_bylabel(train_test[train_test.damage_grade.notnull()],
                                            'area_assesed')


# position,ground_floor_type,other_floor_type,roof_type,foundation_type,
# land_surface_condition,area_assesed,plan_configuration
# position = ['Not attached' 'Attached-1 side' 'Attached-2 side' 'Attached-3 side']
# labelencode
# foundation_type biased
#chi2_contingency(train_test,'ground_floor_type','other_floor_type')



['district_id', 'vdcmun_id', 'ward_id', 'count_floors_pre_eq', 'count_floors_post_eq',
 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq', 'height_ft_post_eq',
 'count_families']

#remove_outliers(train_test,num_feature)
plot_numerical_feature(train_test,'count_floors_post_eq')
plot_numerical_feature_bylabel(train_test,'age_building')

print(train_test.count_families.value_counts())

sns.regplot(data = train_test[train_test.notnull()],x='count_families',y='count_floors_pre_eq')

# for discrete feature calculate outiliers
#  I will call outliers, those values that are present in less than 1% of the houses.
#count_floors_pre_eq ,count_floors_post_eq,count_families,district_id


    



for var in continues_feature:
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    fig = sns.boxplot(y=data[var])
    fig.set_title('')
    fig.set_ylabel(var)
    
    plt.show()




for feat in cat_feature:
    print(data[feat].value_counts()/data[feat].count())



print(train_test[train_test.damage_grade.isnull()]['district_id'].nunique())

plot_heatmap(train_test,num_feature)


not_useful_feature = ['vdcmun_id', 'ward_id']


#Damaged-Not used                           249464
#Damaged-Repaired and used                  211113
#Damaged-Used in risk                       187474
#Damaged-Rubble unclear                     145732
#Damaged-Rubble clear                       132508
#Not damaged                                 71587
#Damaged-Rubble Clear-New building built     54587
#Covered by landslide                          471

train_test.condition_post_eq.value_counts()
plot_categorical_feature(train_test,'condition_post_eq',xlabel_angle=30)
ind = train_test[train_test.condition_post_eq==cond][[x for x in num_feature if x in discrete_feature or x in continues_feature]]

# 'Not damaged' 264 house
# Damaged-Not used  11 house
# Damaged-Repaired and used 267
# Damaged-Used in risk 80

sns.regplot(data = train_test,x = 'count_floors_pre_eq',y = 'height_ft_pre_eq')

cond = 'Not damaged'

mv = train_test.groupby(['count_floors_pre_eq'])['height_ft_pre_eq'].agg(lambda x:x.max())
ind = train_test[(train_test.count_floors_post_eq != train_test.count_floors_pre_eq) 
                & (train_test.height_ft_post_eq == train_test.height_ft_pre_eq)]

# houses for which count floors is diff and ht is same
ind = train_test[(train_test.count_floors_post_eq != train_test.count_floors_pre_eq) 
                & (train_test.height_ft_post_eq == train_test.height_ft_pre_eq)]['condition_post_eq']

# houses for which count floors is same and ht is diff
ind = train_test[(train_test.count_floors_post_eq == train_test.count_floors_pre_eq) 
                & (train_test.height_ft_post_eq != train_test.height_ft_pre_eq)][[x for x in num_feature if x in discrete_feature or x in continues_feature]]

def correct_height(row):
    avg_floor_ht = [10,14,18,24,30,54,63,72,80]
    ind = int(row.count_floors_post_eq-1)
    if( abs(avg_floor_ht[ind]-row.height_ft_pre_eq) <= abs(avg_floor_ht[ind]-row.height_ft_post_eq)):
        return row.height_ft_pre_eq
    else:
        return row.height_ft_post_eq
        
def correct_floors(row):
    avg_floor_ht = [0,10,14,18,24,30,54,63,72,80]
    ind1 = int(row.count_floors_post_eq)
    ind2 = int(row.count_floors_pre_eq)
    if( abs(avg_floor_ht[ind1]-row.height_ft_pre_eq) <= abs(avg_floor_ht[ind2]-row.height_ft_pre_eq)):
        return row.count_floors_post_eq
    else:
        return row.count_floors_pre_eq
        
        
ind.loc[ind.index,'height_ft_post_eq'] = ind.apply(lambda x:correct_height(x),axis=1)
ind.loc[ind.index,'height_ft_pre_eq'] = ind.apply(lambda x:correct_height(x),axis=1)


ind = train_test[(train_test.count_floors_post_eq < train_test.count_floors_pre_eq) 
                & (train_test.height_ft_post_eq > train_test.height_ft_pre_eq)][[x for x in num_feature if x in discrete_feature or x in continues_feature]]

ind.loc[ind.index,'count_floors_post_eq'] = ind.apply(lambda x:correct_floors(x),axis=1)
ind.loc[ind.index,'count_floors_pre_eq'] = ind.apply(lambda x:correct_floors(x),axis=1)

train_test.count_floors_post_eq.value_counts()

# 'Not damaged' 433 house
# Damaged-Not used  391 house
# Damaged-Repaired and used 1359
# # Damaged-Used in risk 759

# sample for which floorcount is different for pre and post
ind = train_test[(train_test.count_floors_post_eq > train_test.count_floors_pre_eq)&(train_test.height_ft_post_eq < train_test.height_ft_pre_eq)][[x for x in num_feature if x in discrete_feature or x in continues_feature]]

ind = train_test[(train_test.count_floors_post_eq > train_test.count_floors_pre_eq)][[x for x in num_feature if x in discrete_feature or x in continues_feature]]

# 'Not damaged' 491 house
# Damaged-Not used  858 house
# Damaged-Repaired and used 1018
# # Damaged-Used in risk 742

# sample for which height is different for pre and post
ind = train_test[(train_test.condition_post_eq==cond)&(train_test.height_ft_post_eq > train_test.height_ft_pre_eq)][continues_feature]


train_test.condition_post_eq.value_counts()



# RFECV selected features

['district_id',
 'has_geotechnical_risk',
 'has_geotechnical_risk_fault_crack',
 'has_geotechnical_risk_land_settlement',
 'has_geotechnical_risk_landslide',
 'has_repair_started',
 'vdcmun_id',
 'ward_id',
 'count_floors_pre_eq',
 'count_floors_post_eq',
 'age_building',
 'plinth_area_sq_ft',
 'height_ft_pre_eq',
 'height_ft_post_eq',
 'land_surface_condition',
 'foundation_type',
 'ground_floor_type',
 'has_superstructure_adobe_mud',
 'has_superstructure_mud_mortar_stone',
 'has_superstructure_stone_flag',
 'has_superstructure_cement_mortar_stone',
 'has_superstructure_mud_mortar_brick',
 'has_superstructure_cement_mortar_brick',
 'has_superstructure_timber',
 'has_superstructure_bamboo',
 'has_superstructure_rc_non_engineered',
 'has_superstructure_rc_engineered',
 'has_superstructure_other',
 'count_families',
 'has_secondary_use',
 'has_secondary_use_agriculture',
 'vdcmun_count',
 'ward_count',
 'district_count',
 'building_volume',
 'count_floors_diff',
 'height_ft_diff',
 'family_category',
 'building_type',
 'isOld',
 'geotechnical_risk_count',
 'has_superstructure_count',
 'area_assesed_Both',
 'area_assesed_Building removed',
 'area_assesed_Exterior',
 'area_assesed_Not able to inspect',
 'roof_type_Bamboo/Timber-Heavy roof',
 'roof_type_Bamboo/Timber-Light roof',
 'roof_type_RCC/RB/RBC',
 'other_floor_type_Not applicable',
 'other_floor_type_RCC/RB/RBC',
 'other_floor_type_TImber/Bamboo-Mud',
 'other_floor_type_Timber-Planck',
 'secondary_use_has_secondary_use_other',
 'condition_post_eq_Damaged-Not used',
 'condition_post_eq_Damaged-Repaired and used',
 'condition_post_eq_Damaged-Rubble',
 'condition_post_eq_Damaged-Used in risk',
 'condition_post_eq_Not damaged'



#1

LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.55,
        learning_rate=0.2, max_bin=400, max_depth=8, min_child_samples=100,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
        n_jobs=-1, nthread=-1, num_leaves=79, objective='multiclass',
        random_state=None, reg_alpha=0.01, reg_lambda=0.0, silent=True,
        subsample=0.9, subsample_for_bin=200000, subsample_freq=1,
        verbose=0)
train score - 0.7738861695394184
test score - 0.759040412197568

#2

LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.55,
        learning_rate=0.2, max_bin=400, max_depth=8, min_child_samples=100,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
        n_jobs=-1, nthread=-1, num_leaves=125, objective='multiclass',
        random_state=None, reg_alpha=0.01, reg_lambda=0.0, silent=True,
        subsample=0.9, subsample_for_bin=200000, subsample_freq=1,
        verbose=0)

train score - 0.7787008863659529
test score - 0.7602573798872887

#3 with feature selection
 
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.55,
        learning_rate=0.2, max_bin=400, max_depth=8, min_child_samples=100,
        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
        n_jobs=-1, nthread=-1, num_leaves=55, objective='multiclass',
        random_state=None, reg_alpha=0.01, reg_lambda=0.0, silent=True,
        subsample=0.9, subsample_for_bin=200000, subsample_freq=1,
        verbose=0)

train score - 0.7705867867676561
test score - 0.7580332385368846

#4 with learning rate 0.24
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.55,
        learning_rate=0.24, max_bin=800, max_depth=8,
        min_child_samples=100, min_child_weight=0.001, min_split_gain=0.0,
        n_estimators=100, n_jobs=-1, nthread=-1, num_leaves=55,
        objective='multiclass', random_state=None, reg_alpha=0.01,
        reg_lambda=0.0, silent=True, subsample=0.9,
        subsample_for_bin=200000, subsample_freq=1, verbose=0)

train score - 0.7768599288585549
test score - 0.7610657089484099