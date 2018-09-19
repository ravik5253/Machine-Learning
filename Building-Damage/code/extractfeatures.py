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
uf = pd.read_csv('./Dataset/useful_data0.csv')


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
# but it does not effect the score after removing the columns            
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



    
def extracting_useful_features0(data):
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
        
    data['count_floors_diff'] = data.apply(lambda r : r.count_floors_pre_eq - r.count_floors_post_eq ,axis=1)
    data['height_ft_diff'] = data.apply(lambda r : r.height_ft_pre_eq - r.height_ft_post_eq ,axis=1)
# division by zero
#    data['floors_ht_pre'] = data.apply(lambda r : (r.height_ft_pre_eq/r.count_floors_pre_eq) ,axis=1)
#    data['floors_ht_post'] = data.apply(lambda r : (r.height_ft_post_eq/r.count_floors_post_eq) ,axis=1)
    # 1 condition_post_eq 
    # Damaged-Rubble clear,'Damaged-Rubble unclear','Damaged-Rubble Clear-New building built'
    # 'Covered by landslide' are all belong to same damage_grade Grade5 can be replaced by single value
    # labelencode the categories
#    condition_post_eq_new = ['Damaged-Rubble clear','Damaged-Rubble unclear',
#                             'Damaged-Rubble Clear-New building built','Covered by landslide']
#    ind = data[data.condition_post_eq.isin(condition_post_eq_new)].index
#    data.loc[ind,'condition_post_eq'] = 'Damaged-Rubble'
    
    #count families rare converting to categorical feature
    
    data.loc[data.count_families==0.0,'family_category'] = 0
    data.loc[data.count_families==1.0,'family_category'] = 1
    data.loc[data.count_families>=2.0,'family_category'] = 2
    data['family_category'] = data['family_category'].astype('int8')

    # plan_configuration has lot of rare category    
    data.loc[((data.plan_configuration!='Rectangular')&(data.plan_configuration!='Square')),'plan_configuration'] = 'rare'   
    
    # age_building dividing into old(>30yrs) new(<10) middle(10-30)
    ind = data[data.age_building<=10].index
    data.loc[ind,'building_type'] = 0
    ind = data[ (data.age_building>10) & (data.age_building<30)].index
    data.loc[ind,'building_type'] = 1
    ind = data[(data.age_building>=30)].index
    data.loc[ind,'building_type'] = 2
    data['building_type'] = data['building_type'].astype('int8')
    
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
    data['secondary_use_count'] = data[col_su].sum(axis=1)
    data['secondary_use_count'] = data['secondary_use_count'].astype('int8')
    
#    data['geotechnical_risk_count'] = data[col_gr].sum(axis=1)
#    data['geotechnical_risk_count'] = data['geotechnical_risk_count'].astype('int8')
    
    # col with has_geotechnical_risk_ prefix
    # col has_geotechnical_risk for 130958 house
    # for 54427 houses has more than one geotechnical_risk
    # this is not one hot encoded we can find sum as new feature
    # found mean as not useful
    data['geotechnical_risk_count'] = data[col_gr].sum(axis=1)
#    data['geotechnical_risk_mean'] = data[col_gr].mean(axis=1)
    data['geotechnical_risk_count'] = data['geotechnical_risk_count'].astype('int8')
#    data['geotechnical_risk_mean'] = data['geotechnical_risk_mean'].astype('int8')
    data['avg_geo_risk_d_v'] = data.groupby(['district_id','vdcmun_id'])['geotechnical_risk_count'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_geo_risk_d_v_w'] = data.groupby(['district_id','vdcmun_id','ward_id'])['geotechnical_risk_count'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_geo_risk_d'] = data.groupby(['district_id'])['geotechnical_risk_count'].transform(lambda x: (x.mean())).astype('int8')
    data['std_geo_risk_d'] = data.groupby(['district_id'])['geotechnical_risk_count'].transform(lambda x: (x.std())).astype('int8')

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
    data['avg_superstructure_d_v'] = data.groupby(['district_id','vdcmun_id'])['has_superstructure_count'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_superstructure_d_v_w'] = data.groupby(['district_id','vdcmun_id','ward_id'])['has_superstructure_count'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_superstructure_d'] = data.groupby(['district_id'])['has_superstructure_count'].transform(lambda x: (x.mean())).astype('int8')
    data['std_superstructure_d'] = data.groupby(['district_id'])['has_superstructure_count'].transform(lambda x: (x.std())).astype('int8')

    return data

def extracting_useful_features1(data):
    # high cardianlity column vdcmun_id,ward_id ,district_id converted to count feature
    data['vdcmun_count'] = data[['vdcmun_id','building_id']].groupby('vdcmun_id')['building_id'].transform(lambda x: x.count()).astype('uint16')
    data['is_small_vdcmun'] = data.apply(lambda r: (1 if r.vdcmun_count<1000 else 0),axis=1).astype('uint16')
    data['ward_count'] = data[['ward_id','building_id']].groupby(['ward_id'])['building_id'].transform(lambda x: x.count()).astype('uint16')
    data['is_small_ward'] = data.apply(lambda r: (1 if r.ward_count<100 else 0),axis=1).astype('uint16')
    data['district_count'] = data[['district_id','building_id']].groupby(['district_id'])['building_id'].transform(lambda x: x.count()).astype('uint16')

    # volume of building and new features
    data['pre_building_volume'] = data.apply(lambda r : r.plinth_area_sq_ft*r.height_ft_pre_eq,axis=1)
    data['post_building_volume'] = data.apply(lambda r : r.plinth_area_sq_ft*r.height_ft_post_eq,axis=1)        
    data['count_floors_diff'] = data.apply(lambda r : r.count_floors_pre_eq - r.count_floors_post_eq ,axis=1)
    data['building_volume_diff'] = data.apply(lambda r : r.pre_building_volume - r.post_building_volume ,axis=1)
    data['height_ft_diff'] = data.apply(lambda r : r.height_ft_pre_eq - r.height_ft_post_eq ,axis=1)
    data['is_multi_floors'] = data.apply(lambda r : 1 if r.count_floors_pre_eq>0 else 0  ,axis=1)
    
    data['avg_height_ft_diff_d_v'] = data.groupby(['district_id','vdcmun_id'])['height_ft_diff'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_height_ft_diff_d_v_w'] = data.groupby(['district_id','vdcmun_id','ward_id'])['height_ft_diff'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_height_ft_diff_d'] = data.groupby(['district_id'])['height_ft_diff'].transform(lambda x: (x.mean())).astype('int8')

    data['avg_building_volume_diff_d_v'] = data.groupby(['district_id','vdcmun_id'])['building_volume_diff'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_building_volume_diff_d_v_w'] = data.groupby(['district_id','vdcmun_id','ward_id'])['building_volume_diff'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_building_volume_diff_d'] = data.groupby(['district_id'])['building_volume_diff'].transform(lambda x: (x.mean())).astype('int8')

    data['avg_count_floors_diff_d_v'] = data.groupby(['district_id','vdcmun_id'])['count_floors_diff'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_count_floors_diff_d_v_w'] = data.groupby(['district_id','vdcmun_id','ward_id'])['count_floors_diff'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_count_floors_diff_d'] = data.groupby(['district_id'])['count_floors_diff'].transform(lambda x: (x.mean())).astype('int8')

# division by zero
#    data['floors_ht_pre'] = data.apply(lambda r : (r.height_ft_pre_eq/r.count_floors_pre_eq) ,axis=1)
#    data['floors_ht_post'] = data.apply(lambda r : (r.height_ft_post_eq/r.count_floors_post_eq) ,axis=1)
    # 1 condition_post_eq 
    # Damaged-Rubble clear,'Damaged-Rubble unclear','Damaged-Rubble Clear-New building built'
    # 'Covered by landslide' are all belong to same damage_grade Grade5 can be replaced by single value
    # labelencode the categories
#    condition_post_eq_new = ['Damaged-Rubble clear','Damaged-Rubble unclear',
#                             'Damaged-Rubble Clear-New building built','Covered by landslide']
#    ind = data[data.condition_post_eq.isin(condition_post_eq_new)].index
#    data.loc[ind,'condition_post_eq'] = 'Damaged-Rubble'
    
    #count families rare converting to categorical feature
    
    data.loc[data.count_families==0.0,'is_home'] = 0
    data.loc[data.count_families!=0.0,'is_home'] = 1
#    data.loc[data.count_families>=2.0,'family_category'] = 2
    data['is_home'] = data['is_home'].astype('int8')

    # plan_configuration has lot of rare category    
#    data.loc[((data.plan_configuration!='Rectangular')&(data.plan_configuration!='Square')),'plan_configuration'] = 'rare'   
    
    # age_building dividing into old(>30yrs) new(<10) middle(10-30)
    ind = data[data.age_building<=10].index
    data.loc[ind,'is_new_building'] = 1
    ind = data[ (data.age_building>10)].index
    data.loc[ind,'is_new_building'] = 0
#    ind = data[(data.age_building>=30)].index
#    data.loc[ind,'building_type'] = 2
    data['is_new_building'] = data['is_new_building'].astype('int8')
    
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
#    data['secondary_use_count'] = data[col_su].sum(axis=1)
#    data['secondary_use_count'] = data['secondary_use_count'].astype('int8')
    
#    data['geotechnical_risk_count'] = data[col_gr].sum(axis=1)
#    data['geotechnical_risk_count'] = data['geotechnical_risk_count'].astype('int8')
    
    # col with has_geotechnical_risk_ prefix
    # col has_geotechnical_risk for 130958 house
    # for 54427 houses has more than one geotechnical_risk
    # this is not one hot encoded we can find sum as new feature
    # found mean as not useful
    data['geotechnical_risk_count'] = data[col_gr].sum(axis=1)
#    data['geotechnical_risk_mean'] = data[col_gr].mean(axis=1)
    data['geotechnical_risk_count'] = data['geotechnical_risk_count'].astype('int8')
#    data['geotechnical_risk_mean'] = data['geotechnical_risk_mean'].astype('int8')
    data['avg_geo_risk_d_v'] = data.groupby(['district_id','vdcmun_id'])['geotechnical_risk_count'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_geo_risk_d_v_w'] = data.groupby(['district_id','vdcmun_id','ward_id'])['geotechnical_risk_count'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_geo_risk_d'] = data.groupby(['district_id'])['geotechnical_risk_count'].transform(lambda x: (x.mean())).astype('int8')
    data['std_geo_risk_d'] = data.groupby(['district_id'])['geotechnical_risk_count'].transform(lambda x: (x.std())).astype('int8')

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
    data['avg_superstructure_d_v'] = data.groupby(['district_id','vdcmun_id'])['has_superstructure_count'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_superstructure_d_v_w'] = data.groupby(['district_id','vdcmun_id','ward_id'])['has_superstructure_count'].transform(lambda x: (x.mean())).astype('int8')
    data['avg_superstructure_d'] = data.groupby(['district_id'])['has_superstructure_count'].transform(lambda x: (x.mean())).astype('int8')
    data['std_superstructure_d'] = data.groupby(['district_id'])['has_superstructure_count'].transform(lambda x: (x.std())).astype('int8')

    return data


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
        
        
def correcting_ht_floors(data):    
    index = data[(data.count_floors_post_eq == data.count_floors_pre_eq) 
                & (data.height_ft_post_eq != data.height_ft_pre_eq)].index
    
    data.loc[index,'height_ft_post_eq'] = data.apply(lambda x:correct_height(x),axis=1)
    data.loc[index,'height_ft_pre_eq'] = data.apply(lambda x:correct_height(x),axis=1)
    
    
    index = data[(data.count_floors_post_eq != data.count_floors_pre_eq) 
                    & (data.height_ft_post_eq == data.height_ft_pre_eq)].index
    data.loc[index,'count_floors_post_eq'] = data.apply(lambda x:correct_floors(x),axis=1)
    data.loc[index,'count_floors_pre_eq'] = data.apply(lambda x:correct_floors(x),axis=1)

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


def label_encode(columns,data,en=1):

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


def get_categories_stats(data,col):
    print(data[data.damage_grade.notnull()][col].nunique())
    print(data[col].unique())
    print(data[col].value_counts())

for col in cat:
    get_categories_stats(data,col)

def _woe(s, tp, tn):
    """Weight of evidence

    woe_i = ln(P_i/TP) - ln(N_i/TN)

    :param s: pandas groupby obj
    :param tp: total positives in full series (target prior)
    :param tn: total negatives in full series
    """
    p = s.sum()
    nom = p / tp
    den = (s.count() - p) / tn
    return np.log(nom / den)


def _micci_barreca_encode(s, tp, min_samples_leaf=1, smoothing=1):
    """Micci Barreca encoding

    This transformation outputs something between supervised ratio and target
    prior, depending on smoothing level.

    :param s: pandas groupby obj
    :param tp: total positives in full series
    """
    smoothing = \
        1 / (1 + np.exp(-(s.count() - min_samples_leaf) / smoothing))
    return tp * (1 - smoothing) + s.mean() * smoothing


target_prior = df['D'].sum()
target_size = df['D'].count()
    aggregation_agenda = \
        {'_woe': lambda x: _woe(x, target_prior, target_size - target_prior),
         '_micci': lambda x: _micci_barreca_encode(x, target_prior,
                                                   min_samples_leaf=100,
                                                   smoothing=10),
         }
    col = 'PostalCode'
    transformed_df = \
        df.groupby([col], as_index=False).D\
            .agg(aggregation_agenda)\
            .rename(columns={agg_key: col+agg_key for
                             agg_key in aggregation_agenda.keys()})


res = pd.get_dummies(ytrain, prefix="pred").astype(int)
prior_0, prior_1, prior_2, prior_3, prior_4 = res[["pred_0.0", "pred_1.0", "pred_2.0","pred_3.0", "pred_4.0"]].mean()

res = data.ward_id.value_counts()
res = xtrain[['building_id','district_id','vdcmun_id','ward_id']]
res['vdcmun_count'] = res[['district_id','vdcmun_id','ward_id','building_id']].groupby(['district_id','vdcmun_id','ward_id'])['building_id'].transform(lambda x: x.count()).astype('uint16')
res['vdcmun_id_updated'] = res[['vdcmun_id','district_id','building_id','vdcmun_count']].groupby(['district_id'])['vdcmun_count'].transform(lambda x: x.mean())
res['is_small_vdcmun'] = res.apply(lambda r: (1 if r.vdcmun_count<1000 else 0),axis=1)

res = data.groupby('district_id')[''].agg({"size": "size", "mean": "mean"})
res["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
res[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob


data = train_test   
handle_missing_values(data)
data= correcting_ht_floors(data)
data  = extracting_useful_features1(data)
cat = get_cat_feature(data)
label_encode(cat,data,en=2)
cat = ['district_id','vdcmun_id','ward_id']
label_encode(cat,data,en=1)
data_head = res.head(30)
data.to_csv('./Dataset/useful_data_final.csv',index=None)


#
##1
#
#LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.55,
#        learning_rate=0.2, max_bin=400, max_depth=8, min_child_samples=100,
#        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
#        n_jobs=-1, nthread=-1, num_leaves=79, objective='multiclass',
#        random_state=None, reg_alpha=0.01, reg_lambda=0.0, silent=True,
#        subsample=0.9, subsample_for_bin=200000, subsample_freq=1,
#        verbose=0)
#train score - 0.7738861695394184
#test score - 0.759040412197568
#
##2
#
#LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.55,
#        learning_rate=0.2, max_bin=400, max_depth=8, min_child_samples=100,
#        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
#        n_jobs=-1, nthread=-1, num_leaves=125, objective='multiclass',
#        random_state=None, reg_alpha=0.01, reg_lambda=0.0, silent=True,
#        subsample=0.9, subsample_for_bin=200000, subsample_freq=1,
#        verbose=0)
#
#train score - 0.7787008863659529
#test score - 0.7602573798872887
#
##3 with feature selection
# 
#LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.55,
#        learning_rate=0.2, max_bin=400, max_depth=8, min_child_samples=100,
#        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
#        n_jobs=-1, nthread=-1, num_leaves=55, objective='multiclass',
#        random_state=None, reg_alpha=0.01, reg_lambda=0.0, silent=True,
#        subsample=0.9, subsample_for_bin=200000, subsample_freq=1,
#        verbose=0)
#
#train score - 0.7705867867676561
#test score - 0.7580332385368846
#
##4 with learning rate 0.24
#LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.55,
#        learning_rate=0.24, max_bin=800, max_depth=8,
#        min_child_samples=100, min_child_weight=0.001, min_split_gain=0.0,
#        n_estimators=100, n_jobs=-1, nthread=-1, num_leaves=55,
#        objective='multiclass', random_state=None, reg_alpha=0.01,
#        reg_lambda=0.0, silent=True, subsample=0.9,
#        subsample_for_bin=200000, subsample_freq=1, verbose=0)
#
#train score - 0.7768599288585549
#test score - 0.7610657089484099
