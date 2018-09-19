#Accounts: This table has information on customers/accounts. These are the 
#accounts of whom we are marketing the properties for sale.
#
#Opportunities: These include the historic deals for the accounts. Basically, 
#this gives a transaction summary of the deals that have happened between 
#the accounts and the properties
#
#Accounts to Properties: This table comprises information on properties that 
#have been already bought by the accounts
#
#Deal to Properties: This table comprises information on the deals that has 
#materialized on the properties


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency,boxcox,skew
from scipy.stats import stats
from sklearn.preprocessing import LabelEncoder ,StandardScaler,OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb 
from sklearn.neighbors import NearestNeighbors

properties = pd.read_csv('./Dataset/Property dataset/Properties.csv')
test = pd.read_csv('./Dataset/Test dataset/Test_Data.csv')
accounts_properties = pd.read_csv('./Dataset/training dataset/Accounts_properties.csv')
accounts = pd.read_csv('./Dataset/training dataset/Accounts.csv')
deals_to_Properties = pd.DataFrame(pd.read_csv('./Dataset/training dataset/Deals_to_Properties.csv',engine='python'))
opportunities = pd.read_csv('./Dataset/training dataset/Opportunities.csv')
sample = pd.read_csv('./Dataset/Test dataset/sample_submission.csv')


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
    
def count_missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percentage = (data.isnull().sum()*100/data.isnull().count()).sort_values(ascending=False)
    return pd.concat([total,percentage],keys = ['count','percentage'],axis=1)

from collections import Counter
def mode_function2(lst):
    counter = Counter(lst)
    mode,val = counter.most_common(1)[0]
    if pd.isnull(mode):
        count=0
        for x,y in counter.items():
            if (y>=count and pd.notnull(x)):
                mode=x
                count=y

    return mode

#return mode if mode is nan then return other value with same count 
def mode_function1(lst):
    counter = Counter(lst)
    mode,val = counter.most_common(1)[0]
    if pd.isnull(mode):
        for x,y in counter.items():
            if (y==val and pd.notnull(x)):
                mode=x
                break;
    
    return mode


def properties_missing_values(df):
    #handle missing values df
    #market 1 missing value replacing with mode 
    df.loc[df.market.isnull(),'market'] = 'Market-0059'
    
    #region__c
    # replacing with value of nearby city and county
    df.loc[df.region__c.isnull(),"region__c"] = df.groupby(['market'])['region__c'].transform(lambda x:mode_function1(x))
    df.loc[df.region__c.isnull(),'region__c'] = 'Southwest'
    
    #sale_status
    
    df.loc[df.sale_status.isnull(),"sale_status"] = df.groupby(['city','county','market'])['sale_status'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_status.isnull(),"sale_status"] = df.groupby(['county','market'])['sale_status'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_status.isnull(),"sale_status"] = df.groupby(['market'])['sale_status'].transform(lambda x:mode_function1(x))
    #property_type_1
    
    df.loc[df.property_type_1.isnull(),"property_type_1"] = df.groupby(['city','county','market'])['property_type_1'].transform(lambda x:mode_function1(x))
    df.loc[df.property_type_1.isnull(),"property_type_1"] = df.groupby(['county','market'])['property_type_1'].transform(lambda x:mode_function1(x))
    df.loc[df.property_type_1.isnull(),"property_type_1"] = df.groupby(['market'])['property_type_1'].transform(lambda x:mode_function1(x))
    
    #size_sf ,size_acres,price_per_unit
    #replacing with mean value of mearket,property_type_1
    df.loc[df.size_sf<50,'size_sf']=np.nan
    df.loc[df.size_sf.isnull(),"size_sf"] = df.groupby(['city','county','market','property_type_1'])['size_sf'].transform(lambda x:x.mean())
    df.loc[df.size_sf<50,'size_sf']=np.nan
    df.loc[df.size_sf.isnull(),"size_sf"] = df.groupby(['county','market','property_type_1'])['size_sf'].transform(lambda x:x.mean())
    df.loc[df.size_sf<50,'size_sf']=np.nan
    df.loc[df.size_sf.isnull(),"size_sf"] = df.groupby(['market','property_type_1'])['size_sf'].transform(lambda x:x.mean())
    df.loc[df.size_sf<50,'size_sf']=np.nan
    df.loc[df.size_sf.isnull(),"size_sf"] = df.groupby(['property_type_1'])['size_sf'].transform(lambda x:x.mean())
    
    #removing column size_acres lot of missing values deal later
    df.drop(labels=['size_acres'],inplace=True,axis=1)
    
    #year_built
    # replacing with mode value check with median
    df.loc[df.year_built<1800,'year_built']=np.nan
    df.loc[df.year_built.isnull(),"year_built"] = df.groupby(['city','county','market','property_type_1'])['year_built'].transform(lambda x:mode_function1(x))
    df.loc[df.year_built.isnull(),"year_built"] = df.groupby(['county','market','property_type_1'])['year_built'].transform(lambda x:mode_function1(x))
    df.loc[df.year_built.isnull(),"year_built"] = df.groupby(['market','property_type_1'])['year_built'].transform(lambda x:mode_function1(x))
    df.loc[df.year_built.isnull(),"year_built"] = df.groupby(['property_type_1'])['year_built'].transform(lambda x:mode_function1(x))
    
    #building_status
    df.loc[df.building_status.isnull(),"building_status"] = df.groupby(['year_built','property_type_1'])['building_status'].transform(lambda x:mode_function2(x))
    df.loc[df.building_status.isnull(),"building_status"] = df.groupby(['year_built'])['building_status'].transform(lambda x:mode_function2(x))
    df.loc[df.year_built==2021,'building_status']='Proposed'
    df.loc[df.building_status.isnull(),'building_status']=df.building_status.mode()[0]

    #occupancy_rate
    # replacing with median value
    df.loc[df.occupancy_rate.isnull(),"occupancy_rate"] = df.groupby(['city','county','market','year_built','property_type_1','building_status'])['occupancy_rate'].transform(lambda x:x.median())
    df.loc[df.occupancy_rate.isnull(),"occupancy_rate"] = df.groupby(['county','market','year_built','property_type_1','building_status'])['occupancy_rate'].transform(lambda x:x.median())
    df.loc[df.occupancy_rate.isnull(),"occupancy_rate"] = df.groupby(['market','year_built','property_type_1','building_status'])['occupancy_rate'].transform(lambda x:x.median())
    df.loc[df.occupancy_rate.isnull(),"occupancy_rate"] = df.groupby(['year_built','property_type_1','building_status'])['occupancy_rate'].transform(lambda x:x.median())
    df.loc[df.occupancy_rate.isnull(),"occupancy_rate"] = df.groupby(['year_built'])['occupancy_rate'].transform(lambda x:x.median())
    df.loc[df.occupancy_rate.isnull(),"occupancy_rate"] = df.groupby(['building_status'])['occupancy_rate'].transform(lambda x:x.median())
    df.loc[df.year_built==2021,'occupancy_rate']=0.0
    df.loc[df.building_status.isnull(),'building_status']=df.occupancy_rate.median()
    
    #num_floors
    df.loc[((df.num_floors>80 )& (df.num_floors==0 )),'num_floors']=np.nan
    df.loc[df.num_floors.isnull(),"num_floors"] = df.groupby(['city','county','market','property_type_1'])['num_floors'].transform(lambda x:mode_function1(x))
    df.loc[df.num_floors.isnull(),"num_floors"] = df.groupby(['county','market','property_type_1'])['num_floors'].transform(lambda x:mode_function1(x))
    df.loc[df.num_floors.isnull(),"num_floors"] = df.groupby(['market','property_type_1'])['num_floors'].transform(lambda x:mode_function1(x))
    df.loc[df.num_floors.isnull(),"num_floors"] = df.groupby(['property_type_1'])['num_floors'].transform(lambda x:mode_function1(x))
    
    #price_per_sq_ft
    df.loc[(df.price_per_sq_ft==0 ),'price_per_sq_ft']=np.nan
    df.loc[df.price_per_sq_ft.isnull(),"price_per_sq_ft"] = df.groupby(['city','county','market','property_type_1'])['price_per_sq_ft'].transform(lambda x:x.mean())
    df.loc[df.price_per_sq_ft.isnull(),"price_per_sq_ft"] = df.groupby(['county','market','property_type_1'])['price_per_sq_ft'].transform(lambda x:x.mean())
    df.loc[df.price_per_sq_ft.isnull(),"price_per_sq_ft"] = df.groupby(['market','property_type_1'])['price_per_sq_ft'].transform(lambda x:x.mean())
    df.loc[df.price_per_sq_ft.isnull(),"price_per_sq_ft"] = df.groupby(['property_type_1'])['price_per_sq_ft'].transform(lambda x:x.mean())
    
    #sale_date__c,sale_amount__c
    df.loc[df.sale_amount__c.isnull(),"sale_amount__c"] = df.apply(lambda x:x.price_per_sq_ft*x.size_sf,axis=1)
    
    df['sale_date__c'] = pd.to_datetime(df['sale_date__c']) 
    df['sale_month'] = df['sale_date__c'].dt.month
    df['sale_year'] = df['sale_date__c'].dt.year
    df['sale_day'] = df['sale_date__c'].dt.day
    
    df.loc[df.sale_year.isnull(),"sale_year"] = df.groupby(['market','property_type_1','year_built'])['sale_year'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_year.isnull(),"sale_year"] = df.groupby(['property_type_1','year_built'])['sale_year'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_year.isnull(),"sale_year"] = df.groupby(['year_built'])['sale_year'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_year.isnull(),"sale_year"] = df.groupby(['property_type_1'])['sale_year'].transform(lambda x:mode_function1(x))
    
    df.loc[df.sale_month.isnull(),"sale_month"] = df.groupby(['market','property_type_1','year_built'])['sale_month'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_month.isnull(),"sale_month"] = df.groupby(['property_type_1','year_built'])['sale_month'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_month.isnull(),"sale_month"] = df.groupby(['year_built'])['sale_month'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_month.isnull(),"sale_month"] = df.groupby(['property_type_1'])['sale_month'].transform(lambda x:mode_function1(x))
    
    df.loc[df.sale_day.isnull(),"sale_day"] = df.groupby(['market','property_type_1','year_built'])['sale_day'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_day.isnull(),"sale_day"] = df.groupby(['property_type_1','year_built'])['sale_day'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_day.isnull(),"sale_day"] = df.groupby(['year_built'])['sale_day'].transform(lambda x:mode_function1(x))
    df.loc[df.sale_day.isnull(),"sale_day"] = df.groupby(['property_type_1'])['sale_day'].transform(lambda x:mode_function1(x))
    
    df.drop(labels=['sale_date__c'],inplace=True,axis=1)

    #class,num_buildings
    df.loc[df['class'].isnull(),"class"] = 'Class D'
    
    df.loc[df.num_buildings>300,'num_buildings']=np.nan
    df.loc[df.num_buildings.isnull(),"num_buildings"] = df.groupby(['city','county','market','property_type_1'])['num_buildings'].transform(lambda x:x.mean())
    df.loc[df.num_buildings.isnull(),"num_buildings"] = df.groupby(['county','market','property_type_1'])['num_buildings'].transform(lambda x:x.mean())
    df.loc[df.num_buildings.isnull(),"num_buildings"] = df.groupby(['market','property_type_1'])['num_buildings'].transform(lambda x:x.mean())
    df.loc[df.num_buildings.isnull(),"num_buildings"] = df.groupby(['property_type_1'])['num_buildings'].transform(lambda x:x.mean())
    
    #num_parking_spaces,size_units,building_tax_expenses
    df.loc[df.num_parking_spaces.isnull(),"num_parking_spaces"] = df.groupby(['city','market','year_built','property_type_1'])['num_parking_spaces'].transform(lambda x:x.mean())
    df.loc[df.num_parking_spaces.isnull(),"num_parking_spaces"] = df.groupby(['market','year_built','property_type_1'])['num_parking_spaces'].transform(lambda x:x.mean())
    df.loc[df.num_parking_spaces.isnull(),"num_parking_spaces"] = df.groupby(['property_type_1','year_built'])['num_parking_spaces'].transform(lambda x:x.mean())
    df.loc[df.num_parking_spaces.isnull(),"num_parking_spaces"] = df.groupby(['property_type_1'])['num_parking_spaces'].transform(lambda x:x.mean())
    
    #size_units
    df.loc[(df.size_units==0 ),'size_units']=np.nan
    df.loc[df.size_units.isnull(),"size_units"] = df.groupby(['city','county','market','property_type_1'])['size_units'].transform(lambda x:x.mean())
    df.loc[df.size_units.isnull(),"size_units"] = df.groupby(['county','market','property_type_1'])['size_units'].transform(lambda x:x.mean())
    df.loc[df.size_units.isnull(),"size_units"] = df.groupby(['market','property_type_1'])['size_units'].transform(lambda x:x.mean())
    df.loc[df.size_units.isnull(),"size_units"] = df.groupby(['property_type_1'])['size_units'].transform(lambda x:x.mean())
    
    #price_per_unit
    df.loc[df.price_per_unit.isnull(),"price_per_unit"] = df.apply(lambda x:x.price_per_sq_ft*x.size_sf,axis=1)
    
    #todo building_tax_expenses calculation
    df.drop(labels=['building_tax_expenses'],inplace=True,axis=1)
    
    #
    #Number of properties in the market
    market_count = df['market'].value_counts().to_dict()
    df['N-market_count'] = df['market'].map(market_count)
    
    #Number of properties in the city
    city_count = df['city'].value_counts().to_dict()
    df['N-city_count'] = df['city'].map(city_count)
    
    #Number of properties in the couty
    region_count = df['county'].value_counts().to_dict()
    df['N-county_count'] = df['county'].map(region_count)    
    return df

def opportunities_missing_values(df):
#    replacing with mode
    df.loc[df.property_type.isnull(),'property_type'] = 'Apartments'
    df.loc[df.property_group.isnull(),'property_group'] = 'Multi-Housing'
    df.loc[df.deal_type.isnull(),'deal_type'] = df.groupby(['property_group','property_type'])['deal_type'].transform(lambda x: mode_function1(x))    
    df.loc[df.platform.isnull(),'platform'] = df.groupby(['property_group','property_type'])['platform'].transform(lambda x: mode_function1(x))    
#  debt_yield 50 percent value 0 
#  except platform debt all have either 0 or nan values   
    df.loc[df.debt_yield.isnull(),'debt_yield'] = df.groupby(['platform','property_group','deal_type'])['debt_yield'].transform(lambda x: x.mean())
    df.loc[df.debt_yield.isnull(),'debt_yield'] = df.groupby(['platform','deal_type'])['debt_yield'].transform(lambda x: x.mean())
    df.loc[df.debt_yield.isnull(),'debt_yield'] = df.groupby(['platform'])['debt_yield'].transform(lambda x: x.mean())
    df.loc[df.debt_yield.isnull(),'debt_yield'] = 0
    df['nonzero_debt_yield'] = df.apply(lambda x : 1 if x.debt_yield > 0 else 0,axis=1)
# best_initial_bid 23368 values 0
    df.loc[df.best_initial_bid.isnull(),'best_initial_bid'] = df.groupby(['platform','property_group','deal_type'])['best_initial_bid'].transform(lambda x: x.median())
    df.loc[df.best_initial_bid.isnull(),'best_initial_bid'] = df.groupby(['platform','deal_type'])['best_initial_bid'].transform(lambda x: x.median())
    df['nonzero_best_initial_bid'] = df.apply(lambda x : 1 if x.best_initial_bid > 0 else 0,axis=1)
#   accounting_date ==> closeddate and date_closed==>fiscal
    
    df['accounting_date'] = pd.to_datetime(df['accounting_date'])
    df['accounting_date_year'] = pd.to_datetime(df['accounting_date']).dt.year
    df['accounting_date_month'] = pd.to_datetime(df['accounting_date']).dt.month
    df['accounting_date_day'] = pd.to_datetime(df['accounting_date']).dt.day
    df['date_closed'] = pd.to_datetime(df['date_closed'])
    df['date_closed_year'] = pd.to_datetime(df['date_closed']).dt.year
    df['date_closed_month'] = pd.to_datetime(df['date_closed']).dt.month
    df['date_closed_day'] = pd.to_datetime(df['date_closed']).dt.day
    df['closedate'] = pd.to_datetime(df['closedate'])    
    df['closedate_year'] = pd.to_datetime(df['closedate']).dt.year
    df['closedate_month'] = pd.to_datetime(df['closedate']).dt.month
    df['closedate_day'] = pd.to_datetime(df['closedate']).dt.day
    df.loc[df.accounting_date_year.isnull(),'accounting_date_year'] = df['date_closed_year']
    df.loc[df.date_closed_year.isnull(),'date_closed_year'] = df['accounting_date_year']
    df.loc[df.accounting_date_year.isnull(),'accounting_date_year'] = df.groupby(['platform','property_group','deal_type'])['accounting_date_year'].transform(lambda x: mode_function1(x))
    df.loc[df.accounting_date_day.isnull(),'accounting_date_day'] = df.groupby(['platform','property_group','deal_type'])['accounting_date_day'].transform(lambda x: mode_function1(x))
    df.loc[df.accounting_date_day.isnull(),'accounting_date_day'] = df.groupby(['platform','deal_type'])['accounting_date_day'].transform(lambda x: mode_function1(x))
    df.loc[df.accounting_date_month.isnull(),'accounting_date_month'] = df.groupby(['platform','property_group','deal_type'])['accounting_date_month'].transform(lambda x: mode_function1(x))
    df.loc[df.accounting_date_month.isnull(),'accounting_date_month'] = df.groupby(['platform','deal_type'])['accounting_date_month'].transform(lambda x: mode_function1(x))
    df.loc[df.date_closed_year.isnull(),'date_closed_year'] = df.groupby(['platform','property_group','deal_type'])['date_closed_year'].transform(lambda x: mode_function1(x))
    df.loc[df.date_closed_month.isnull(),'date_closed_month'] = df.groupby(['platform','property_group','deal_type'])['date_closed_month'].transform(lambda x: mode_function1(x))
    df.loc[df.date_closed_month.isnull(),'date_closed_month'] = df.groupby(['platform','deal_type'])['date_closed_month'].transform(lambda x: mode_function1(x))
    df.loc[df.date_closed_day.isnull(),'date_closed_day'] = df.groupby(['platform','property_group','deal_type'])['date_closed_day'].transform(lambda x: mode_function1(x))
    df.loc[df.date_closed_day.isnull(),'date_closed_day'] = df.groupby(['platform','deal_type'])['date_closed_day'].transform(lambda x: mode_function1(x))
    df.drop(labels = ['accounting_date','date_closed'],inplace=True,axis=1)

    return df

def remove_redundant_columns():
    # column id_props and id_deals is same removing one
    res = properties['id_props'] ==properties['id_deals']
    res.value_counts()
    properties.drop(labels=['id_deals'],inplace=True,axis=1)
    
    # column id and id_deals is same removing one
    res = accounts_properties['id'] ==accounts_properties['id_deals']
    res.value_counts()
    accounts_properties.drop(labels=['id'],inplace=True,axis=1)    
    
    # column id_deals.1 and id_deals is same removing one
    res = opportunities['id_deals.1'] ==opportunities['id_deals']
    res.value_counts()
    opportunities.drop(labels=['id_deals.1'],inplace=True,axis=1)
    
    # column createdbyid contain single value    
    deals_to_Properties.createdbyid.value_counts()
    deals_to_Properties.drop(labels=['createdbyid'],inplace=True,axis=1)


def find_remove_skew(data,cols):
    skew_feat = data[cols].apply(lambda x : skew(x))
    skew_feat =skew_feat[skew_feat>0.3].index
    print(skew_feat)
    for feat in skew_feat:
        if(data[feat].min() < 0 ):
            data[feat],_ = boxcox(data[feat]+1 - data[feat].min())
        else:
            data[feat],_ = boxcox(data[feat]+1)

def label_encode(data,columns,en=1):
    if en ==1:
        lb = LabelEncoder()
        for col in columns:
            data[col] = lb.fit_transform(data[col].astype('str'))
    else:
        ndf = data[columns]
        for x in columns:
            ndf[x] = ndf[x].astype('category')
        
        ndf = pd.get_dummies(ndf,prefix=columns,columns=columns)
        ndfcolumns = ndf.columns.tolist()
        
        for x in ndfcolumns:
            data[x] = ndf[x].astype('int8')
        data.drop(labels=columns,axis=1,inplace=True)    


#missing_values = count_missing_values(opportunities)
properties = properties_missing_values(properties)
opportunities = opportunities_missing_values(opportunities)
properties.to_csv('properties.csv',index=False)
opportunities.to_csv('opportunities.csv',index=False)

remove_redundant_columns()

#accounts
#'id_accs', 'active_deals', 'activity_count', 'buyer_book',
#   'servicing_contract', 'investor_type', 'cmbs'7, 'consultant',
#   'correspondent'15, 'foreign'178, 'master_servicer'2, 'lender_book'1247,
#   'loan_sales_book'1792, 'loan_servicing'1, 'num_deals_as_client',
#   'num_deals_as_investor', 'number_of_properties',
#   'number_of_related_deals', 'number_of_related_properties',
#   'number_of_won_deals_as_client'

# drop columns consultant from accounts all false
accounts.drop(labels=['consultant'],inplace=True,axis=1)
acc_cat_column = [x for x in accounts.columns if ((accounts[x].dtype=='object')or(accounts[x].dtype=='bool'))and( x!='id_accs')]
acc_num_column = [x for x in accounts.columns if x not in acc_cat_column and( x!='id_accs') ]
find_remove_skew(accounts,acc_num_column)
label_encode(accounts,acc_cat_column,1)
#properties
properties.dtypes
['id_props', 'building_status', 'city', 'class', 'county', 'sale_status',
       'portfolio', 'market', 'num_buildings', 'num_floors',
       'num_parking_spaces', 'occupancy_rate', 'price_per_sq_ft',
       'price_per_unit', 'property_type_1', 'region__c', 'sale_amount__c',
       'size_sf', 'size_units', 'year_built', 'sale_month', 'sale_year',
       'sale_day']

plot_numerical_feature(properties,'sale_day')
plot_categorical_feature(properties,'region__c')
accounts.loan_servicing.value_counts()

prop_cont_column = ['num_buildings','size_units','size_sf','num_floors','sale_amount__c','price_per_unit','num_parking_spaces','occupancy_rate','price_per_sq_ft',]
prop_cat_column = [x for x in properties.columns if (properties[x].dtype=='object')or(properties[x].dtype=='bool')]
find_remove_skew(properties,prop_cont_column)
label_encode(properties,['building_status','class','property_type_1','region__c'],0)
label_encode(properties,['city','county','sale_status','portfolio','market'],1)

#table accounts_properties 
# columns id_deals 
# columns id_props from properties table
# columns id_accs from accs table

accounts_properties.drop_duplicates(inplace =True)
accounts_properties_df = accounts_properties.merge(accounts,how='inner',left_on='id_accs',right_on='id_accs')
accounts_properties_df = accounts_properties_df.merge(properties,how='inner',left_on='id_props',right_on='id_props')

#table deals_to_properties 
# columns id_deals from opportunity table
# columns id_props from properties table
# columns id from 
# columns createdbyid from contain same value hence removed
deals_to_Properties.drop_duplicates(inplace =True)
deals_to_Properties_df = deals_to_Properties.merge(opportunities,how='inner',left_on='id_deals',right_on='id_deals')
deals_to_Properties_df = deals_to_Properties_df.merge(properties,how='inner',left_on='id_props',right_on='id_props')
deals_to_Properties_df = deals_to_Properties_df.merge(accounts,how='inner',left_on='id_accs',right_on='id_accs')


#
opportunities.drop_duplicates(inplace =True)
opportunities_df = opportunities.merge(accounts,how='inner',left_on='id_accs',right_on='id_accs')

    
def knn_get_top_prop(prop_id):
    cols = [x for x in properties.columns if x not in ['id_props','cluster']]
    X = properties.loc[properties.id_props==prop_id][cols]   
    distance,indices = prop_neighbors.kneighbors(X)
    return distance,indices

    
def knn_get_top_accs(acc_id): 
    cols = [x for x in accounts.columns if x not in ['id_accs','cluster']]
    X = accounts.loc[accounts.id_accs==acc_id][cols]   
    distance,indices = acc_neighbors.kneighbors(X)
    return distance,indices
     
    
def get_properties_purch_by_accs(ids_accs):
    properties = accounts_properties_df[accounts_properties_df.id_accs.isin(ids_accs)]['id_props'].tolist()
    return np.unique(properties)   
    
    
def get_properties_recommended_for_account(accs_id):
    distance,indices = knn_get_top_accs(accs_id)
    threshold_ind = indices[0]
#    for i in range(10):
#        if(distance[0][i] < 1.5):
#            threshold_ind.append(indices[0][i])
#            
#    if len(threshold_ind)==0:
#        threshold_ind = indices
        
    similar_accounts = accounts["id_accs"][threshold_ind].tolist()
    ids_props = get_properties_purch_by_accs(similar_accounts)
    print('account ',accs_id,' is recommended ',len(ids_props),' properties purchased by',len(similar_accounts))
    df = pd.DataFrame(columns = ['distance','indices'])    
    for id_props in ids_props:
        distance,indices = knn_get_top_prop(id_props)
        df = df.append(pd.DataFrame(np.vstack((distance[0],indices[0])).T,columns=['distance','indices']))
    
    df['indices']=df.indices.astype('int')
    indices = df[df.distance<20]['indices'].tolist()
    return properties_df['id_props'][indices].unique().tolist()
    

def get_recommendations():
    sub = pd.DataFrame(columns = ['id_accs','id_prop'])
    for accs in list(test['id_accs']):
        ids_props = get_properties_recommended_for_account(accs)
        ids_accs = [accs for i in range(len(ids_props))]
        sub = sub.append(pd.DataFrame(np.vstack((ids_accs,ids_props)).T,columns=['id_accs','id_prop']),ignore_index=True)
    
    sub.to_csv('knn_based.csv',index=False)


cols = [x for x in accounts.columns if x not in ['id_accs','cluster']]
acc_neighbors = NearestNeighbors(n_neighbors=5,algorithm='ball_tree').fit(np.array(accounts[cols]))
cols = [x for x in properties.columns if x not in ['id_props','cluster']]
already_purchased = accounts_properties[accounts_properties.id_props.isin(list(properties['id_props']))]['id_props'].unique()
properties_df = properties[~properties.id_props.isin(already_purchased)].reset_index(drop=True)
properties_df = properties_df[properties_df.building_status_Demolished==0]
X = np.array(properties_df[cols])
prop_neighbors = NearestNeighbors(n_neighbors=5,algorithm='ball_tree').fit(X)

get_recommendations()
sub = pd.read_csv('knn_based.csv')
res = sub.id_accs.value_counts()