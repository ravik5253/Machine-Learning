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
#properties = properties_missing_values(properties)
#opportunities = opportunities_missing_values(opportunities)
#properties.to_csv('properties.csv',index=False)
#opportunities.to_csv('opportunities.csv',index=False)

remove_redundant_columns()

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




# test 
# no properties are purchased by test
# filter with user matching

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



















# KNN classifier to find similar user and find their interesting products
from sklearn.neighbors import NearestNeighbors

cols = [x for x in accounts.columns if x!='id_accs']
X = np.array(accounts[cols])
train_percent = 0.8
train_size = int(train_percent * len(X))

df_train = X[:train_size]
df_test = X[train_size:]

len_train = len(df_train)
len_test = len(df_test)

neighbors = NearestNeighbors(n_neighbors=4,algorithm='ball_tree').fit(df_train)
distance,indices = neighbors.kneighbors(df_train)


lst = []
for i in range(len_test):
    lst.append(neighbors.kneighbors([df_test[i]])[1])
    first_related_user = [item[0] for item in lst[i]]
    first_related_user = str(first_related_user).strip('[]')
    first_related_user = int(first_related_user)
    second_related_user = [item[1] for item in lst[i]]
    second_related_user = str(second_related_user).strip('[]')
    second_related_user = int(second_related_user)
    
    print ("Based on  reviews, for user is ", accounts["id_accs"][len_train + i])
    print ("The first similar user is ", accounts["id_accs"][first_related_user], ".") 
    print ("He/She likes following properties")
    properti = accounts_properties_df[accounts_properties_df.id_accs==accounts["id_accs"][first_related_user]]['id_props']
    
    print ( list(properti))
    print ("--------------------------------------------------------------------")
    
    print ("Based on  reviews, for user is ", accounts["id_accs"][len_train + i])
    print ("The second similar user is ", accounts["id_accs"][second_related_user], ".") 
    print ("He/She likes following properties")
    properti = accounts_properties_df[accounts_properties_df.id_accs==accounts["id_accs"][second_related_user]]['id_props']
    
    print ( list(properti))
    print ("--------------------------------------------------------------------")


from sklearn.model_selection import train_test_split

train_accounts = accounts[~accounts.id_accs.isin(list(test['id_accs']))]
interactions_full_df = accounts_properties_df[accounts_properties_df.building_status_Demolished==0]
train_len = (int)(len(train_accounts) * 0.7 ) 
interactions_train_df = interactions_full_df[interactions_full_df.id_accs.isin(list(train_accounts[:train_len]['id_accs']))]
interactions_test_df = interactions_full_df[interactions_full_df.id_accs.isin(list(train_accounts[train_len:]['id_accs']))]

interactions_full_indexed_df = interactions_full_df.set_index('id_accs')
interactions_train_indexed_df = interactions_train_df.set_index('id_accs')
interactions_test_indexed_df = interactions_test_df.set_index('id_accs')

def get_propertys_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    return set([])
    interacted_propertys = interactions_df.loc[person_id]['id_props']
    return set(interacted_propertys if type(interacted_propertys) == pd.Series else [interacted_propertys])

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_propertyS = 100

import random

class ModelEvaluator:

    def get_not_interacted_propertys_sample(self, person_id, sample_size, seed=42):
        interacted_propertys = get_propertys_interacted(person_id, interactions_full_indexed_df)
        all_propertys = set(properties['id_props'])
        non_interacted_propertys = all_propertys - interacted_propertys

        random.seed(seed)
        non_interacted_propertys_sample = random.sample(non_interacted_propertys, sample_size)
        return set(non_interacted_propertys_sample)

    def _verify_hit_top_n(self, property_id, recommended_propertys, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_propertys) if c == property_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the propertys in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['id_props']) == pd.Series:
            person_interacted_propertys_testset = set(interacted_values_testset['id_props'])
        else:
            person_interacted_propertys_testset = set([(interacted_values_testset['id_props'])])  
        interacted_propertys_count_testset = len(person_interacted_propertys_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_propertys(person_id, 
                                               propertys_to_ignore=get_propertys_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each property the user has interacted in test set
        for property_id in person_interacted_propertys_testset:
            #Getting a random sample (100) propertys the user has not interacted 
            #(to represent propertys that are assumed to be no relevant to the user)
            non_interacted_propertys_sample = self.get_not_interacted_propertys_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_propertyS, 
                                                                          )

            #Combining the current interacted property with the 100 random propertys
            propertys_to_filter_recs = non_interacted_propertys_sample.union(set([property_id]))

            #Filtering only recommendations that are either the interacted property or from a random sample of 100 non-interacted propertys
            valid_recs_df = person_recs_df[person_recs_df['id_props'].isin(propertys_to_filter_recs)]                    
            valid_recs = valid_recs_df['id_props'].values
            #Verifying if the current interacted property is among the Top-N recommended propertys
            hit_at_5, index_at_5 = self._verify_hit_top_n(property_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(property_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted propertys that are ranked among the Top-N recommended propertys, 
        #when mixed with a set of non-relevant propertys
        recall_at_5 = hits_at_5_count / float(interacted_propertys_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_propertys_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_propertys_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df



    
#Computes the most popular properties
properties_popularity_df = interactions_full_df.groupby('id_props')['id_deals'].count().sort_values(ascending=False).reset_index().rename(index=str,columns = {'id_deals':'count'})

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, prop_df=None):
        self.popularity_df = popularity_df
        self.prop_df = prop_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_propertys(self, user_id, propertys_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular propertys that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['id_props'].isin(propertys_to_ignore)] \
                               .sort_values('count', ascending = False) \
                               .head(topn)

        if verbose:
            if self.prop_df is None:
                raise Exception('"propertys_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.prop_df, how = 'left', 
                                                          left_on = 'id_props', 
                                                          right_on = 'id_props')


        return recommendations_df
    
    
popularity_model = PopularityRecommender(properties_popularity_df,properties)

res = popularity_model.recommend_propertys('0012A000023XlbuQAC')

model_evaluator = ModelEvaluator()
print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)




def build_users_profiles(): 
    interactions_indexed_df = interactions_full_df[interactions_full_df['id_props'] \
                                                   .isin(properties['id_props'])].set_index('id_accs')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles


def build_users_profile(person_id, interactions_indexed_df):
    col = [x for x in interactions_indexed_df.columns if x not in ['id_props','id_deals'] ]
    interactions_person_df = interactions_indexed_df.loc[person_id]
    #average of property profiles by the interactions strength
    if type(interactions_person_df) != pd.Series :
        user_property_avg= interactions_person_df[col].mean()
    else:
        user_property_avg = interactions_person_df[col]
        
    user_profile_norm = sklearn.preprocessing.normalize(user_property_avg.reshape(1,-1))
    return user_profile_norm


user_profiles = build_users_profiles()

  
def build_accounts_profiles(): 
    user_profiles = {}
    user_profiles_matrix = []
    for person_id in accounts['id_accs'].unique():
            col = [x for x in accounts.columns if x not in ['id_accs'] ]
            interactions_person_df = accounts[accounts.id_accs == person_id][col].values
            user_profiles[person_id] = sklearn.preprocessing.normalize((interactions_person_df).reshape(1,-1))
            user_profiles_matrix.append(interactions_person_df.reshape(-1))
    return user_profiles,user_profiles_matrix

account_profiles,account_profiles_matrix = build_accounts_profiles()


from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds


class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, propertys_df=None):
        self.property_ids = interactions_full_indexed_df['id_props'].tolist()
        self.propertys_df = propertys_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_propertys_to_user_profile(self, person_id, topn=10):
        #Computes the cosine similarity between the user profile and all property profiles
        col = [x for x in interactions_full_indexed_df.columns if x not in ['id_props','id_deals'] ]
        user_profiles_matrix = np.array(interactions_full_df[col])
        cosine_similarities = cosine_similarity(user_profiles[person_id], user_profiles_matrix)
        #Gets the top similar propertys
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar propertys by similarity
        similar_propertys = sorted([(self.property_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_propertys
        
    def recommend_propertys(self, user_id, propertys_to_ignore=[], topn=10, verbose=False):
        similar_propertys = self._get_similar_propertys_to_user_profile(user_id)
        #Ignores propertys the user has already interacted
        similar_propertys_filtered = list(filter(lambda x: x[0] not in propertys_to_ignore, similar_propertys))
        
        recommendations_df = pd.DataFrame(similar_propertys_filtered, columns=['id_props', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.propertys_df is None:
                raise Exception('"propertys_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.propertys_df, how = 'left', 
                                                          left_on = 'id_props', 
                                                          right_on = 'id_props')[['recStrength', 'id_props', 'title', 'url', 'lang']]


        return recommendations_df

    
content_based_recommender_model = ContentBasedRecommender(interactions_full_indexed_df)
print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)


res1 = content_based_recommender_model.recommend_propertys('0012A00002AXHJUQA5')


sub = pd.DataFrame(columns = ['id_accs','id_prop'])
for accs in list(test['id_accs']):
    res1 = content_based_recommender_model.recommend_propertys(accs)['id_props']
    for id_prop in list(res['id_props'].unique()):
        print('here',id_prop,"     ",accs)
        sub = sub.append({'id_accs':accs,'id_prop':id_prop},ignore_index=True)

sub.to_csv('popularity_based_rec.csv',index=False)
sub = pd.read_csv('popularity_based_rec.csv')










#SVD

ind = accounts_properties[['id_accs','id_props']].duplicated()
ind = ind.value_counts()

score_df = pd.DataFrame(columns = ['user','item'],data=[('a','x'),('a','x'),('a','y'),('c','z')])
r_df = score_df.pivot(index = 'user', columns = 'item',values = 'score')

score_df = accounts_properties[['id_accs','id_props']]
score_df['score'] = 1
score_df.drop_duplicates(inplace=True)
r_df = score_df.pivot(index = 'id_accs', columns = 'id_props',values = 'score').fillna(0)
r = r_df.as_matrix()
user_ratings_mean = np.mean(r,axis=1)
r_dmeaned = r - user_ratings_mean.reshape(-1,1)

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(r_dmeaned, k = 50)

sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = r_df.columns)



# clustering algorithm

import time                          # To time processes 
import warnings                      # To suppress warnings
from sklearn import cluster, mixture # For clustering 
from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
warnings.filterwarnings('ignore')

class ClusterMethodList(object) :
    def get_cluster_instance(self, argument,input_data,X):
        method_name = str(argument).lower()+ '_cluster'
        method = getattr(self,method_name,lambda : "Invalid Clustering method")
        return method(input_data,X)
    
    def kmeans_cluster(self,input_data,X):
        km = cluster.KMeans(n_clusters =input_data['n_clusters'] )
        return km.fit_predict(X)
   
    def meanshift_cluster(self,input_data,X):
        ms = cluster.MeanShift(bandwidth=input_data['bandwidth'])
        return  ms.fit_predict(X)
    
    def minibatchkmeans_cluster(self,input_data,X):
        two_means = cluster.MiniBatchKMeans(n_clusters=input_data['n_clusters'])
        return two_means.fit_predict(X)
   
    def dbscan_cluster(self,input_data,X):
        db = cluster.DBSCAN(eps=input_data['eps'])
        return db.fit_predict(X)
    
    def spectral_cluster(self,input_data,X):
        sp = cluster.SpectralClustering(n_clusters=input_data['n_clusters'])
        return sp.fit_predict(X)
   
    def affinitypropagation_cluster(self,input_data,X):
        affinity_propagation =  cluster.AffinityPropagation(damping=input_data['damping'], preference=input_data['preference'])
        affinity_propagation.fit(X)
        return affinity_propagation.predict(X)
       
    
    def birch_cluster(self,input_data,X):
        birch = cluster.Birch(n_clusters=input_data['n_clusters'])
        return birch.fit_predict(X)
   
    def gaussian_mixture_cluster(self,input_data,X):
        gmm = mixture.GaussianMixture( n_components=input_data['n_clusters'], covariance_type='full')
        gmm.fit(X)
        return  gmm.predict(X)
    

# Define Clustering Prcoess

def startClusteringProcess(list_cluster_method,input_data,no_columns,data_set):
    fig,ax = plt.subplots(no_rows,no_columns, figsize=(10,10)) 
    cluster_list = ClusterMethodList()
    i = 0
    j=0
    for cl in list_cluster_method :
        cluster_result = cluster_list.get_cluster_instance(cl,input_data,data_set)
        #convert cluster result array to DataFrame
        data_set[cl] = pd.DataFrame(cluster_result)
        ax[i,j].scatter(data_set.iloc[:, 4], data_set.iloc[:, 5],  c=cluster_result)
        ax[i,j].set_title(cl+" Cluster Result")
        j=j+1
        if( j % no_columns == 0) :
            j= 0
            i=i+1
    plt.subplots_adjust(bottom=-0.5, top=1.5)
    plt.show()
    
    
list_cluster_method = ['KMeans',"MeanShift","MiniBatchKmeans","DBScan","Spectral","AffinityPropagation","Birch","Gaussian_Mixture"]
# For Graph display 
no_columns = 2
no_rows = 4
# NOT all algorithms require this parameter
n_clusters= 3
bandwidth = 0.1 
# eps for DBSCAN
eps = 0.3
## Damping and perference for Affinity Propagation clustering method
damping = 0.9
preference = -200
input_data = {'n_clusters' :  n_clusters, 'eps' : eps,'bandwidth' : bandwidth, 'damping' : damping, 'preference' : preference}

startClusteringProcess(list_cluster_method,input_data,no_columns,accounts.iloc[:,1:])




    
from sklearn import cluster,mixture
from sklearn.cluster import MeanShift as meanShift
from sklearn.cluster import MiniBatchKMeans as miniBatchKMeans
from sklearn.cluster import DBSCAN as dbscan
from sklearn.cluster import KMeans as kMeans
from sklearn.cluster import SpectralClustering as spectralClustering
from sklearn.cluster import AffinityPropagation as affinityPropagation
from sklearn.cluster import Birch as birch
from sklearn.mixture import GaussianMixture as gaussianMixture


#KMEANS 20 FOR properties 14


data = properties


def normalizedData(x):
    normalised = StandardScaler()
    normalised.fit_transform(x)
    return(x)


data.iloc[:,1:] = normalizedData(data.iloc[:,1:])

def Kmeans(x,y):
    km = kMeans(x)
    kmres = km.fit_predict(y)
    print(km.score(y))
    return kmres

def Kmeans_k(y):
    res = []
    for x in range(1,50,1):
        km = kMeans(int(x))
        km.fit(y)
        res.append(km.score(y))
    return res

def clusters(x,y):
    km = affinityPropagation(damping = 0.9,preference = -200)
    kmres = km.fit_predict(y)
    return kmres

km_result = clusters(2,data.iloc[:,1:])
data['DBSCAN'] = pd.DataFrame(km_result)


kmk = Kmeans_k(data.iloc[:,1:])

plt.plot(kmk)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

km_result = Kmeans(500,data.iloc[:,1:])
data['cluster'] = pd.DataFrame(km_result)
data['cluster'].value_counts()
plt.scatter(properties.iloc[:, 0], data.iloc[:, 1],  c=km_result)
plt.show()  
