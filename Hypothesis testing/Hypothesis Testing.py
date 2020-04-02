
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import re


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[2]:


# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[74]:


def get_list_of_university_towns():
    file = open("university_towns.txt","r") 
    uni_towns = file.readlines()
    file.close()
    uni_towns = [i.replace('\n','') for i in uni_towns]
    state_region = []
    for i in uni_towns:
        if i[-6:] == '[edit]':
            state = i[:-6]
        elif "(" in uni_towns: 
            region = i.split('(')[0].rstrip()
            state_region.append((state,region))
        else:
            region = i.split('(')[0].rstrip()
            state_region.append((state,region))
    state_region = pd.DataFrame(state_region,columns = ['State','RegionName'])

    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
    return state_region


# In[4]:


get_list_of_university_towns()


# In[5]:


def read_GDP():
    GDP = pd.read_excel('gdplev.xls',skiprows = 5,header=0,usecols = [4,6])
    GDP = GDP.iloc[2:]
    GDP = GDP.iloc[212:].reset_index()[[1,2]]
    GDP = GDP.rename(columns = {'Unnamed: 4':'Time','GDP in billions of chained 2009 dollars.1':'GDP'})
    return GDP



# In[6]:


def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    GDP = read_GDP()
    GDP['GDP diff'] = GDP['GDP'].diff()
    GDP_dec = GDP[GDP['GDP diff']<0]
    GDP_dec['Index'] = GDP_dec.index
    GDP_dec['Index diff'] = GDP_dec['Index'].diff()
    min_index = GDP_dec['Index diff'].idxmin()
    recession_start = GDP['Time'].iloc[min_index-1]

    return recession_start


# In[7]:


get_recession_start()


# In[8]:


def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    GDP = read_GDP()
    GDP['GDP diff'] = GDP['GDP'].diff()
    GDP_rise = GDP[GDP['GDP diff']>0]
    GDP_rise['Index'] = GDP_rise.index
    GDP_rise['Index diff'] = GDP_rise['Index'].diff()
    max_index = GDP_rise['Index diff'].idxmax()
    recession_end = GDP['Time'].iloc[max_index+1]  
    return recession_end


# In[9]:


get_recession_end()


# In[10]:


def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    GDP = read_GDP()
    GDP = GDP.set_index('Time')
    GDP_recession = GDP.loc[get_recession_start():get_recession_end()]
    min_gdp = GDP_recession['GDP'].min()
    recession_bottom = GDP_recession.index[GDP_recession['GDP']==min_gdp][0]

    return recession_bottom


# In[11]:


get_recession_bottom()


# In[12]:


def get_quarter():
    housing = pd.read_csv('City_Zhvi_AllHomes.csv')
    housing = housing.drop(list(housing.columns.values)[6:51],axis = 1)
    cols = list(housing.columns.values)#[6:]
    
    for i in range(6,len(cols)):
        if ('-01'in cols[i] )|('-02'in cols[i] )|('-03' in cols[i]):
            cols[i] = cols[i][0:4]+'q1'
            
        elif ('-04'in cols[i] )|('-05'in cols[i] )|('-06' in cols[i]):
            cols[i] = cols[i][0:4]+'q2'
            
        elif ('-07'in cols[i] )|('-08'in cols[i] )|('-09' in cols[i]):
            cols[i] = cols[i][0:4]+'q3'
        
        elif ('-10'in cols[i] )|('-11'in cols[i] )|('-12' in cols[i]):
            cols[i] = cols[i][0:4]+'q4'
            
    return cols
cols = get_quarter()


# In[13]:


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    housing = pd.read_csv('City_Zhvi_AllHomes.csv')
    housing = housing.drop(list(housing.columns.values)[6:51],axis = 1)
    housing.columns = cols
    housing['State'] = housing['State'].map(states)
    housing = housing.set_index(["State","RegionName"]).sort_index()
    housing = housing.drop(housing.columns[[0,1,2,3]],axis=1)
    housing = housing.groupby(housing.columns,axis = 1).mean()
    return housing


# In[55]:


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    
    housing  = convert_housing_data_to_quarters()
    
    before_rec = (housing.columns.get_loc(get_recession_start())-1)
    rec_bottom = housing.columns.get_loc(get_recession_bottom())
    
    all_homes = housing[[before_rec,rec_bottom]].dropna()
    all_homes['price_ratio'] = all_homes['2008q2']/all_homes['2009q2']
    uni = get_list_of_university_towns().set_index(['State','RegionName'])
    uni_values = pd.merge(all_homes,uni,how = 'inner',left_index = True, right_index = True)
    nonuni_values = all_homes.drop(uni_values.index)
    p_value = ttest_ind(uni_values['price_ratio'], nonuni_values['price_ratio']).pvalue
    
    
    if p_value <0.01:
        different = True
    else:
        different = False

    if uni_values['price_ratio'].mean() < nonuni_values['price_ratio'].mean():
        better='university town'
    else:
        better='non-university town'

    return (different, p_value, better)


# In[56]:


run_ttest()


# In[ ]:




