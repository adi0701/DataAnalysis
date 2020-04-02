#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import pickle
import warnings 
import streamlit as st
from sklearn import metrics


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


# model training script


# In[4]:


# Load and split the data
file = pd.read_csv('heloc_dataset_v1.csv')
file = file.replace([-9], np.nan)
file = file.dropna()


# In[5]:


X = file.copy().drop('RiskPerformance',axis = 1)
y = file['RiskPerformance']
# categorical_MaxDelq2PublicRecLast12M = pd.get_dummies(X['MaxDelq2PublicRecLast12M'])
# categorical_MaxDelqEver = pd.get_dummies(X['MaxDelqEver'])
# file_categorical = pd.concat((categorical_MaxDelq2PublicRecLast12M,categorical_MaxDelqEver),axis=1,ignore_index = True)
# file_numeric = X.copy().drop(['MaxDelq2PublicRecLast12M','MaxDelqEver'],axis = 1)
# X = pd.concat((file_numeric,file_categorical),axis = 1)
y = pd.factorize(y)[0]


# In[6]:


np.random.seed(1)
X_train, X_test = train_test_split(X, test_size=0.25, random_state=1)
y_train, y_test = train_test_split(y, test_size=0.25, random_state=1)


# In[7]:


# Create a pipeline to fit the data
pipe_model = Pipeline([('standardscaler', StandardScaler()), ('model',
            AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=1) )])
pipe_model.fit(X_train,y_train)
# Save the data and pipeline
pickle.dump(X_train, open('X_train.sav', 'wb'))
pickle.dump(pipe_model, open('pipe_model.sav', 'wb'))
pickle.dump(X_test, open('X_test.sav', 'wb'))
pickle.dump(y_test, open('y_test.sav', 'wb'))


# In[8]:


# streamlit_demo


# In[9]:


# Load the pipeline and data
pipe = pickle.load(open('pipe_model.sav', 'rb'))
X_test = pickle.load(open('X_test.sav', 'rb'))
y_test = pickle.load(open('y_test.sav', 'rb'))


# In[51]:


dic = {0: 'Bad', 1: 'Good'}


#Function to test certain index of dataset
def test_demo(index):
    values = X_test.iloc[index]  # Input the value from dataset

    # Create sliders in the sidebar

    a = st.sidebar.slider('ExternalRiskEstimate', 0.0, 100.0, values['ExternalRiskEstimate'], 1.0)
    b = st.sidebar.slider('MSinceOldestTradeOpen', 0.0, 1000.0, values['MSinceOldestTradeOpen'], 1.0)
    c = st.sidebar.slider('MSinceMostRecentTradeOpen', 0.0, 500.0, values['MSinceMostRecentTradeOpen'], 1.0)
    d = st.sidebar.slider('AverageMInFile', 0.0, 500.0, values['AverageMInFile'], 1.0)
    e = st.sidebar.slider('NumSatisfactoryTrades', 0.0, 100.0, values['NumSatisfactoryTrades'], 1.0)
    f = st.sidebar.slider('NumTrades60Ever2DerogPubRec', 0.0, 100.0, values['NumTrades60Ever2DerogPubRec'], 1.0)
    g = st.sidebar.slider('NumTrades90Ever2DerogPubRec', 0.0, 100.0, values['NumTrades90Ever2DerogPubRec'], 1.0)
    h = st.sidebar.slider('PercentTradesNeverDelq', 0.0, 100.0, values['PercentTradesNeverDelq'], 1.0)
    i = st.sidebar.slider('MSinceMostRecentDelq', -8.0, 100.0, values['MSinceMostRecentDelq'], 1.0)
    j = st.sidebar.slider('NumTotalTrades', 0.0, 200.0, values['NumTotalTrades'], 1.0)
    k = st.sidebar.slider('NumTradesOpeninLast12M', 0.0, 100.0, values['NumTradesOpeninLast12M'], 1.0)
    l = st.sidebar.slider('PercentInstallTrades', 0.0, 200.0, values['PercentInstallTrades'], 1.0)
    m = st.sidebar.slider('MSinceMostRecentInqexcl7days', -8.0, 50.0, values['MSinceMostRecentInqexcl7days'], 1.0)
    n = st.sidebar.slider('NumInqLast6M', 0.0, 100.0, values['NumInqLast6M'], 1.0)
    o = st.sidebar.slider('NumInqLast6Mexcl7days', 0.0, 100.0, values['NumInqLast6Mexcl7days'], 1.0)
    p = st.sidebar.slider('NetFractionRevolvingBurden', -8.0, 500.0, values['NetFractionRevolvingBurden'], 1.0)
    q = st.sidebar.slider('NetFractionInstallBurden', -8.0, 500.0, values['NetFractionInstallBurden'], 1.0)
    r = st.sidebar.slider('NumRevolvingTradesWBalance', -8.0, 100.0, values['NumRevolvingTradesWBalance'], 1.0)
    s = st.sidebar.slider('NumInstallTradesWBalance', -8.0, 100.0, values['NumInstallTradesWBalance'], 1.0)
    t = st.sidebar.slider('NumBank2NatlTradesWHighUtilization', -8.0, 100.0, values['NumBank2NatlTradesWHighUtilization'], 1.0)
    u = st.sidebar.slider('PercentTradesWBalance', -8.0, 100.0, values['PercentTradesWBalance'], 1.0)                     
    v = st.sidebar.slider('MaxDelq2PublicRecLast12M', -8.0, 100.0, values['MaxDelq2PublicRecLast12M'], 1.0)
    w = st.sidebar.slider('MaxDelqEver', -8.0, 100.0, values['MaxDelqEver'], 1.0)         
    
    
    #Print the prediction result
    alg = ['Best model','Decision Tree', 'Random Forest','Support Vector Machine', 'KNN','Logistic Regression']
    classifier = st.selectbox('Which algorithm?', alg)

    if classifier == 'Decision Tree':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s= scaler.transform(X_test)
        dtc = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=1, splitter='best')
        dtc.fit(X_train_s, y_train)
        acc = dtc.score(X_test_s, y_test)
        res = dtc.predict(np.array([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])       
        st.write('Accuracy: ', acc)
        pred = dtc.predict(X_test_s)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Confusion matrix: ', cm)
        st.write('Precision: ',metrics.precision_score(y_test,pred, average='macro'))
        st.write('Recall: ', metrics.recall_score(y_test,pred, average='macro'))
    

    elif classifier == 'Support Vector Machine':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s= scaler.transform(X_test)
        svm = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
                  kernel='linear', max_iter=-1, probability=False, random_state=1,
                  shrinking=True, tol=0.001, verbose=False)
        svm.fit(X_train_s, y_train)
        acc = svm.score(X_test_s, y_test)
        res = svm.predict(np.array([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])       
        st.write('Accuracy: ', acc)
        pred = svm.predict(X_test_s)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Confusion matrix: ', cm)
        st.write('Precision: ',metrics.precision_score(y_test,pred, average='macro'))
        st.write('Recall: ', metrics.recall_score(y_test,pred, average='macro'))
    
    elif classifier == 'Random Forest':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s= scaler.transform(X_test) 
        rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features=0.2, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=30,
                       n_jobs=None, oob_score=False, random_state=1, verbose=0,
                       warm_start=False)
        rf.fit(X_train_s, y_train)
        acc = rf.score(X_test_s, y_test)
        res = rf.predict(np.array([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])       
        st.write('Accuracy: ', acc)
        pred = rf.predict(X_test_s)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Confusion matrix: ', cm)
        st.write('Precision: ',metrics.precision_score(y_test,pred, average='macro'))
        st.write('Recall: ', metrics.recall_score(y_test,pred, average='macro'))
    
    elif classifier == 'KNN':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s= scaler.transform(X_test)
        knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
        knn.fit(X_train_s, y_train)
        acc = knn.score(X_test_s, y_test)
        res = knn.predict(np.array([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])       
        st.write('Accuracy: ', acc)
        pred = knn.predict(X_test_s)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Confusion matrix: ', cm)
        st.write('Precision: ',metrics.precision_score(y_test,pred, average='macro'))
        st.write('Recall: ', metrics.recall_score(y_test,pred, average='macro'))  
        
    elif classifier == 'Logistic Regression':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s= scaler.transform(X_test)
        lr = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l1',
                   random_state=1, solver='warn', tol=0.0001, verbose=0,
                   warm_start=False)
        lr.fit(X_train_s, y_train)
        acc = lr.score(X_test_s, y_test)
        res = lr.predict(np.array([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])       
        st.write('Accuracy: ', acc)
        pred = lr.predict(X_test_s)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Confusion matrix: ', cm)
        st.write('Precision: ',metrics.precision_score(y_test,pred, average='macro'))
        st.write('Recall: ', metrics.recall_score(y_test,pred, average='macro'))

    else:
        pipe = pickle.load(open('pipe_model.sav', 'rb'))
        res = pipe.predict(np.array([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w]).reshape(1, -1))[0]
        st.write('Prediction:  ', dic[res])
        st.write('Best Classifier Name: ', 'AdaBoost')
        best_model = '''AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=1)'''
        st.write('Best Model: ',best_model)
        pred = pipe.predict(X_test)
        score = pipe.score(X_test, y_test)
        cm = metrics.confusion_matrix(y_test, pred)
        st.write('Accuracy: ', score)
        st.write('Confusion Matrix: ', cm)
        st.write('Precision: ',metrics.precision_score(y_test,pred, average='macro'))
        st.write('Recall: ', metrics.recall_score(y_test,pred, average='macro'))


# In[52]:


# title
st.title('The risk of HELOC')
# show data
if st.checkbox('Show dataframe'):
    st.write(X_test)
# st.write(X_train) # Show the dataset

number = st.text_input('Choose a row of information in the dataset (0~2465):', 5)  # Input the index number


# In[53]:


test_demo(int(number))  # Run the test function


# In[ ]:




