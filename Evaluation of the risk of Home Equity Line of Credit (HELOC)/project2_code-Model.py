
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
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
from sklearn import metrics
from sklearn.metrics import roc_auc_score


# In[2]:


file = pd.read_csv('heloc_dataset_v1.csv')


# In[3]:


# detect special values

for i in file.columns:
    count_7 = len(file[file[i] == -7])
    print(i,' -7:',count_7,count_7/file.shape[0])
    count_8 = len(file[file[i] == -8])
    print(i,' -8:',count_8,count_8/file.shape[0])
    count_9 = len(file[file[i] == -9])
    print(i,' -9:',count_9,count_9/file.shape[0])


# In[4]:


# delete all the rows which contain -9
file = file.replace([-9], np.nan)
file = file.dropna()
# file.shape


# In[5]:


X = file.copy().drop('RiskPerformance',axis = 1)
# file_X.shape


# In[6]:


y = file['RiskPerformance']
# file_Y.shape


# In[7]:


#categorical variable
categorical_MaxDelq2PublicRecLast12M = pd.get_dummies(X['MaxDelq2PublicRecLast12M'])
categorical_MaxDelqEver = pd.get_dummies(X['MaxDelqEver'])
file_categorical = pd.concat((categorical_MaxDelq2PublicRecLast12M,categorical_MaxDelqEver),
                             axis=1,ignore_index = True)


# In[8]:


file_numeric = X.copy().drop(['MaxDelq2PublicRecLast12M','MaxDelqEver'],axis = 1)


# In[9]:


X = pd.concat((file_numeric,file_categorical),axis = 1)


# In[10]:


y = pd.factorize(y)[0]


# In[11]:


# The function `init_classifiers` returns a list of classifiers to be trained on the datasets
def init_classifiers():
    return([(SVC(), model_names[0], param_grid_svc), 
            (LogisticRegression(), model_names[1], param_grid_logistic),
            (KNeighborsClassifier(), model_names[2], param_grid_knn),
            (GaussianNB(), model_names[3], param_grid_nb),
            (DecisionTreeClassifier(), model_names[4], param_grid_tree),
            (RandomForestClassifier(), model_names[6], param_grid_rf),
            (AdaBoostClassifier(), model_names[7], param_grid_boost)
           ])

# 'model_names' contains the names  that we will use for the above classifiers
model_names = ['SVM','LR','KNN','NB','Tree','QDA','RF','Boosting']

# the training parameters of each model
param_grid_svc = [{'C':[0.1,1],'kernel':['rbf','linear','poly'], 'max_iter':[-1],'random_state':[1]}]
param_grid_logistic = [{'C':[0.1,1], 'penalty':['l1','l2'],'random_state':[1]}]
param_grid_knn = [{},{'n_neighbors':[1,2,3,4]}]
param_grid_nb = [{}]
param_grid_tree = [{'random_state':[1]},{'criterion':['gini'], 'max_depth':[2,3,4], 'min_samples_split':[3,5],'random_state':[1]}]
param_grid_rf = [{'random_state':[1]},{'n_estimators':[10,20,30],'max_features':[0.2, 0.3], 'bootstrap':[True],'random_state':[1]}]
param_grid_boost = [{'random_state':[1]},{'n_estimators':[10,20,30],'learning_rate':[0.1,0.5,1],'random_state':[1]}]


# In[12]:


def evaluate_model(X,Y,model, model_name, params):
    #split training set and test set
    np.random.seed(1)
    X_train, X_test = train_test_split(X, test_size=0.25, random_state=1)
    y_train, y_test = train_test_split(y, test_size=0.25, random_state=1)
    
    
    #standard scaler
    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X_train)
    X_test_scaled_values = scaler.transform(X_test)
    
    
    grid_search = GridSearchCV(model, params, cv=3)
    grid_search.fit(X_scaled_values,y_train)
    
    
    #evaluation on test set
    final_model = grid_search.best_estimator_
    final_predictions = final_model.predict(X_test_scaled_values)
    
    lin_mse = mean_squared_error(y_test,final_predictions)
    lin_rmse = np.sqrt(lin_mse)
    
    dic={}
    dic['Classifier']= model_name
    dic['Accuracy']= (y_test==final_predictions).mean()
    dic['CV Score']= grid_search.best_score_
    dic['Precision'] = metrics.precision_score(y_test,final_predictions, average='macro')
    dic['Recall'] = metrics.recall_score(y_test,final_predictions, average='macro')
    dic['RMSE'] = lin_rmse
    dic['best estimator'] = final_model
    
    return dic
    pass


# In[13]:


import warnings
warnings.filterwarnings('ignore')


# In[14]:


classifiers = init_classifiers()
res_list = []
for m in range(len(classifiers)):
    res_list.append(evaluate_model(X,y,classifiers[m][0],classifiers[m][1],classifiers[m][2]))
df_model_comparison = pd.DataFrame(res_list)


# In[15]:


df_model_comparison 


# In[17]:


best_model = df_model_comparison.iloc[6]['best estimator']


# In[18]:


best_model


# In[19]:


Tree = df_model_comparison.iloc[4]['best estimator']
Tree


# In[20]:


SVM = df_model_comparison.iloc[0]['best estimator']
SVM


# In[21]:


KNN = df_model_comparison.iloc[2]['best estimator']
KNN


# In[22]:


RF = df_model_comparison.iloc[5]['best estimator']
RF


# In[23]:


LR= df_model_comparison.iloc[1]['best estimator']
LR


# In[ ]:




