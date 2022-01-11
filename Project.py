#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
pandas.__version__


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# In[5]:


import time


# In[6]:


import sklearn.metrics as m


# In[7]:


conda install -c glemaitre imbalanced-learn


# In[8]:


get_ipython().system('pip install imblearn')


# In[9]:


from imblearn.over_sampling import SMOTE


# In[10]:


import imblearn


# In[11]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[12]:


#Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[13]:


#Load Data
my_dataset=pd.read_csv("my_dataset.csv")


# In[14]:


my_dataset.info()


# In[15]:


my_dataset.head()


# In[16]:


drop_cols = []
for i in my_dataset.columns:
    if len(my_dataset[i].unique())==1:
        drop_cols.append(i)
print("Total columns with only 1 unique value:", len(drop_cols))
my_dataset.drop(drop_cols, 1, inplace=True)


# In[17]:


my_dataset.dropna(1,inplace=True)


# In[18]:


my_dataset.info()


# In[19]:


my_dataset= my_dataset.dropna('columns')# drop columns with NaN

my_dataset= my_dataset[[col for col in my_dataset if my_dataset[col].nunique() > 1]]# keep columns where there are more than 1 unique values

corr = my_dataset.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr)

plt.show()


# In[20]:


#Split dataset on train and test
from sklearn.model_selection import train_test_split
train, test=train_test_split(my_dataset,test_size=0.3, random_state=10)

#Exploratory Analysis
# Descriptive statistics
train.describe()
test.describe()


# In[21]:


# Packet Attack Distribution
train['Label'].value_counts()
test['Label'].value_counts()


# In[22]:


#Scalling numerical attributes
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[23]:


# extract numerical attributes and scale it to have zero mean and unit variance  
cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))


# In[24]:


# turn the result back to a dataframe
sc_traindf = pd.DataFrame(sc_train, columns = cols)
sc_testdf = pd.DataFrame(sc_test, columns = cols)


# In[25]:


# importing one hot encoder from sklearn 
from sklearn.preprocessing import OneHotEncoder 


# In[26]:


# creating one hot encoder object 
onehotencoder = OneHotEncoder() 


# In[27]:


trainDep = train['Label'].values.reshape(-1,1)
trainDep = onehotencoder.fit_transform(trainDep).toarray()
testDep = test['Label'].values.reshape(-1,1)
testDep = onehotencoder.fit_transform(testDep).toarray()


# In[28]:


train_X=sc_traindf
train_y=trainDep[:,0]

test_X=sc_testdf
test_y=testDep[:,0]


# In[29]:


#Feature Selection
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier();

# fit random forest classifier on the training set
rfc.fit(train_X, train_y);

# extract important features
score = np.round(rfc.feature_importances_,3)
importances = pd.DataFrame({'feature':train_X.columns,'importance':score})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

# plot importances
plt.rcParams['figure.figsize'] = (11, 4)
importances.plot.bar();


# In[30]:


#Recursive feature elimination
from sklearn.feature_selection import RFE
import itertools

rfc = RandomForestClassifier()

# create the RFE model and select 10 attributes
rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(train_X, train_y)

# summarize the selection of the attributes
feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_X.columns)]
selected_features = [v for i, v in feature_map if i==True]

selected_features

a = [i[0] for i in feature_map]
train_X = train_X.iloc[:,a]
test_X = test_X.iloc[:,a]


# In[31]:


#Dataset Partition
X_train,X_test,Y_train,Y_test = train_test_split(train_X,train_y,train_size=0.70, random_state=2)

#Fitting Models
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[32]:


# Train KNeighborsClassifier Model
KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(X_train, Y_train); 


# In[33]:


# Train LogisticRegression Model
LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(X_train, Y_train);


# In[34]:


# Train Gaussian Naive Baye Model
BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X_train, Y_train)


# In[35]:


# Train Decision Tree Model
DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, Y_train)


# In[36]:


# Train SVM
SVM_Classifier = SVC(gamma = 'scale')
SVM_Classifier.fit(X_train, Y_train)


# In[40]:


#Evaluate Models
from sklearn import metrics
models = []
models.append(('Naive Baye Classifier', BNB_Classifier))
models.append(('Decision Tree Classifier', DTC_Classifier))
models.append(('KNeighborsClassifier', KNN_Classifier))
models.append(('LogisticRegression', LGR_Classifier))
models.append(('Support Vector Machine', SVM_Classifier))

for i, v in models:
    start_time = time.time()
    scores = cross_val_score(v, X_train, Y_train, cv=10)
    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
    classification = metrics.classification_report(Y_train, v.predict(X_train))
    MAE = metrics.mean_absolute_error(Y_train, v.predict(X_train))
    R2_Score = metrics.r2_score(Y_train, v.predict(X_train))
    F1_Score = metrics.f1_score(Y_train, v.predict(X_train))
    end_time = time.time()
    print()
    print('============================== {} Model Evaluation =============================='.format(i))
    print()
    print ("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()
    print("Mean Absolute Error:" "\n", MAE) 
    print()
    print("R2_score:" "\n", R2_Score ) 
    print()
    print("F1_score:" "\n", F1_Score ) 
    print()
    print("Testing time: ",end_time-start_time)
    print()
    


# In[41]:


#Validate Models
for i, v in models:
    start_time = time.time()
    accuracy = metrics.accuracy_score(Y_test, v.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(Y_test, v.predict(X_test))
    classification = metrics.classification_report(Y_test, v.predict(X_test))
    MAE = metrics.mean_absolute_error(Y_test, v.predict(X_test))
    R2_Score = metrics.r2_score(Y_test, v.predict(X_test))
    F1_Score = metrics.f1_score(Y_test, v.predict(X_test))
    end_time = time.time()
    print()
    print('============================== {} Model Test Results =============================='.format(i))
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()
    print("Mean Absolute Error:" "\n", MAE)  
    print()
    print("R2_score:" "\n", R2_Score ) 
    print()
    print("F1_score:" "\n", F1_Score ) 
    print()
    print("Testing time: ",end_time-start_time)
    print()


# In[48]:


#Accuracy BarChart of Evaluation Models
names = ['NB','DT','KNN','LR','SVM','ANN']
values = [99.95,100.00,99.99,100.00,100.00,100.00]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.ylim(5,102)
plt.bar(names,values)


# In[52]:


#Accuracy BarChart of Validation Models
names = ['NB','DT','KNN','LR','SVM','ANN']
values = [99.99,100.00,100.00,100.00,99.99,100.00]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.ylim(5,102)
plt.bar(names,values)


# In[49]:


#BarChart of Training Time on Evaluating the Models 
names = ['NB','DT','KNN','LR','SVM','ANN']
values = [3.213,3.0338,3133.770,27.888,42.535,114.953]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.ylim(10,102)
plt.bar(names,values)


# In[51]:


#BarChart of Test Time on Validating the Models 
names = ['NB','DT','KNN','LR','SVM','ANN']
values = [0.440,0.346,55.761,0.319,8.354,0.9023]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.ylim(10,102)
plt.bar(names,values)


# In[ ]:




