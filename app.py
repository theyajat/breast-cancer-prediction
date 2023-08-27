#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing required Modules


# In[97]:


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import warnings


# In[98]:


#Raw Breast Cancer data


# In[99]:


df=pd.read_csv('BRCA.csv')
df.head()


# In[100]:


#To increase the Model Accucary we need to eliminate the NULL values


# In[101]:


df.isnull().sum() 


# In[102]:


df=df.dropna()
df.isnull().sum()


# In[103]:


# Now we can do some EDA


# In[104]:


plt.bar(list(df['Gender'].value_counts().keys()),list(df['Gender'].value_counts()),color='Red')
plt.show()


# In[105]:


#Stage of Tumour of the patients
stage=df['Tumour_Stage'].value_counts()
stages_tumour=stage.index
quantity=stage.values

fig=px.pie(df,values=quantity,names=stages_tumour,hole=0.5,title="Tumour Stages of Patients")
fig.show()


# In[106]:


surgery = df["Surgery_type"].value_counts()
surg_ind = surgery.index
quantity = surgery.values
figure = px.pie(df, 
             values=quantity, 
             names=surg_ind,hole = 0.5, 
             title="Type of Surgery of Patients")
figure.show()


# In[107]:


#To train a machine learning model, we need to transform the values of all the Categorical columns
warnings.filterwarnings("ignore")
df["Tumour_Stage"] = df["Tumour_Stage"].map({"I": 1, "II": 2, "III": 3})
df["Histology"] = df["Histology"].map({"Infiltrating Ductal Carcinoma": 1, 
                                           "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma": 3})
df["ER status"] = df["ER status"].map({"Positive": 1})
df["PR status"] = df["PR status"].map({"Positive": 1})
df["HER2 status"] = df["HER2 status"].map({"Positive": 1, "Negative": 2})
df["Gender"] = df["Gender"].map({"MALE": 0, "FEMALE": 1})
df["Surgery_type"] = df["Surgery_type"].map({"Other": 1, "Modified Radical Mastectomy": 2, 
                                                 "Lumpectomy": 3, "Simple Mastectomy": 4})


# In[108]:


df.head()


# In[109]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(max_depth=4)
from sklearn.model_selection import train_test_split


# In[110]:


X = np.array(df[['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 
                   'Tumour_Stage', 'Histology', 'ER status', 'PR status', 
                   'HER2 status', 'Surgery_type']])
y = np.array(df[['Patient_Status']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)


# In[111]:


dtc.fit(X_train,y_train)


# In[112]:


# Prediction
# features = [['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 'Tumour_Stage', 'Histology', 'ER status', 'PR status', 'HER2 status', 'Surgery_type']]
features = np.array([[36.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2,]])
print(dtc.predict(features))


# In[113]:


# accuracy of the model in percentage
dtc.score(X_test,y_test)*100

