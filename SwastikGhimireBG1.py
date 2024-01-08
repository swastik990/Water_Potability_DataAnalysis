#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessery libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


#loading the dataset
wqds=pd.read_csv('water_potability.csv')
wqds


# In[3]:


wqds.head(10)


# In[4]:


wqds.describe()


# In[5]:


wqds.info()


# # Analysis of potability

# ### For drinkable water

# In[6]:


wqds[wqds["Potability"] == 1].describe()


# ### For non-drinkable water

# In[7]:


wqds[wqds["Potability"] == 0].describe()


# ### Drinkable vs non-drinkable water in piechart

# In[8]:


import plotly.express as px
d = pd.DataFrame(wqds["Potability"].value_counts())
fig = px.pie(d,values="Potability",names=["Not Drinkable","Drinkable"])
fig.update_layout(title = dict(text = "Fig: Pie chart of potability of water"))
fig.show()


# ##### Correlation matrix To see if there are continuous features that are mutually associated

# In[9]:


wqds.corr()


# In[10]:


import seaborn as sb
sb.clustermap(wqds.corr(), cmap = "coolwarm", dendrogram_ratio = (0.1, 0.2), annot = True, linewidths = .8, figsize = (9,10))
plt.show()


# # Analysis of null values

# In[11]:


wqds.isna().sum()


# In[12]:


#filling null values with mean before further analysis
columns_to_fill = ["ph", "Sulfate", "Trihalomethanes"]

for column in columns_to_fill:
    wqds[column].fillna(value=wqds[column].mean(), inplace=True)


# In[13]:


#after filling null values
wqds.head(10)


# # Data pre processing for model

# In[14]:


X = wqds.drop("Potability", axis = 1).values
y = wqds["Potability"].values


# In[15]:


#testing and training the dataset to build a model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 55)
print("X_train",X_train.shape)
print("X_test",X_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)


# In[16]:


x_train_max = np.max(X_train)
x_train_min = np.min(X_train)
X_train = (X_train - x_train_min)/(x_train_max-x_train_min)
X_test = (X_test - x_train_min)/(x_train_max-x_train_min)


# # Building Model using Decision Tree

# In[17]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, confusion_matrix

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
model_result = model.predict(X_test)
score = precision_score(y_test, model_result)

#confusion to visualise performence
cm = confusion_matrix(y_test, model_result)

finalResults = "Model1:", score
cmList = [("Model1", cm)]
finalResults


# In[18]:


#visualising confusing matrix
for name, i in cmList:
    plt.figure()
    sb.heatmap(i, annot = True, fmt = ".1f")
    plt.title(name)
    plt.xlabel('fig: Confusion Matrix')
    plt.show()


# # Decision tree visualization

# In[19]:


from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(25, 20))
tree.plot_tree(model,
               feature_names=wqds.columns.tolist()[:-1],
               class_names=["0", "1"],
               filled=True,
               precision=5)
plt.show()


# # Building model using logistic regression

# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)


# In[21]:


accuracy = accuracy_score(y_test, logistic_predictions)
cm = confusion_matrix(y_test,logistic_predictions)
print("Model2:", accuracy)


# In[22]:


sb.heatmap(cm, annot=True, cmap='Blues', fmt = ".1f")
plt.xlabel('fig: Confusion Matrix')
plt.ylabel('Actual Label')
plt.title('Model2')
plt.show()


# # Comparing two models used

# In[25]:


model_names = ['Model1: Decision Tree','Model2: Logistic Regression']
accuracy_scores = [0.79, 0.60]


# In[26]:


plt.bar(model_names, accuracy_scores)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model accuracy comparision')
for i, score in enumerate(accuracy_scores):
    plt.text(i, score, f'{score:.2f}', ha='center', va='bottom')
plt.show()


# In[ ]:




