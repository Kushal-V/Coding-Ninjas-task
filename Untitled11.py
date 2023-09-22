#!/usr/bin/env python
# coding: utf-8

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


# In[26]:


data = pd.read_csv('C:\cardataset\dataset.csv')


# In[27]:


data.head()


# In[29]:


data.shape


# In[30]:


data.isnull().sum()


# In[31]:


data['Engine'] = data['Engine'].str.replace('CC', '')
data['Mileage'] = data['Mileage'].str.replace('km/kg', '')
data['Mileage'] = data['Mileage'].str.replace('kmpl', '')
data['Power'] = data['Power'].str.replace('bhp', '')
data['Power'] = data['Power'].str.replace('null', '0')
data


# In[32]:


data['Mileage'] = data['Mileage'].astype(str).astype(float)
data['Engine'] = data['Engine'].astype(str).astype(float)
data['Power'] = data['Power'].astype(str).astype(float)


# In[33]:


data['Seats'] = data['Seats'].fillna(data['Seats'].mean())
data['Mileage'] = data['Mileage'].fillna(data['Mileage'].mean())
data['Engine'] = data['Engine'].fillna(data['Engine'].mean())
data['Power'] = data['Power'].fillna(data['Power'].mean())
data['New_Price'].fillna(data['Price'], inplace=True)


# In[34]:


data


# In[35]:


def convert_cr_to_lakh(value):
    if isinstance(value, str) and 'Cr' in value:
        parts = value.split(' ')
        numeric_part = float(parts[0])
        if parts[1] == 'Cr':
            lakh_value = numeric_part * 100
            return f'{lakh_value:.1f} Lakh'
    return value


# In[36]:


data['New_Price'] = data['New_Price'].apply(convert_cr_to_lakh)
data


# In[37]:


def remove_lakh(value):
    if isinstance(value, str) and 'Lakh' in value: 
        return value.replace(' Lakh', '')
    else: 
        return value


# In[38]:


data['New_Price'] = data['New_Price'].apply(remove_lakh)
data


# In[39]:


print(data.Location.value_counts())
print(data.Year.value_counts())
print(data.Fuel_Type.value_counts())
print(data.Transmission.value_counts())
print(data.Owner_Type.value_counts())


# In[40]:


data.replace({'Location': {'Mumbai':0,'Hyderabad':1,'Kochi':2,'Coimbatore':3,'Pune':4,'Delhi':5,'Kolkata':6,'Chennai':7,'Jaipur':8,'Bangalore':9,'Ahmedabad':10 }}, inplace=True)
data.replace({'Fuel_Type': {'Diesel':0,'Petrol':1,'CNG':2,'LPG':3,'Electric':4 }}, inplace=True)
data.replace({'Transmission': {'Manual':0,'Automatic':1}}, inplace=True)
data.replace({'Owner_Type': { 'First':1,'Second':2,'Third':3,'Fourth & Above':4}}, inplace=True)
data = data.drop(['Price','Name','Unnamed: 0'], axis=1)
data


# In[51]:


X = data.drop(['New_Price'], axis=1)
Y = data['New_Price']


# In[61]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[62]:


model = LinearRegression()


# In[63]:


model.fit(X_train,Y_train)


# In[64]:


pred_train = model.predict(X_train)


# In[65]:


mean_squared_error(Y_train,pred_train)


# In[70]:


plt.scatter(Y_train,pred_train)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()


# In[67]:


pred_test = model.predict(X_test)


# In[68]:


mean_squared_error(Y_test,pred_test)


# In[89]:


plt.scatter(Y_test,pred_test)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

