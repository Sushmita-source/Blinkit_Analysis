#!/usr/bin/env python
# coding: utf-8

# ##  DATA ANALYSIS PYTHON PROJECT - BLINKIT ANALYSIS 

# ##### **Import Libraries** 

# In[7]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# ##### **Import Raw Data** 

# In[8]:


df = pd.read_csv("BlinkIT _Grocery _Data.csv")


# ##### **Import Sample Data** 

# In[9]:


df.head(20)


# In[10]:


df.tail(20)


# ##### **Size Of  Data** 

# In[11]:


print("Size of Data:",df.shape)


# ##### **Field info** 

# In[12]:


df.columns


# ##### **Data Types** 

# In[13]:


df.dtypes


# ##### **Data  Cleaning** 

# In[14]:


print(df['Item Fat Content'].unique())


# In[15]:


df['Item Fat Content']=df['Item Fat Content'].replace({'LF':'Low Fat',
                                                       'low fat':'Low Fat',
                                                       'reg':'Regular'})


# In[16]:


print(df['Item Fat Content'].unique())


# ##### **BUSINESS REQUIREMENTS** 

# ##### **KPI'S REQURENMENTS** 

# In[17]:


#Total sales 
total_sales=df['Sales'].sum()

#Average sales 
Avg_sales=df['Sales'].mean()

#No of items
no_of_items=df['Sales'].count()

#Average Rating 
Avg_Rating=df['Rating'].mean()

#Display 
print(f"Total_Sales:${total_sales:,.0f}")
print(f"Average_Sales:${Avg_sales:,.1f}")
print(f"Avg_Rating:{Avg_Rating:,.1f}")
print(f"No_Of_Items:{no_of_items:,.0f}")


# ##### **CHAT'S REQURENMENTS** 

# ##### **Total Sales by Fat Content** 

# In[18]:


sales_by_fat=df.groupby('Item Fat Content')['Sales'].sum()

plt.pie(sales_by_fat,labels=sales_by_fat.index,
                      autopct='%.0f%%' ,
                      startangle =90 )

plt.title("sales by Fat Content")
plt.axis('equal')
plt.show()


# ##### **Total Sales by Item** 

# In[19]:


sales_by_type =df.groupby('Item Type')['Sales'].sum().sort_values(ascending=False)

plt.figure(figsize=(10,6))
bars=plt.bar(sales_by_type.index,sales_by_type .values)

plt.xticks(rotation=-90)
plt.xlabel('Item Type')
plt.ylabel('Total Sales')
plt.title('Total Sales by Item Type')

for bar in bars:
    plt.text(bar.get_x()+bar.get_width()/2,bar.get_height(),
            f'{bar.get_height():,.0f}',ha='center',va='bottom',fontsize=8)

plt.tight_layout()
plt.show()


# ##### **Fat Content by Outlet for Total Sales** 

# In[20]:


grouped=df.groupby(['Outlet Location Type','Item Fat Content'])['Sales'].sum().unstack()
grouped=grouped[['Regular','Low Fat']]

ax=grouped.plot(kind='bar',figsize=(8,5),title='Outlet Tier by item Fat Content')
plt.xlabel('Outlet Location Tier')
plt.ylabel('Total Sales')
plt.legend(title='Item Fat Content')
plt.tight_layout()
plt.show()


# ##### **Total Sales by Outlet Establishment** 

# In[21]:


sales_by_year=df.groupby('Outlet Establishment Year')['Sales'].sum().sort_index()

plt.figure(figsize=(9,5))
plt.plot(sales_by_year.index,sales_by_year.values,marker='o',linestyle='-')

plt.xlabel('Outlet Location Tier')
plt.ylabel('Total Sales')
plt.title='Outlet Establishment'

for x,y in zip(sales_by_year.index,sales_by_year.values):
    plt.text(x,y,f'{y:,.0f}',ha='center',va='bottom',fontsize=8)

plt.tight_layout()
plt.show()


# ##### **Sales by Outlet Location** 

# In[22]:


sales_by_size=df.groupby('Outlet Size')['Sales'].sum()
plt.figure(figsize=(4,4))
plt.pie(sales_by_size,labels=sales_by_size.index,autopct='%1.1f%%',startangle=90)
plt.title='Outlet Size'
plt.tight_layout()
plt.show()


# ##### **Sales by Outlet Location** 

# In[23]:


sales_by_location=df.groupby('Outlet Location Type')['Sales'].sum().reset_index()
sales_by_location = sales_by_location.sort_values('Sales',ascending=False)

plt.figure(figsize=(8,3))
ax=sns.barplot(x='Sales',y='Outlet Location Type',data=sales_by_location)

plt.title='Total Sales Outlet Location Type'
plt.xlabel('Total Sales')
plt.ylabel('Outlet Location Type')

plt.tight_layout()
plt.show()


# ##### **Appy ML Algorithms** 

# ##### **XGBoost Regressor(Extreme Gradient Boosting)** 

# In[24]:


# Importing some libraries related to ML Algorithms
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# ##### **Handle categorical features with Label Encoding** 

# In[26]:


le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col].astype(str))


# ##### **Features and Target** 

# In[32]:


X = df.drop("Sales", axis=1)
y = df["Sales"]


# In[33]:


X


# In[34]:


y


# ##### **Train-Test Split** 

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ##### **Train XGBoost Regressor** 

# In[37]:


model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    random_state=42
)
model.fit(X_train, y_train)


# ##### **Predictions** 

# In[39]:


y_pred = model.predict(X_test)


# ##### **Evaluation** 

# In[41]:


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")


# ##### **Feature importance** 

# In[43]:


import matplotlib.pyplot as plt
xgb.plot_importance(model, importance_type="gain")
plt.show()


# ##### **Random Forest Regressor** 

# In[45]:


#importing Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ##### **Handle categorical variables** 

# In[57]:


encoder = LabelEncoder()
categorical_cols = ['Item Fat Content', 'Item Type', 'Outlet Identifier', 
                    'Outlet Size', 'Outlet Location Type', 'Outlet Type']

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])


# ##### **Features and Target** 

# In[58]:


X = df.drop("Sales", axis=1)   # Independent vars
y = df["Sales"]                # Target var


# ##### **Train-test split** 

# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[60]:


rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)


# ##### **Predictions** 

# In[65]:


y_pred = rf_model.predict(X_test)


# ##### **Evaluation** 

# In[66]:


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Random Forest Results:")
print("RMSE:", rmse)
print("R² Score:", r2)


# In[ ]:





# In[70]:


import matplotlib.pyplot as plt
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title="Top 10 Important Features - Random Forest"
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




