#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[6]:


df=pd.read_csv('/home/guest/Downloads/MallCustomers.csv')
df.head()


# In[7]:


#universe analysis


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


plt.plot(df['Annual Income (k$)'])
plt.show()


# In[10]:


data=np.array(df['Age'])
plt.plot(data,linestyle = 'dotted')


# In[11]:


sns.countplot(df['Age'])


# In[12]:


df['Annual Income (k$)'].plot(kind='density')


# In[13]:


sns.countplot(df['Gender'])


# In[14]:


sns.boxplot(df['Annual Income (k$)'])


# In[15]:


plt.hist(df['Annual Income (k$)'])


# In[16]:


#Bivariate Analysis


# In[17]:


sns.stripplot(x=df['Age'],y=df['Annual Income (k$)'])


# In[18]:


sns.stripplot(x=df['Age'],y=df['Spending Score (1-100)'])


# In[19]:


plt.scatter(df['Age'],df['Annual Income (k$)'],color='blue')
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")


# In[20]:


sns.violinplot(x='Age',y='Spending Score (1-100)',data=df)


# In[21]:


#Multivariate Analysis


# In[22]:


sns.pairplot(df)


# In[23]:


#Descriptive Statistics


# In[24]:


sns.heatmap(df.corr(),annot=True)


# In[25]:


df.shape


# In[26]:


df.isnull().sum()


# In[27]:


df.info()


# In[28]:


df.describe()


# In[29]:


df.mean()


# In[30]:


df.median()


# In[31]:


df.mode()


# In[32]:


df['Gender'].value_counts()


# In[33]:


#check of missing values


# In[34]:


df.isna().sum()


# In[35]:


#Heading outliers


# In[36]:


sns.boxplot(df['Annual Income (k$)'])


# In[37]:


Q1 = df['Annual Income (k$)'].quantile(0.25)
Q3 = df['Annual Income (k$)'].quantile(0.75)
IQR = Q3 - Q1
whisker_width = 1.5
lower_whisker = Q1 -(whisker_width*IQR)
upper_whisker = Q3 +(whisker_width*IQR)
df['Annual Income (k$)']=np.where(df['Annual Income (k$)']>upper_whisker,upper_whisker,np.where(df['Annual Income (k$)']<lower_whisker,lower_whisker,df['Annual Income (k$)']))


# In[38]:


sns.boxplot(df['Annual Income (k$)'])


# In[39]:


#Encoding Categorical Values


# In[40]:


numeric_data = df.select_dtypes(include=[np.number]) 
categorical_data = df.select_dtypes(exclude=[np.number]) 
print("Number of numerical variables: ", numeric_data.shape[1]) 
print("Number of categorical variables: ", categorical_data.shape[1])


# In[41]:


print("Number of categorical variables: ", categorical_data.shape[1]) 
Categorical_variables = list(categorical_data.columns)
Categorical_variables


# In[42]:


df['Gender'].value_counts()


# In[43]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = le.fit_transform(df['Gender'])
df["Gender"] = label


# In[44]:


df['Gender'].value_counts()


# In[45]:


#Scaling the data


# In[46]:


X = df.drop("Age",axis=1)
Y = df['Age']


# In[47]:


from sklearn.preprocessing import StandardScaler
object= StandardScaler()
scale = object.fit_transform(X) 
print(scale)


# In[48]:


X_scaled  = pd.DataFrame(scale, columns = X.columns)
X_scaled


# In[49]:


#train test split
from sklearn.model_selection import train_test_split
# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=0)


# In[50]:


X_train.shape


# In[51]:


X_test.shape


# In[52]:


Y_train.shape


# In[53]:


Y_test.shape


# In[54]:


#Clustering Algorithm


# In[55]:


x = df.iloc[:, [3, 4]].values 


# In[56]:


#finding optimal number of clusters using the elbow method  
from sklearn.cluster import KMeans  
wcss_list= []  #Initializing the list for the values of WCSS  
  
#Using for loop for iterations from 1 to 10.  
for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)  
    kmeans.fit(x)  
    wcss_list.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss_list)  
plt.title('The Elobw Method Graph')  
plt.xlabel('Number of clusters(k)')  
plt.ylabel('wcss_list')  
plt.show()  


# In[57]:


#training the K-means model on a dataset  
kmeans = KMeans(n_clusters=5, init='k-means++', random_state= 42)  
y_predict= kmeans.fit_predict(x) 


# In[58]:


#visulaizing the clusters  
plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster  
plt.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  
plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
plt.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   
plt.title('Clusters of customers')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend()  
plt.show()  


# In[59]:


#the End


# In[ ]:




