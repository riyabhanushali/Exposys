# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:14:30 2020

@author: Riya Katariya
"""


#importing the libraies required for data preprocessing and visualization
import os                           # to change the working directory
import pandas as pd                 # to work with dataframes
import numpy as np                  # to perform mathematical functions
import matplotlib
import matplotlib.pyplot as plt     # extension of numpy used to plot framework
import seaborn as sns               # for statical data visualization



#version of the libraries
print(pd.__version__)           #version of pandas used is 0.25.3 
print(np.__version__)           #version of numpy used is 1.17.4
print(matplotlib.__version__)   #version of matplotlib used is 3.1.1
print(sns.__version__)          #version of seaborn used is 0.9.0

#importing the dataset
data_customer=pd.read_csv("Mall_Customers.csv")

#making the copy of the original dataset
data = data_customer.copy()

#analyzing the dataset
data.size             # gives the total size of the dataset(no. of rows * no. of columns)
data.shape            # gives the dimension of the dataset
data.index            # gives the row label of the dataframe
data.columns          # list all the column name from the dataframe
data.head(10)         # returns the first 10 rows 
data.tail(10)         # returns the last 10 rows
data.dtypes           # returns the datatype of each column
data.info()           # gives the summary of the dataframe
data.isnull().sum()   # returns the counts the null values present in the dataframe 
data.describe()       # calculate statical data


############################################################################################
# now we will visualize various columns of the dataframe to check which benfit most for vendor

# plotting barplot to check distribution of male and female in gender column
genders = data.Gender.value_counts()
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index, y=genders.values)
plt.show()
#hence from plot we can estimates that more number of the female visit mall as comapred to male

# plotting the boxplot of spending score and annual income
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.boxplot(y=data["Spending Score (1-100)"])
plt.subplot(1,2,2)
sns.boxplot(y=data["Annual Income (k$)"])
plt.show()
#from this we can clearly see that the customers spending score is more than his/her annual income

data['Age'].min()
data['Age'].max()
# here min age of the customer is 18 years whereas max age is 70 years
# we will code to see the which age grp of the customers contributes more to the sales in the mall
age18_25 = data.Age[(data.Age <= 25) & (data.Age >= 18)]
age26_35 = data.Age[(data.Age <= 35) & (data.Age >= 26)]
age36_45 = data.Age[(data.Age <= 45) & (data.Age >= 36)]
age46_55 = data.Age[(data.Age <= 55) & (data .Age >= 46)]
age55above = data.Age[data.Age >= 56]

x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y)
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()
# from this barplot we can analyze that the customers between the age grp (26-35) visit more to the mall

#similar to age we can compute for spending score and annual income of the customer

# for Spending Score
data['Spending Score (1-100)'].min()
data['Spending Score (1-100)'].max()

s_score1_20 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 1) & (data["Spending Score (1-100)"] <= 20)]
s_score21_40 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 21) & (data["Spending Score (1-100)"] <= 40)]
s_score41_60 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 41) & (data["Spending Score (1-100)"] <= 60)]
s_score61_80 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 61) & (data["Spending Score (1-100)"] <= 80)]
s_score81_100 = data["Spending Score (1-100)"][(data["Spending Score (1-100)"] >= 81) & (data["Spending Score (1-100)"] <= 100)]
s_scorex = ["1-20", "21-40", "41-60", "61-80", "81-100"]
s_scorey = [len(s_score1_20.values), len(s_score21_40.values), len(s_score41_60.values), len(s_score61_80.values), len(s_score81_100.values)]

plt.figure(figsize=(14,5))
sns.barplot(x=s_scorex, y=s_scorey)
plt.title("Spending Scores")
plt.show()
# from this we can compute that maximum customer has their spending score between 41-60

# for annual income
data['Annual Income (k$)'].min()
data['Annual Income (k$)'].max()
# here the minimum income of the customer is 15k$ whereas maximum is 137k$

a_income0_30 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 0) & (data["Annual Income (k$)"] <= 30)]
a_income31_60 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 31) & (data["Annual Income (k$)"] <= 60)]
a_income61_90 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 61) & (data["Annual Income (k$)"] <= 90)]
a_income91_120 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 91) & (data["Annual Income (k$)"] <= 120)]
a_income121_150 = data["Annual Income (k$)"][(data["Annual Income (k$)"] >= 121) & (data["Annual Income (k$)"] <= 150)]

a_incomex = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
a_incomey = [len(a_income0_30.values), len(a_income31_60.values), len(a_income61_90.values), len(a_income91_120.values), len(a_income121_150.values)]

plt.figure(figsize=(14,5))
sns.barplot(x=a_incomex, y=a_incomey)
plt.title("Annual Incomes")
plt.show()
# from this graph we realize that most of the customers have their annual income between the range of 60k$ to 90k$

#ploting the graph for Spending score vs Annual Incomes
plt.figure(figsize=(14,5))
sns.barplot(x=a_incomey, y=s_scorey)
plt.title("Spending score vs Annual Incomes")
plt.xlabel("Annual Incomes")
plt.ylabel("Spending score")
plt.show()
# from the resulatant graph we can understand that as the income of the cuatomer increases its spending score also increases

#########################################################################################
# we will consider the main two factors i.e Spending score and Annual income of the customer as the affects the most for customer segmentation
ss_ai= data.iloc[:, [3,4]].values
#Kmeans algorithm to decide the optimum cluster number
from sklearn.cluster import KMeans
wcss = []
#let us assume the max number of cluster would be 10

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(ss_ai)
    wcss.append(kmeans.inertia_)       #inertia_ is used to segregate the Visualizing the elbow method to get optimal value of k
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11), wcss,linewidth=2, color="red", marker ="8")

plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()
# From the graph we can notice that the last elbow comes at k=5 hence we can build 5 clusters for our dataframe

#########################################################################################
#Model Building
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
model= kmeansmodel.fit_predict(ss_ai) 

#Visualizing the cluster 
plt.scatter(ss_ai[model == 0, 0], ss_ai[model == 0, 1], s = 100, c = 'yellow', label = 'Cluster 1')
plt.scatter(ss_ai[model == 1, 0], ss_ai[model == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(ss_ai[model == 2, 0], ss_ai[model == 2, 1], s = 100, c = 'brown', label = 'Cluster 3')
plt.scatter(ss_ai[model == 3, 0], ss_ai[model == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(ss_ai[model == 4, 0], ss_ai[model == 4, 1], s = 100, c = 'grey', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# from this plot we can conclude the following result
# There are total 5 clusters in which the customers can bhi segmented
# cluster1 specifies the group of customers having balance between their spending score and annual income
# cluster2 specifies group of customers having more spending score as compared to their annual income
# cluster3 specifies the group of customers having spending score is high along with their annual income
# cluster4 specifies the group of customers having spending score is low along with their annual income
# cluster5 specifies the group of customers having their spending score much low as compared to their annual income



