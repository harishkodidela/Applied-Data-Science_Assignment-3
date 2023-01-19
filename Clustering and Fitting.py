# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:07:36 2023

@author: dheer
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skmet
import sklearn.cluster as cluster
import numpy as np

#working with datasets
def handling_data(path):
    ''' generating two dataset with country and year as columns
    parameters: Path- location of the dataset.
    '''
    df = pd.read_csv(path)
    df=df.drop(columns=['Indicator Code','Country Code'])
    df1 = df.transpose()
    df1.columns = df1.iloc[0].values.tolist()
    df1=df1.iloc[1:]
    return df,df1

def norm(array):
    """ Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe"""
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df):

    """
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    """
    # iterate over all numerical columns
    for col in df.columns[0:]: # excluding the first column
        df[col] = norm(df[col])
    return df

def Clusters_data(df):
    for i in range(2,7):
        kmeans = cluster.KMeans(n_clusters = i)
        kmeans.fit(df)
        labels = kmeans.labels_
        print(i, skmet.silhouette_score(df, labels))

def scatter_plot(df, n, col1, col2):
    kmeans = cluster.KMeans(n_clusters=n)
    kmeans.fit(df)
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(6.0, 6.0))
    
    # Individual colours can be assigned to symbols. The label l is used to the
    # l-th number from the colour table.
    plt.scatter(df[col1], df[col2], c=labels, cmap="Accent")
    # colour map Accent selected to increase contrast between colours
    # show cluster centres
    for ic in range(n):
        xc, yc = cen[ic,:]
        plt.plot(xc, yc, "dk", markersize=10)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'{n} clusters for {col1} vs {col2}')
    plt.show()
    plt.savefig("Cluster_Plot", dpi=720)
    

country,year=handling_data('Data World Bank_Climate change data.csv')
columns=year.iloc[0].unique()
countries=['India','Pakistan','China','Japan']
country_select=year[countries]

col = ['Agriculture,forestry and fishing','Energy use', 
       'School Enrollment','co2 emissions','Urban Population','Forest area']
#correlation of the data
country=year[['China']]
data_list=[]
indicators_index=[0,10,11,49,67,44]
for i in indicators_index:
    data_list.append(country.iloc[:,i])
d= pd.DataFrame(data_list)
d=d.transpose()
d.columns=col
d=d[31:60]
d=d[1:]
d=d.fillna(d.median())

ax = sns.clustermap(d.corr(), annot=True)
plt.title("China indicators correlation")
plt.savefig("clustermap.png",dpi=720)
plt.show()

pd.plotting.scatter_matrix(d, figsize=(9.0,9.0))
plt.tight_layout()
plt.savefig("scatter_matrix.png",dpi=720)
plt.show()

d_fit = d[['co2 emissions', 'Energy use']].copy()
d_fit = norm_df(d_fit)

        
Clusters_data(d_fit)
scatter_plot(d_fit, 3, 'co2 emissions', 'Energy use')  
scatter_plot(d_fit, 4, 'co2 emissions', 'Energy use')       


d_fitting = d[['Agriculture,forestry and fishing', 'Forest area']].copy()
d_fitting = norm_df(d_fitting)

        
Clusters_data(d_fitting)
scatter_plot(d_fitting, 3, 'Agriculture,forestry and fishing', 'Forest area')  
scatter_plot(d_fitting, 4, 'Agriculture,forestry and fishing', 'Forest area')       

