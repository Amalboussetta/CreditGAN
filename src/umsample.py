#%%
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
# import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
#import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler


#%%
def upsamle() :
    df1 = pd.read_csv('UCI_Credit_Card.csv', delimiter=',')
    df1.dataframeName = 'UCI_Credit_Card.csv'
    dataset = df1.copy()
    X,y = dataset.iloc[:,:-1],dataset.iloc[:,-1]
#       #y = dataset[:,-1]
    X,y = make_classification(n_samples=30000, n_features=24, n_informative=1,
                                 n_redundant=0, n_repeated=0, n_classes=2,
                                )
      
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X,y)
    return Counter(X_resampled),Counter(y_resampled)

    #return X.head , y.head
#       
# %%
