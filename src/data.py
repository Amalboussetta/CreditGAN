
#%%
#from mpl_toolkits.mplot3d import Axes3D
import pickle
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
# import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
#import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas import read_csv
import seaborn


#%%

def data():
      df1 = pd.read_csv('UCI_Credit_Card.csv', delimiter=',')
      df1.dataframeName = 'UCI_Credit_Card.csv'
      
      creditdata = df1.copy()
      
      #creditdata= creditdata['EDUCATION'].drop(axis=1,index = ['5','6'])
      creditdata =creditdata.drop(['ID','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis = 1)
      #creditdata =creditdata.drop(['ID','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis = 1)#'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'],axis=1)
      
      creditdata.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)
      
      
      
      
      
      
      
      
      #real_data = creditdata.loc[(creditdata['def_pay']==1)]
      #real_data.to_csv('real_data.csv')
      creditdata.to_csv('real_df.csv')
      return creditdata #, real_data.head()









#%%

def load_d():
      

      
      creditdata = data()
      creditdata_for_scaler = creditdata.drop(['def_pay'],axis =1)
      scaler_g = preprocessing.MinMaxScaler()
      scaler_g.fit(creditdata_for_scaler)
      with open('model/scaler_g_model.pkl', 'wb') as f:
            pickle.dump(scaler_g, f)
      creditdata_for_scaler.to_csv('real_df.csv')

      #df = preprocessing.normalize(creditdata)
      scaler = preprocessing.MinMaxScaler()
      # with open('scalar_model.pkl', 'wb') as f:

      # pickle.dump(scalar, f)
      names = creditdata.columns
      d = scaler.fit_transform(creditdata)

      scaled_df = pd.DataFrame(d, columns=names)
      #return scaled_df.head()
      df = scaled_df.to_numpy()
      
      return df
#      
# %%
# #creditdata_for_scaler.to_csv('real_df.csv')
      
#       #df = preprocessing.normalize(creditdata)
#       scaler = preprocessing.MinMaxScaler()
#       # with open('scalar_model.pkl', 'wb') as f:

#       # pickle.dump(scalar, f)
#       names = creditdata.columns
#       d = scaler.fit_transform(creditdata)

#       scaled_df = pd.DataFrame(d, columns=names)
#       #return scaled_df.head()
#       df = scaled_df.to_numpy()
#       #df_X = df[:,-1:]
#       #df_y = df[:,-1]
#       #trainx,trainy = train_test_split(df,test_size=0.2)
#       #train_test_split()
#       #creditdata.to_numpy()
#       #creditdata[creditdata.columns] = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(creditdata[creditdata.columns])
#       #newdata = creditdata.drop(['ID','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis = 1)
#       df = creditdata.to_numpy()
#       #return train_df,test_df #, trainx.shape,trainy.shape,testx.shape,testy.shape #, 