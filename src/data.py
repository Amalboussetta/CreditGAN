
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
#%%
'''test data for later'''      
def data_test():
      df2 = pd.read_csv('test.csv', delimiter=';')
      df2.dataframeName = 'test.csv'
      df2 =df2.drop(['ID','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis = 1)
      scaler = preprocessing.MinMaxScaler()
      # with open('scalar_model.pkl', 'wb') as f:

      # pickle.dump(scalar, f)
      names = df2.columns
      d1 = scaler.fit_transform(df2)

      scaled_df2 = pd.DataFrame(d1, columns=names)
      #return scaled_df.head()
      df2 = scaled_df2.to_numpy()
      
      
      
      #df3 = pd.DataFrame(df2,index=None, columns=["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","def_pay"])
      return df2
# %%
