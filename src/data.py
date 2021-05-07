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
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import recall_score
#from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
#from imblearn import over_sampling
#from imblearn.over_sampling import RandomOverSampler
#import warnings
#warnings.filterwarnings('ignore')

#%%
# def load_data():
      
#       scaler = preprocessing.MinMaxScaler()
#       df1 = pd.read_csv('UCI_Credit_Card.csv', delimiter=',')
#       df1.dataframeName = 'UCI_Credit_Card.csv'

#       # print(f'There are {nRow} rows and {nCol} columns')
#       ### shape of the data
#       creditdata = df1.copy()
#       #creditdata = creditdata.drop(['ID'],axis =1)
#       scaler = preprocessing.MinMaxScaler()
#       names = creditdata.columns
#       d = scaler.fit_transform(creditdata)
#       scaled_df = pd.DataFrame(d, columns=names)
#       #return scaled_df.head()
#       df = scaled_df.to_numpy()
#       #df_X = df[:,:24]
#       #df_y = df [:,-1]
#       #trainx, trainy,testx,testy = train_test_split(df_X,df_y,test_size=0.2)
#       # #X = df_X []
#       return df 
      #return trainx,trainy,testx,testy

#load_data()
      #creditdata =creditdata.drop(['ID','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis = 0)
      #creditdata.drop(['ID','LIMIT_BAL',"SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","default.payment.next.month"],axis=0)
      #creditdata.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)
      
#%%
      
# def upsamle(dataset) :

#       X,y = dataset#[:,:-1],dataset[:,-1]
#       #y = dataset[:,-1]
#       X,y = make_classification(n_samples=10000, n_features=25, n_informative=2,
#                                  n_redundant=0, n_repeated=0, n_classes=2,
#                                 )
      
#       ros = RandomOverSampler(random_state=0)
#       X_resampled, y_resampled = ros.fit_resample(dataset)
#       return X_resampled,y_resampled

#%%

# def load_data():
    


      
      
#       df1 = pd.read_csv('UCI_Credit_Card.csv', delimiter=',')
#       df1.dataframeName = 'UCI_Credit_Card.csv'
#       #nRow, nCol = df1.shape
#       # print(f'There are {nRow} rows and {nCol} columns')
#       ### shape of the data
#       creditdata = df1.copy()
#       # X_resampled,y_resampled = upsamle(creditdata)
#       # data = X_resampled,y_resampled
#       creditdata = creditdata.drop(['ID'],axis =1)
#       creditdata = preprocessing.normalize(creditdata)
#       return creditdata #, creditdata.shape







def load_data():
      
      df1 = pd.read_csv('UCI_Credit_Card.csv', delimiter=',')
      df1.dataframeName = 'UCI_Credit_Card.csv'
      #nRow, nCol = df1.shape
      # print(f'There are {nRow} rows and {nCol} columns')
      ### shape of the data
      creditdata = df1.copy()
      # X_resampled,y_resampled = upsamle(creditdata)
      # data = X_resampled,y_resampled
      creditdata = creditdata.drop(['ID'],axis =1)
      #creditdata= creditdata['EDUCATION'].drop(axis=1,index = ['5','6'])
      #creditdata =creditdata.drop(['ID','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis = 1)
      #creditdata =creditdata.drop(['ID','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis = 1)#'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'],axis=1)
      creditdata.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)
       #return creditdata, creditdata.shape
      
      creditdata_for_scaler = creditdata.drop(['def_pay'],axis =1)
      scaler_g = preprocessing.MinMaxScaler()
      scaler_g.fit(creditdata_for_scaler)
      with open('model/scaler_g_model.pkl', 'wb') as f:
            pickle.dump(scaler_g, f)


      #df = preprocessing.normalize(creditdata)
      scaler = preprocessing.MinMaxScaler()
      names = creditdata.columns
      d = scaler.fit_transform(creditdata)

      scaled_df = pd.DataFrame(d, columns=names)
      #return scaled_df.head()
      df = scaled_df.to_numpy()
      #creditdata.to_numpy()
      #creditdata[creditdata.columns] = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(creditdata[creditdata.columns])
      #newdata = creditdata.drop(['ID','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis = 1)
      #df = creditdata.to_numpy()
      return df #, df.shape
#       return df
#       #creditdata.to_numpy()
#       #creditdata[creditdata.columns] = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(creditdata[creditdata.columns])
#       #newdata = creditdata.drop(['ID','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis = 1)
      
      
#       # df_X = creditdata.drop(['def_pay'], axis=1).to_numpy()
#       # df_y = creditdata.def_pay.to_numpy()

#       # #X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=10)
      
#       # print(creditdata.shape)
#       # data_dim = df_X.shape[1]
#       # print(data_dim)
#       # label_dim = df_y.ndim
#       # print(label_dim)
#       # #label_dim = df_y.shape[1]
#       # return df_X, df_y
#     # creditdata[creditdata.columns] = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(creditdata[creditdata.columns])

#   # print(creditdata)
# #### make the data more gaussian probably
# #%%
# # df1 = pd.read_csv('UCI_Credit_Card.csv', delimiter=',')
# # df1.dataframeName = 'UCI_Credit_Card.csv'
# # nRow, nCol = df1.shape
# # print(f'There are {nRow} rows and {nCol} columns')
# # ### shape of the data
# # creditdata = df1.copy()
# # print(creditdata.shape)
# #creditdata.head()

# # creditdata.describe().T
# #%%
# def load_data2():
    
      
      
#       df1 = pd.read_csv('heart.csv', delimiter=',')
#       df1.dataframeName = 'heart.csv'
#       #nRow, nCol = df1.shape
#       # print(f'There are {nRow} rows and {nCol} columns')
#       ### shape of the data
#       heartdata = df1.copy()
#       heartdata =heartdata.drop(['restecg','thalachh'], axis =1 )
#       #heartdata =creditdata.drop(['ID','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'],axis=1)
#       #creditdata.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)
#       scaler = preprocessing.MinMaxScaler()
#       names = heartdata.columns
#       d = scaler.fit_transform(heartdata)
#       scaled_df = pd.DataFrame(d, columns=names)
#       #return scaled_df.head()
#       df = scaled_df.to_numpy()
#       ##creditdata.to_numpy()
#       #creditdata[creditdata.columns] = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(creditdata[creditdata.columns])
#       #newdata = creditdata.drop(['ID','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis = 1)
#       #df = creditdata.to_numpy()
#       #df = heartdata
#       return df 
# # ####check datatype
# #def plot_data()
# # creditdata.info()
# # ''' no missing value hence no need for imputation '''

# # creditdata.shape

# # # %%

# # ### how mny default cases do I have => is the data fit dor modelling?
# # creditdata.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)
# # def_cnt = (creditdata.def_pay.value_counts(normalize=True)*100)
# # def_cnt.plot.bar(figsize=(6,6))
# # plt.xticks(fontsize=12, rotation=0)
# # plt.yticks(fontsize=12)
# # plt.title("Probability Of Default", fontsize=15)
# # for x,y in zip([0,1],def_cnt):
# #     plt.text(x,y,y,fontsize=12)
# # plt.show()
# # creditdata =creditdata.drop(['ID','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'], axis = 1)
# # creditdata.shape
# # ''' We can see that the dataset consists of 77% 
# # clients are not expected to default payment whereas 23% 
# # clients are expected to default the payment.'''
# # #%%
# # creditdata.shape
# # #%%
# # df_X = creditdata.drop(['def_pay'], axis=1)
# # df_y = creditdata.def_pay

# # X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=10)
# # # creditdata[creditdata.columns] = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(creditdata[creditdata.columns])
# print( X_train,X_test, y_train,y_test)
# # print(creditdata)

# X_train.shape
# #

# #%%
# X_train

# %%


    

# %%
