#%%
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# %%
df1 = pd.read_csv('UCI_Credit_Card.csv', delimiter=',')
df1.dataframeName = 'UCI_Credit_Card.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
# %%
defaulters = df1.copy()
print(defaulters.shape)
defaulters.head()
#defaulters = d[['SEX','MARRIAGE','AGE','BILL_AMT1','EDUCATION','PAY_0','def_pay']]

# %%
def_cnt = (defaulters.def_pay.value_counts(normalize=True)*100)
def_cnt.plot.bar(figsize=(6,6))
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)
plt.title("Probability Of Defaulting Payment Next Month", fontsize=15)
for x,y in zip([0,1],def_cnt):
    plt.text(x,y,y,fontsize=12)
plt.show()
# %%

plt.subplots(figsize=(20,5))
plt.subplot(121)
sns.distplot(defaulters.LIMIT_BAL)

#%%
plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(defaulters.AGE)


''''''

plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(defaulters.MARRIAGE)



plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(defaulters.SEX)



plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(defaulters.EDUCATION)


plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(defaulters.PAY_0)
plt.show()


# %%
