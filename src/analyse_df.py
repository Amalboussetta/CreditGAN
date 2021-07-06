
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
real_data = pd.read_csv('real_df.csv')
generated_df = pd.read_csv('good generated data.csv')

#plt.hist(real_data['LIMIT_BAL'])
#plt.hist(generated_df['LIMIT_BAL'])
#generated_df.head()
#%%
'''Compare between original and generated data'''
plt.hist(real_data['EDUCATION'])
plt.hist(generated_df['EDUCATION'])
plt.show()

plt.hist(real_data['MARRIAGE'])
plt.hist(generated_df['MARRIAGE'])
plt.show()

plt.hist(real_data['AGE'])
plt.hist(generated_df['AGE'])
plt.show()

plt.hist(real_data['SEX'])
plt.hist(generated_df['SEX'])
plt.show()

plt.hist(real_data['PAY_0'])
plt.hist(generated_df['PAY_0'])
plt.show()

plt.hist(real_data['LIMIT_BAL'])
plt.hist(generated_df['LIMIT_BAL'])
plt.show()



#%%
'''plot generated data'''


plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(generated_df.MARRIAGE)

#%%
plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(generated_df.AGE)


plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(generated_df.MARRIAGE)



plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(generated_df.SEX)



plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(generated_df.EDUCATION)


plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(generated_df.PAY_0)
plt.show()


# use seaborn
# %%
