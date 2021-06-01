
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%%
real_data = pd.read_csv('real_df.csv')
generated_df = pd.read_csv('generated_df.csv')
#generated_df.head()

plt.hist(real_data['AGE'])
plt.hist(generated_df['AGE'])
plt.show()

#%%

plt.subplots(figsize=(20,5))
plt.subplot(121)
sns.distplot(generated_df.LIMIT_BAL)


plt.subplots(figsize=(20,5))
plt.subplot(122)
sns.distplot(generated_df.AGE)


''''''

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


# plt.hist(real_data[real_data['AGE'])
# plt.hist(generated_df[generated_df['AGE'])
# plt.show()

# use seaborn
# %%
