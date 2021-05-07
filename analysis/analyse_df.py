
#%%
import pandas as pd
import matplotlib.pyplot as plt

real_df = pd.read_csv('real_df.csv')
generated_df = pd.read_csv('generated_df.csv')

plt.hist(real_df[real_df['def_pay']['AGE']])
plt.hist(generated_df[generated_df['def_pay']['AGE']])
plt.show()

# use seaborn