# example of loading the classifier model and generating images
#%%
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.models import load_model
from src.data import load_data 
import pickle
#%%
# load the model
def classifer_accurancy():
      
      dataset = load_data()
      X,y = dataset[:,:-1],dataset[:,-1]
      trainx,testx,trainy,testy= train_test_split(X,y,test_size=0.2)
      model = load_model('c_model_4500.h5')
    
    # evaluate the model
      _, train_acc = model.evaluate(trainx, trainy, verbose=0)
      print('Train Accuracy: %.3f%%' % (train_acc * 100))
      _, test_acc = model.evaluate(testx, testy, verbose=0)
      print('Test Accuracy: %.3f%%' % (test_acc * 100))

#       from sklearn.metrics import confusion_matrix
# from matplotlib import pyplot as plt

# conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print('Confusion matrix:\n', conf_mat)

# labels = ['Class 0', 'Class 1']
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
# fig.colorbar(cax)
# ax.set_xticklabels([''] + labels)
# ax.set_yticklabels([''] + labels)
# plt.xlabel('Predicted')
# plt.ylabel('Expected')
# plt.show()
# %%
from math import sqrt
import numpy as np 
import pandas as pd
from numpy import asarray
from numpy.random import randn
from keras.models import load_model
import matplotlib.pyplot as plt
 
#%%
  # generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_class):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	#labels = asarray([n_class for _ in range(n_samples)])
	return z_input
#%%
def load_real_samples(dataset,n_samples):
      
      dataset = load_data()

      df_X = dataset[:,:-1]
      df_y = dataset[:,-1]
      selected_ix = df_y==1
      X = df_X[selected_ix]
      #y = df_y[selected_ix]      
      # choose random instances
      return X #,y]
#%%
def generated_data(latent_dim,n_examples,n_class):
      
    model = load_model('../model_0120.h5')
    # sneaker
    # generate images
    latent_points = generate_latent_points(latent_dim, n_examples, n_class)
    # generate images
    X  = model.predict(latent_points)
    with open('scalar_model.pkl', 'rb') as f:
           scalar = pickle.load(f)
    df = pd.DataFrame (data=X,index = None, columns = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"])
    
    with open('../scaler_g_model.pkl', 'rb') as f:
        scaler_g = pickle.load(f)
    X_transf = scaler_g.inverse_transform(df)
    plt.hist(pd.DataFrame(X_transf)[0])
    plt.show()


    #df = pd.DataFrame(data=X,index = None,column = ["ID","LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","default.payment.next.month"])
    return #df.head
# create and save a plot of generated images
#def save_plot(examples, n_examples):
	# plot images
	# for i in range(n_examples):
	# 	# define subplot
	# 	plt.subplot(sqrt(n_examples), sqrt(n_examples), 1 + i)
	# 	# turn off axis
	# 	plt.axis('off')
	# 	# plot raw pixel data
	# 	plt.imshow(examples[i, :, :, 0], cmap='gray_r')
	# plt.show()
#%%
def compare():
    
    dataset = load_data()
    model = load_model('model_6000.h5')
    latent_dim = 24
     # must be a square
    n_class = 1 # default
    # generate images
    latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
    # generate images
    X  = model.predict(latent_points)

    real_data = load_real_samples(dataset,100)
    plt.scatter(X,real_data)
    #gen_samples = pd.DataFrame(X,columns= data + labels)
    #plt.show()
    return X, real_data 
# scale from [-1,1] to [0,1]
#X = (X + 1) / 2.0
# plot the result
#save_plot(X, n_examples)
# %%
