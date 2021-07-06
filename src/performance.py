# example of loading the classifier model and generating images
#%%
import pickle
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras.models import load_model
from data import load_d, data_test
#from model import scaler_g_model
from math import sqrt
import numpy as np 
import pandas as pd
from numpy import asarray, around
from numpy.random import randn
from keras.models import load_model
#import matplotlib.pyplot as plt
#%%
'''Accuracy on the train and test dataset'''
# load the model
def classifer_accurancy():
      dataset = load_d()
      #trainx,testx,trainy,testy = load_d()
      X,y = dataset[:,:-1],dataset[:,-1]
      #trainx,testx,trainy,testy= train_test_split(X,y,test_size=0.2)
      model = load_model('supervised_classifier.h5')
    
    # evaluate the model
      _, train_acc = model.evaluate(X, y, verbose=0)
      print('Train Accuracy: %.3f%%' % (train_acc * 100))
#%%
def classifier_accuracy_test():
      test_data = data_test()
      test_X,test_y = test_data[:,:-1],test_data[:,-1]
      model = load_model('supervised_classifier.h5')
      _, test_acc = model.evaluate(test_X, test_y, verbose=0)
      
      print('Test Accuracy: %.3f%%' % (test_acc * 100))

 
'''generate fake data for plotting and testing '''
#%%
  # generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	#labels = asarray([n_class for _ in range(n_samples)])
	return z_input#,labels

#%%
def generated_data(latent_dim,n_examples):
      
    model = load_model('unsupervised_model.h5')
    # sneaker
    # generate images
    latent_points = generate_latent_points(latent_dim, n_examples)
    # generate images
    X  = model.predict(latent_points)
    with open('../model/scaler_g_model.pkl', 'rb') as f:

      scaler_g = pickle.load(f)
    
    #df = pd.DataFrame (data=X,index = None, columns = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0"])#,"PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"])
    
    
    X_transf = scaler_g.inverse_transform(X)
    Y= np.around(X_transf)
    df = pd.DataFrame(data=Y,index = None, columns = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0"])
    df.to_csv('generated_df.csv')
    return df
    
# %%
