#%%

# semi-supervised learning

from h5py._hl.dataset import Dataset
import numpy as np
import argparse
import tensorflow as tf
from sklearn.model_selection import  train_test_split
from src.semisupmodel import generator_network, discriminator_network, gan_model
from src.data import load_d 
from numpy import full
from numpy import asarray
from numpy.random import randn , normal
from numpy.random import randint 
import matplotlib.pyplot as plt 
tf.get_logger().setLevel('ERROR')

# from matplotlib import pyplot
#%%


# def load_data():
#     dataset = load_d()
#     X,y = dataset[:,:-1],dataset[:,-1]
#     trainx,_,trainy,_= train_test_split(X,y,test_size=0.2)
#     #print(trainx.shape,trainy.shape)
#     return trainx,trainy


#%%
'''This part is for the supervised learning. Here we select a number of samples with labels. These labels are
kept and will trained by supervised discriminator '''
def select_supervised_samples(dataset,n_samples,n_classes=2):
        
        X,y = dataset[:,:-1],dataset[:,-1]
        trainx,_,trainy,_= train_test_split(X,y,test_size=0.2)

        trainx_list , trainy_list = list(), list()
        n_per_class= int(n_samples/n_classes)
        for i in range(n_classes):
                X_with_class = trainx[trainy==i]
                ix = np.random.randint(0,len(X_with_class),n_per_class)
                [trainx_list.append(X_with_class[j]) for j in ix]
                [trainy_list.append(i)for j in ix]
        
        return asarray(trainx_list), asarray(trainy_list)
#%%
'''A sample of data and labels is selected. 
This same function can be used to retrieve examples from the 
labeled and unlabeled dataset, later when we train the models. 
In the case of the “unlabeled dataset“, we will ignore the labels.
This part is for generating the real data and see if the discriminator would be 
able to recognize it
'''
def generate_real_samples(dataset, n_samples):  

        # split into data and labels
    dataset = load_d()

    df_X,df_y = dataset[:,:-1],dataset[:,-1]
    

    # choose random instances
    idx = np.random.randint(0, df_X.shape[0], n_samples)
    X = df_X[idx]
    labels = df_y[idx]
    # select data and labels
    
    #labels = df_y.iloc[ix]
    # generate class labels
    
   # y = ones((n_samples, ))
    y = np.full([n_samples,1],0.9)
    #y = np.full([n_samples,1],0.1)
    return [X, labels ], y
# 	#print(X.shape)
# 	#print(labels.shape)


#%%
''''generate points from the latent space and use the generator to generate some fake data
unction will call this function to generate a batch worth of data
that can be fed to the unsupervised discriminator model
 or the composite GAN model during training.'''

def generate_latent_points(latent_dim, n_samples,n_classes):
        # generate points in the latent space
    x_input = np.random.normal(0,1,latent_dim*n_samples)
    #x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    _ = randint(0, n_classes, n_samples)
    return [z_input,_]



#%%

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input ,_= generate_latent_points(latent_dim, n_samples,n_classes)
    # predict outputs
    model = generator_network(latent_dim,units)
    data = model.predict(z_input)
    # create class labels
    #y = np.full([n_samples,1],0.9)
    y = np.full([n_samples,1],0.1)
    return data, y

#%%

def summarize_performance(step, latent_dim, n_samples=500):

    # save the generator model
    filename1 = 'model_%04d.h5' % (step+1)
    g_model.save(filename1)
    X = dataset[:,:-1]
    y = dataset[:,-1]
    _, acc = c_model.evaluate(X, y, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    #save the classifier model
    filename2 = 'c_model_%04d.h5' % (step+1)
    c_model.save(filename2)
    print('>Saved:  %s and %s' % ( filename1,filename2))
#def summarize_performance(step,g_model,latent_dim,n_samples):
# create a line plot of loss for the gan and save to file

def plot_history(d1_hist,d2_hist,g_hist,c_hist,c1_hist):
    plt.subplot(2,1,1)
    plt.plot(d1_hist,label='dis_real')
    plt.plot(d2_hist,label ='dis_fake')
    plt.plot(g_hist,label ='gen')
    plt.legend()

    #plot discriminator accuracy
    plt.subplot(2,1,2)
    plt.plot(c_hist,label = 'accuracy')
    plt.plot(c1_hist, label = 'classifier_loss')
    plt.legend()

    plt.savefig('results_convergence/plot_line_plot_loss.png')
    plt.close()

#%%
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=2048):
    X_sup,y_sup = select_supervised_samples(dataset,n_samples)
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    d1_hist,d2_hist, g_hist, c_hist,c1_hist = list(), list(), list(),list(),list()
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected 'real' samples
        [X_real, y_real], _= generate_real_samples([X_sup,y_sup], half_batch)
        # # update discriminator model weights
        c_loss,acc = c_model.train_on_batch(X_real,y_real)
        #update unsupervised dicriminator d
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        # # generate 'fake' examples
        X_fake,y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_loss2 = d_model.train_on_batch(X_fake,y_fake)
        # prepare points in latent space as input for the generator
        
        # create inverted labels for the fake samples
        [X_gan,_], y_gan = generate_latent_points(latent_dim, n_batch,n_classes), np.full([n_batch,1],0.9)
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        #record history
        d1_hist.append(d_loss1)
        d2_hist.append(d_loss2)
        g_hist.append(g_loss)
        c1_hist.append(c_loss)
        c_hist.append(acc)
        #record history

        # update the generator via the discriminator's error
       
        # summarize loss on this batch
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, acc*100, d_loss1, d_loss2, g_loss))

        #evaluate the model performance every 'epoch'
        if (i+1) % (bat_per_epo * 100) == 0:
            summarize_performance(i,g_model,latent_dim)
    plot_history(d1_hist,d2_hist,g_hist,c_hist,c1_hist)



n_samples = 1000
latent_dim = 6#23#24#17
data_dim = 6#23#24#17#11#24
n_classes = 2
units = 128


d_model,c_model = discriminator_network(data_dim,units)
# create the generator
g_model = generator_network(latent_dim,units)
# create the gan
gan_model = gan_model(g_model, d_model)
# load image data
dataset = load_d()
#dataset.shape[0] = size
# train model
train(g_model, d_model,c_model, gan_model, dataset, latent_dim)





