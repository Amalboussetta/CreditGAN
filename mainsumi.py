#%%

# semi-supervised learning

import numpy as np
import tensorflow as tf
from sklearn import preprocessing, model_selection
from sklearn.model_selection import  train_test_split
from src.semisupmodel import generator_network, discriminator_network, gan_model
from src.data import load_data #, load_data2
from numpy import zeros,full
from numpy import ones
from numpy import asarray
from numpy.random import randn , normal
from numpy.random import randint 
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')

# from matplotlib import pyplot
#%%


######## remove the first line 
#power transform

#%%
'''This part is for the supervised learning. Here we select a number of samples with labels. These labels are
kept and will trained by supervised discriminator '''
def select_supervised_samples(dataset,n_samples,n_classes=2):
        dataset = load_data()
        X,y = dataset[:,:-1],dataset[:,-1]
        trainx,_,trainy,_= train_test_split(X,y,test_size=0.2)
        
        #X = dataset [:,:24]

        #y = dataset[:,-1]

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
    dataset = load_data()

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
    model = generator_network(latent_dim,label_dim,units)
    data = model.predict(z_input)
    # create class labels
    y = np.full([n_samples,1],0.1)
    return data, y

#%%

def summarize_performance(step, latent_dim, n_samples=500):
#         # prepare fake examples
#     #[X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
#     # scale from [-1,1] to [0,1]
#     #X = (X + 1) / 2.0
#     # # plot 
#     # for i in range(100):
#     #     # define subplot
#     #     plt.subplot(10, 10, 1 + i)
#     #     # turn off axis
#     #     plt.axis('off')
#     #     # plot raw pixel data
#     #     plt.imshow(X[i, :, :, 0], cmap='gray_r')
#     # # save plot to file
#     # filename1 = 'generated_plot_%04d.png' % (step+1)
#     # plt.savefig(filename1)
#     # plt.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step+1)
    g_model.save(filename2)
    X = dataset[:,:-1]
    y = dataset[:,-1]
    _, acc = c_model.evaluate(X, y, verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    #save the classifier model
    filename3 = 'c_model_%04d.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved:  %s and %s' % ( filename3,filename2))
#def summarize_performance(step,g_model,latent_dim,n_samples):
      
#)
#%%
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=500):
    X_sup,y_sup = select_supervised_samples(dataset,n_samples)
    # calculate the number of batches per training epoch
    bat_per_epo = int(30000 / n_batch)
    
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
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

        # update the generator via the discriminator's error
       
        # summarize loss on this batch
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, acc*100, d_loss1, d_loss2, g_loss))

        #evaluate the model performance every 'epoch'
        if (i+1) % (bat_per_epo * 25) == 0:

                
            summarize_performance(i,g_model,latent_dim)

n_samples = 1000
latent_dim = 23#24#17
data_dim = 23#24#17#11#24
label_dim = 1
n_classes = 2
units = 128

d_model,c_model = discriminator_network(data_dim,units)
# create the generator
g_model = generator_network(latent_dim,n_classes,units)
# create the gan
gan_model = gan_model(g_model, d_model)
# load image data
dataset = load_data()


#dataset.shape[0] = size
# train model
train(g_model, d_model,c_model, gan_model, dataset, latent_dim)
#summarize_performance(100,g_model='gene')
#summarize_performance(100,g_model='gene')
#%%
#%%
#%%
# load the data

#%%
# do i even need this? what to do with my data
# def load_real_samples():
# 	# load dataset
# 	(trainX, trainy), (_, _) = load_data()
# 	# expand to 3d, e.g. add channels
# 	X = expand_dims(trainX, axis=-1)
# 	# convert from ints to floats
# 	X = X.astype('float32')
# 	# scale from [0,255] to [-1,1]
# 	X = (X - 127.5) / 127.5
# 	print(X.shape, trainy.shape)
# 	return [X, trainy]
#%%
# select a supervised subset of the dataset, ensures classes are balanced
''' We can also define a function to select a subset of the training dataset in which we keep the labels and train the supervised version of the discriminator model.

The select_supervised_samples() function
 below implements this and is careful to ensure 
 that the selection of examples is random and that
  the classes are balanced. The number of labeled 
  examples is parameterized and set at 100, meaning 
  that each of the 10 classes will have 10 randomly 
  selected examples.'''
#if i use 10 samples the generator is not doing great