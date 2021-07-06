#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense , LeakyReLU, Dropout , Input , concatenate, Activation, Embedding,BatchNormalization
from tensorflow.keras.optimizers import Adam
#%%
#### declaration of some demensions
units = 128
a = 0.1
data_dim = 6 #data dimension
latent_dim = 6
# good best 
opt_d = Adam(lr=0.0002, beta_1=0.5)
opt_c = Adam(lr=0.0002,beta_1=0.5)
opt_g = Adam(lr=0.00001, beta_1=0.5)
#trial 1 4600_1 good
# opt_d = Adam(lr=0.0003, beta_1=0.5)
# opt_c = Adam(lr=0.0003,beta_1=0.5)
# opt_g = Adam(lr=0.00001, beta_1=0.5)
# # 4600 similar and good
# opt_d = Adam(lr=0.0004, beta_1=0.5)
# opt_c = Adam(lr=0.0004,beta_1=0.5)
# opt_g = Adam(lr=0.00001, beta_1=0.5)

#try a higher rate
#n_classes= 2 #binary classification

#%%
''''Generator network'''
def generator_network(latent_dim,units): 

    
    in_lat = Input(shape=(latent_dim,))
    x = Dense(units*1)(in_lat)
    
    #x = BatchNormalization()(x) # 1
    x = LeakyReLU(alpha=a)(x)
    #
    #x = Dropout(rate=0.4)(x)
    x = Dense(units*2)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=a)(x)
    #x = Dropout(rate=0.4)(x) # 2
    x = Dense(units*4)(x)
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=a)(x)
    #x = Dropout(rate=0.4)(x)
    x = Dense(data_dim)(x)
    
    
    #x = Dense(units*4,activation='relu')(x)

    g_output = Activation('tanh')(x)
    g_model = Model(inputs=[in_lat],outputs=[g_output])
    #g_model.compile(optimizer=opt,loss = 'binary_crossentropy')   
    return g_model


#%%
'''Discriminator network with two outputs'''
def discriminator_network(data_dim,units):
    input_d= Input(shape=(data_dim,))  #(data_dim+label_dim,))
    x =Dense(units*4)(input_d)
    
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=a)(x)
    x = Dense(units*2)(x)
    #x = BatchNormalization()(x)
    #x = Dropout(rate=0.5)(x)
    x = LeakyReLU(alpha=a)(x)
    #x = Dropout(rate=0.5)(x)
    x = Dense(units*1)(x) 
    #x = BatchNormalization()(x)
    x = LeakyReLU(alpha=a)(x)
    #x = Dropout(rate=0.5)(x)
     #this dicri model has two outputs one for the supervised and one for unsupervised 
    #classifier
    out2 = Dense(1,activation='sigmoid')(x)
    c_model = Model(input_d, out2)
    c_model.compile(loss='binary_crossentropy', optimizer=opt_c,metrics=['accuracy'])

    #fake or not fake
    out1 = Dense(1,activation='sigmoid')(x)
    d_model = Model(input_d,out1)
    d_model.compile(loss='binary_crossentropy', optimizer=opt_d)
    return d_model, c_model
	# compile model
#%%  
'''CreditGAN'''  
def gan_model(g_model, d_model):
    for layer in d_model.layers :
        if not isinstance(layer,BatchNormalization):
            layer.trainable = False

    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    model.compile(loss='binary_crossentropy', optimizer=opt_g)
    return model 
# %%

# %%
