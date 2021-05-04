#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense , LeakyReLU, Dropout , Input , concatenate, Activation, Embedding,BatchNormalization
from tensorflow.keras.optimizers import Adam
from src.data import load_data
#import keras.backend as K

#%%
load_data()
# def get_f1(y_true, y_pred): #taken from old keras source code
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     recall = true_positives / (possible_positives + K.epsilon())
#     f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
#     return f1_val

#%%
#### declaration of some demensions
units = 128
latent_dim = 23#24#17#24 # equal to the data dimension
data_dim = 23#24 #17#11#24 
label_dim = 1 # class column with o or 1 output
opt_d = Adam(lr=0.0004, beta_1=0.5)

opt_c = Adam(lr=0.0001,beta_1=0.5)
opt_g = Adam(lr=0.0001, beta_1=0.5)
#n_classes= 2 #binary classification

#%%
#genrator with labels to try if this model works. the other alternative would be normal 
#generator
# def generator_network(latent_dim,label_dim,units,n_classes): 

#     labels = Input(shape=(1,))
#     #li = Embedding(n_classes,1)(labels)
#     in_lat = Input(shape=(latent_dim,))
#     x = Dense(units*1)(in_lat) # 1
#     x = LeakyReLU(alpha=0.1)(x)
#     x = Dense(units*2)(x)
#     x = LeakyReLU(alpha=0.1)(x) # 2
#     x = Dense(units*4)(x)
#     x = LeakyReLU(alpha=0.1)(x)
#     x = Dense(data_dim)(x)
#     x = concatenate([x,labels])
    
#     #x = Dense(units*4,activation='relu')(x)

#     g_output = Activation('tanh')(x)
#     g_model = Model(inputs=[in_lat,labels],outputs=[g_output])
#     #g_model.compile(optimizer=opt,loss = 'binary_crossentropy')   
#     return g_model
def generator_network(latent_dim,label_dim,units): 

    
    in_lat = Input(shape=(latent_dim,))
    x = Dense(units*1)(in_lat)
    x = BatchNormalization()(x) # 1
    x = LeakyReLU(alpha=0.1)(x)
    #
    #x = Dropout(rate=0.4)(x)
    x = Dense(units*2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    #x = Dropout(rate=0.4)(x) # 2
    x = Dense(units*4)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    #x = Dropout(rate=0.4)(x)
    x = Dense(data_dim)(x)
    
    
    #x = Dense(units*4,activation='relu')(x)

    g_output = Activation('tanh')(x)
    g_model = Model(inputs=[in_lat],outputs=[g_output])
    #g_model.compile(optimizer=opt,loss = 'binary_crossentropy')   
    return g_model


#%%
def discriminator_network(data_dim,units):
    input_d= Input(shape=(data_dim,))  #(data_dim+label_dim,))
    x =Dense(units*4)(input_d)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(units*2)(x)
    x = BatchNormalization()(x)
    #x = Dropout(rate=0.5)(x)
    x = LeakyReLU(alpha=0.1)(x)
   # x = Dropout(rate=0.5)(x)
    x = Dense(units*1)(x) 
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
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
