# GAN-for-credit-default
File Organization:

CreditGAN.py: The model architecture

Data.py: Data loading, preprocessing, training, testing

Performance.py: see the results of the supervised and the 
unsupervised task

Analyse.py: plot the results of the unsupervised tasks

Vizualization.py: to see and plot the original data.

File model: contains 2 unsupervised model (a good one and a bad one), supervised classifier and the scaler model used for preprocessing and scaling the data.

File plots: contains two plots of a good and a bad CreditGAN

Csv Files:

real_df.csv: real data 

bad generated data.csv: data generated from a mode collapse model

good generated data.csv: data generated from a successful model


test.csv: testing data 20%
UCI_credit_card.csv: training data 80%

Original data.csv: the original dataset (not used in the code, only used the training and testing seperately)

To run the code install the virtual environment and run python main.py
