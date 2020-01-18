import dl_numpy as DL
from utilities import *
import numpy as np
import sklearn
from sklearn import datasets

learning_rate = .001
batch_size    = 20
num_samples   = 500
num_epochs    = 5

linear_classifier = [DL.Linear(2,3)]
innitializeNetwork(linear_classifier)
all_params = []
for f in linear_classifier:all_params+=f.getParams()
optim   = DL.SGD(all_params,learning_rate)
loss_fn = DL.SoftmaxWithLoss()

data,target  = sklearn.datasets.make_blobs(num_samples)
data_gen     = DataGenerator(data,target,batch_size)
Train(linear_classifier,optim,data_gen,loss_fn,num_epochs)
