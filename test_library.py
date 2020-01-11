import dl_numpy as DL
import numpy as np

def Train(net,optim,data_gen_class,loss_fn,num_epochs,batch_size):
    loss_history = []
    for e in range(num_epochs):
        data_gen = data_gen_class(batch_size = batch_size)
        for b,(batch_data,batch_target) in enumerate(data_gen):
            optim.zeroGrad()
            batch_loss = 0
            for d,t in zip(batch_data,batch_target): 
                for f in net: d=f.forward(d)
                batch_loss += loss_fn.forward(d,t)
                grad = loss_fn.backward()
                #print(grad)
                for f in net[::-1]:grad = f.backward(grad) 
            loss_history+=[batch_loss/len(batch_data)]
            print("Loss at epoch = {} and iteration = {}: {}".format(e,b,loss_history[-1]))
            optim.step()


def innitializeNetwork(net):
    for f in net:
        if f.type=='linear':
            weights,bias    = f.getParams()
            weights['data'] = np.random.randn(weights['data'].shape[0],weights['data'].shape[1])/np.sqrt(weights['data'].shape[0])
            bias['data']    = 0.


####Test Library on IRIS Data######
import sklearn
from sklearn import datasets

def BlobDataGen(num_samples = 1000,batch_size = 10):
    data,target  = sklearn.datasets.make_blobs(num_samples)
    shuffled     = np.random.permutation(len(data))
    batch_data   = []
    batch_target = []
    for b,(x,y) in enumerate(zip(data[shuffled],target[shuffled])):
        batch_data  +=[np.array(x).reshape(1,-1)]
        batch_target+=[y]
        if ((b+1)%batch_size==0 or b==len(data)-1) and len(batch_data)>0:
            yield batch_data,batch_target
            batch_data   = []
            batch_target = []


learning_rate = .0001
batch_size = 10
num_epochs = 5

linear_classifier = [DL.Linear(2,3),DL.Softmax()]
innitializeNetwork(linear_classifier)
all_params = []
for f in linear_classifier:all_params+=f.getParams()
optim = DL.SGD(all_params,learning_rate)
loss_fn = DL.NLLLoss()
Train(linear_classifier,optim,BlobDataGen,loss_fn,num_epochs,batch_size)


