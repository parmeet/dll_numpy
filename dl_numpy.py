import numpy as np

class  Function():
    def forward(self): 
        pass
    
    def backward(self): 
        pass
    
    def getParams(self):
        return []

class Linear(Function):
    def __init__(self,input,output):
        self.weights = {'data':np.ndarray((input,output),np.float32),\
                        'grad':np.ndarray((input,output),np.float32)}

        self.bias    = {'data':np.ndarray((output,),np.float32),\
                        'grad':np.ndarray((output,),np.float32)}

        self.type = 'linear'

    def forward(self,x):
        output = np.dot(x,self.weights['data'])+self.bias['data']
        self.input = x 
        return output

    def backward(self,d_x):
        self.weights['grad'] += np.dot(self.input.T,d_x)
        self.bias['grad']    += d_x
        grad_input            = np.dot(d_x,self.weights['data'].T)
        return grad_input

    def getParams(self):
        return [self.weights,self.bias]

class Softmax(Function):
    def __init__(self):
        self.type = 'normalization'

    def forward(self,x):
        unnormalized_proba      = np.exp(x-np.max(x))
        self.normalized_proba   = unnormalized_proba/np.sum(unnormalized_proba)
    
        return self.normalized_proba

    def backward(self,d_x):
        return d_x*self.normalized_proba*(1-self.normalized_proba) 

class  ReLU(Function):
    def __init__(self):
        self.type = 'activation'
    
    def forward(self,x):
        self.activated = x>0
        return x*self.activated

    def backward(self,d_x):
        return d_x*self.activated

class Tanh(Function):
    def __init__(self):
        self.type = 'activation'
    
    def forward(self,x):
        self.activation = np.tanh(x)
        return self.activation

    def backward(self,d_x):
        return d_x*(1.-self.activation**2)

class NLLLoss(Function):
    def __init__(self):
        self.type = 'loss'

    def forward(self,prediction,target):
        self.prediction  = prediction.flatten()
        self.target      = target
        loss             = -np.log(self.prediction[self.target])

        return loss

    def backward(self,scale = 1.0):
        d_x = np.zeros_like(self.prediction)
        d_x[self.target] = -scale*1.0/(self.prediction[self.target]+1.e-10)
        return d_x

class Optimier():
    def step(self):
        pass

    def zeroGrad(self):
        pass

class SGD(Optimier):
    def __init__(self,parameters,lr=.001):
        self.parameters = parameters
        self.lr         = lr

    def step(self):
        for p in self.parameters:
            p['data']+=-self.lr*p['grad']

    def zeroGrad(self):
        for p in self.parameters:
            p['grad'] = 0.0

##############################Library Implementation **END##############################