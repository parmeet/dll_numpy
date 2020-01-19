import dl_numpy as DL
import utilities
import numpy as np
import sklearn
from sklearn import datasets

if __name__=="__main__":
    samples_per_class   = 100
    num_classes         = 3
    data,target  = utilities.genSpiralData(samples_per_class,num_classes)
    model = utilities.Model()
    model.add(DL.Linear(2,100))
    model.add(DL.ReLU())
    model.add(DL.Linear(100,3))
    optim = DL.SGD(model.parameters,lr=1,weight_decay=0.001,momentum=.9)
    loss_fn = DL.SoftmaxWithLoss()
    model.fit(data,target,300,6000,optim,loss_fn)
    predicted_labels = np.argmax(model.predict(data),axis=1)
    accuracy = np.sum(predicted_labels==target)/len(target)
    print("Model Accuracy = {}".format(accuracy))
    utilities.plot2DDataWithDecisionBoundary(data,target,model)


