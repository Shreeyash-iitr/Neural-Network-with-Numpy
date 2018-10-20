# -*- coding: utf-8 -*-

import numpy as np
from NeuralNetwork import *
import matplotlib.pyplot as plt
x = []
y = []
for i in range(100):
    x.append(np.random.normal(0,2,(3,1)))
    y.append(np.full((1,1),-2))
    x.append(np.random.normal(3.5,2,(3,1)))
    y.append(np.full((1,1),2))
 
x = np.array(x)
y = np.array(y)    

x = np.squeeze(x).T
y = np.reshape(np.squeeze(y),(1,200))
plt.plot(x.T,y.T)
nn_architecture = NN_architecture()
parameters,cost, acc = train(x,y,0.001,500,nn_architecture)
print(np.mean(np.array(acc)))
t = np.full((3,200),8)
predict(t,parameters,nn_architecture)