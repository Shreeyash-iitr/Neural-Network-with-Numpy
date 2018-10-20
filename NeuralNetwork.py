#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:25:37 2018

@author: shreeyash
"""
import numpy as np

#=========  define neural network architecture ==============
def NN_architecture(depth = 4,width = 5,input_dim = 3, output_dim = 1,output_act = 'sigmoid',hidden_act='relu'):
    # for first layer
    nn_architecture = [{"input_dim":input_dim,"output_dim":width,"activation":hidden_act}]
    # for hidden layers
    for layer in range(depth-2):
        temp = {"input_dim":width,"output_dim":width,"activation":hidden_act}
        nn_architecture.append(temp)
    # for last layer    
    nn_architecture.append({"input_dim":width,"output_dim":output_dim,"activation":output_act}) 
    return nn_architecture  

#=========  initialize parameters  ==========================
def init_parameters(nn_architecture, init_with = 'glorot'):
    parameters = {}
    if init_with == 'random':
        for index, layer in enumerate(nn_architecture):
            parameters['W'+str(index+1)] = np.random.randn(layer['output_dim'],layer['input_dim'])*0.1 
            parameters['b'+str(index+1)] = np.random.randn(layer['output_dim'],1)*0.1        
    
    if init_with == 'glorot':
        for index, layer in enumerate(nn_architecture):
            std = np.sqrt(2/(layer['output_dim']+layer['input_dim']))
            parameters['W'+str(index+1)] = np.random.normal(0,std,(layer['output_dim'],layer['input_dim']))
            parameters['b'+str(index+1)] = np.zeros((layer['output_dim'],1))
        
    return parameters    
        
    
#===========  Activation functions =============================
    
def sigmoid(z):
    return 1/(1+np.exp(-z))

def relu(z):
    return np.maximum(0,z)

def tanh(z):
    return np.tanh(z)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))
 
#============= backward activations derivatives ================================

def sigmoid_backward(da_prev, z):
    sigma = sigmoid(z)
    return sigma*(1-sigma)*da_prev

def relu_backward(da_prev, z):
    dZ = np.array(da_prev, copy = True)
    dZ[z<0] = 0
    return dZ
    
def tanh_backward(da_prev, z):
    return (1-np.square(tanh(z)))*da_prev

def softmax_backward(da_prev, z):
    return (softmax(z)-np.square(softmax(z)))*da_prev



#=========== Forward single layer propagation ================================
    
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation = 'relu'):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    if activation == 'relu':
        A_curr = relu(Z_curr)
    elif activation == 'sigmoid':
        A_curr = sigmoid(Z_curr)
    elif activation == 'tanh':
        A_curr = tanh(Z_curr)
    elif activation == 'softmax':
        A_curr = softmax(Z_curr)    
    else:
        raise Exception("softmax/sigmoid/relu/tanh")
    return A_curr, Z_curr        

#=============== full forward propagation ==================================

def full_forward_propagation(nn_architecture, parameters, inputs):
    A_curr = inputs
    bp_memory = {'A0':inputs}
    
    for index, layer in enumerate(nn_architecture):
        A_prev = A_curr
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, parameters['W'+str(index+1)],parameters['b'+str(index+1)],layer['activation'])
        bp_memory['Z'+str(index+1)] = Z_curr
        bp_memory['A'+str(index+1)] = A_curr
    y_hat = A_curr
    return y_hat, bp_memory   


#============== loss function ===============================================
    
def loss(y_hat, y_actual):
    if y_actual.shape == y_hat.shape:
        return np.squeeze(-np.sum((y_actual*np.log(np.abs(y_hat)+0.001))+((1-y_actual)*np.log(np.abs(1-y_hat)+0.001)))/y_actual.shape[1])
    else:
        raise Exception('y_hat and y_actual have different shapes')


#================= Accuracy =================================================
        
def accuracy(y_hat, y_actual):
    y_hat[y_hat>0.5] = 1
    y_hat[y_hat<0.5] = 0
    y_hat[y_hat==y_actual] = 1
    y_hat[y_hat!=y_actual] = 0
    return np.mean(y_hat)



#================ Backpropagation in single layer ============================
    
def single_layer_backpropagation(dA_curr, Z_curr, W_curr, b_curr, A_pre, activation = 'relu'):
    if activation == 'relu':
        dZ_curr = relu_backward(dA_curr,Z_curr)
    elif activation == 'sigmoid':
        dZ_curr = sigmoid_backward(dA_curr,Z_curr)
    elif activation == 'tanh':
        dZ_curr = tanh_backward(dA_curr,Z_curr)
    elif activation == 'softmax':
        dZ_curr = softmax_backward(dA_curr,Z_curr) 
    else:
        raise Exception('invalid activation derivative')
    dW_curr = np.dot(dZ_curr,A_pre.T)/A_pre.shape[1]
    db_curr = np.sum(dZ_curr,axis=1,keepdims=True)/A_pre.shape[1]
    dA_pre = np.dot(W_curr.T, dZ_curr)
    
    return dA_pre,dW_curr, db_curr
    
#================= full backpropagation =====================================
    
def backpropagation(nn_architecture,parameters,y_hat,y_actual, memory):
    if(y_hat.shape != y_actual.shape):
        y_actual = np.reshape(y_actual,y_hat.shape)
        
    gradients = {}    
    dA_curr = -(np.divide(y_actual,y_hat+0.001)-np.divide(1-y_actual,1-y_hat+0.001))
    
    for index, layer in reversed(list(enumerate(nn_architecture))): 
        Z_curr = memory['Z'+str(index+1)]
        W_curr = parameters['W'+str(index+1)]
        b_curr = parameters['b'+str(index+1)]
        A_pre = memory['A'+str(index)]
        activation = layer['activation']
        dA_pre, dW_curr, db_curr = single_layer_backpropagation(dA_curr,Z_curr,W_curr, b_curr, A_pre,activation)
        gradients['dW'+str(index+1)] = dW_curr
        gradients['db'+str(index+1)] = db_curr
        dA_curr = dA_pre
    return gradients

#====================== Update parameters ==================================

def update_parameters(parameters,gradients, nn_architecture, lr=0.01):
    for index, layer in enumerate(nn_architecture):
        parameters['W'+str(index+1)] -= lr*gradients['dW'+str(index+1)]
        parameters['b'+str(index+1)] -= lr*gradients['db'+str(index+1)]
    return parameters    
    
        

#====================  TRAINING  ===========================================
    
def train(X,Y,learning_rate,epochs,nn_architecture):
    cost_history = []
    acc_history = []
    parameters = init_parameters(nn_architecture,init_with='random')
    for epoch in range(epochs):
        y_hat, memory = full_forward_propagation(nn_architecture, parameters, inputs=X)
        cost_history.append(loss(y_hat, Y))
        acc_history.append(accuracy(y_hat, Y))
        gradients = backpropagation(nn_architecture, parameters, y_hat, Y, memory)
        parameters = update_parameters(parameters, gradients, nn_architecture, learning_rate)
    return parameters, cost_history, acc_history

#===================  INFERENCE ===========================================
    
def predict(X, parameters, nn_architechture):
    Y,memory = full_forward_propagation(nn_architechture,parameters,X)
    return Y
        




