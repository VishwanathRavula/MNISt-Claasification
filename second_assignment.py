#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:01:43 2019

@author: vish
"""
import numpy as np
import keras
from keras.datasets import mnist
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

npa = np.array
# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax2(X, t = 1.0):
    exps = np.exp(X)
    print('Ex: ',exps)
    return exps / np.sum(exps)
#    e = np.exp(npa(X) / t)
#    dist = e / np.sum(e)
#    return dist

def relu(X):
   return np.maximum(X, 0)

def feed_forward(model,X):
#    hi
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    print('x shape',X.dot(W1).shape)
    Z1 = X.dot(W1) + b1;
    a1 = sigmoid_v(Z1)
#    print(a1);
#    print('buhj',a1.dot(W2).shape)
    Z2 = a1.dot(W2) + b2
#    ztemp = np.asarray(z2)
    out = softmax(Z2);
    print('Out shape: ',out.shape)
#    print(out[0])
    return Z1, a1, Z2, out

def calculate_Loss(model,X,Y):
    num_examples = X.shape[0]
    error = 0;
    act_error = Y
    a1, y1, a2, out = feed_forward(model, X)
    for i in range(3000):
        t  = Y[i]
        te = out[i]
        err = (t - te)*(t - te)*(1/2)
        error = error + err
        act_error[i] = error
        error = 0
#    mse = np.square(Y - out)
#    print(Y - out)
    loss = np.sum(act_error)
    return loss/num_examples

def backpropagation(X,Y,model,Z1,a1,Z2,out):
#    print(X.shape)
#    print(Y.shape)
#    print(X.shape)
#    print(X.shape)
    delta3 = (Y-out).dot((out.T).dot(1-out))
    dW2 = (a1.T).dot(delta3)
    
    db2 = np.sum(delta3, axis=0)
    
    delta2 = delta3.dot((model['W2'].T).dot((a1.T).dot(1-a1)))
    
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)

    return dW1, dW2, db1, db2
    
def train(model,X,Y,learning_rate):
    number_epoch=10
    previous_loss = float('inf')
    losses = []
    for i in range(number_epoch):
        
        loss = calculate_Loss(model,X,Y)
        losses.append(loss)
        if(previous_loss-loss) < 0.01*previous_loss:
            break;
            
        print('epoch number', i,'loss is ', loss);
        Z1,a1,Z2,out = feed_forward(model,X);
        dW1, dW2, db1, db2 = backpropagation(X,Y,model,Z1,a1,Z2,out)
        model['W1'] -= learning_rate * dW1
        model['b1'] -= learning_rate * db1
        model['W2'] -= learning_rate * dW2
        model['b2'] -= learning_rate * db2
        previous_loss=loss
    
    return model, losses
        
    
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000,784));
y_train = y_train.reshape((60000,1));
y_train = to_categorical(y_train)
training_size = 3000;
trX = x_train[:training_size]
trY = y_train[:training_size]
W1 = np.random.rand(784,20);
b1 = np.random.rand(1,20);
W2 = np.random.rand(20,10);
b2 = np.random.rand(1,10);
print(trX.shape)
print(trY.shape)
model = {}
model['W1'] = W1;
model['b1'] = b1;
model['W2'] = W2;
model['b2'] = b2;
learning_rate=0.4;

model,losses = train(model,trX, trY, 5);

model['W1'] = model['W1'].T;
from PIL import Image
import matplotlib.pyplot as plt
print(model['W1'].shape)
arr_img = model['W1'].reshape((20,28,28))[4];
#img = Image.fromarray(arr_img, 'RGB')
print('image shape: ',arr_img.shape)
#arr_img = arr_img;
plt.gray()
plt.imshow(arr_img)
