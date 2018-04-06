import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def featureNormalize(X):
    X_norm = X
    mu = np.zeros(len(X))
    sigma = np.zeros(len(X))
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    size = np.shape(X)
    for i in range(size[1]):
        for j in range(size[0]):
            X_norm[j][i] = (X[j][i] - mu[i])/sigma[i]
    return X_norm,mu,sigma

def gradDescent(X,y,theta,alpha, iterations):
    m = len(y)
    for i in range(iterations):
        hx = np.dot(X, theta)
        beta = np.dot(np.transpose(X), (hx - y))
        theta = theta - (alpha * beta) / m
    return theta

with open('HouseData.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]
X1 = [i.split(',', 2)[0] for i in content]
X2 = [i.split(',', 2)[1] for i in content]
y = [i.split(',', 2)[2] for i in content]
y = np.array([float(i) for i in y]).reshape(len(y),1)
X1 = np.array([float(i) for i in X1]).reshape(len(X1),1)
X2 = np.array([float(i) for i in X2]).reshape(len(X2),1)
X = np.concatenate((X1,X2),axis=1)

#Running feature Normalization

print('Normalizing Features.......................\n')
X,mu,sigma = featureNormalize(X)
o = np.ones((len(y),1))
Xn = np.concatenate((o,X),axis=1)

#Running Gradient Descent

print('Running Gradient Descent......')
alpha = 0.01
iterations = 400
n = np.shape(Xn)
theta = np.zeros((n[1],1))
theta = gradDescent(Xn,y,theta,alpha,iterations)
print('Theta computed from gradient descent: \n')
print('\n', theta)
case = np.array([1,1650,3])
mu = np.concatenate((np.zeros(1),mu),axis=0)
sigma = np.concatenate((np.ones(1),sigma),axis=0)
total = np.zeros(len(case))
for i in range(len(case)):
    total[i] = (case[i]-mu[i])/sigma[i]
price = np.dot(total,theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n', price[0])