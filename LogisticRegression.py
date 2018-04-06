import numpy as np
from scipy import optimize

def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g

def costFunction(theta,X,y):
    m = len(y)
    z = np.dot(X, theta)
    hx = sigmoid(z)
    cost = -(np.dot(np.transpose(y), np.log(hx)) + np.dot(np.transpose(1-y), np.log(1-hx)))/m
    return cost

def gradient(theta,X,y):
    m = len(y)
    z = np.dot(X, theta)
    hx = sigmoid(z)
    grad = np.array((np.dot(np.transpose(hx - y), X)) / m)
    return grad

#Initialize X and y

with open('ExamMarks.txt') as f:
    contents = f.readlines()
contents = [x.strip() for x in contents]
X1 = [i.split(',',2)[0] for i in contents]
X2 = [i.split(',',2)[1] for i in contents]
y = [i.split(',',2)[2] for i in contents]
X1 = np.array([float(i) for i in X1]).reshape(len(X1),1)
X2 = np.array([float(i) for i in X2]).reshape(len(X2),1)
y = np.array([float(i) for i in y]).reshape(len(y),1)
X = np.concatenate((X1,X2),axis=1)

#Compute Cost and Gradient

Xn = np.concatenate((np.ones((len(y),1)),X),axis=1)
n = np.shape(Xn)
initial_theta = np.zeros((n[1], 1))
cost = costFunction(initial_theta ,Xn, y)
grad = gradient(initial_theta ,Xn, y)
print('Cost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): ')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
#Test Theta
t = [[-24] ,[0.2] ,[0.2]]
cost = costFunction(t, Xn, y)
grad = gradient(t, Xn, y)
print('\nCost at test Theta: ', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test Theta : ',grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

#Optimization

result = optimize.fmin(costFunction, x0=initial_theta, args=(Xn, y), maxiter=400, full_output=True)
print(result)