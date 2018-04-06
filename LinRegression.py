import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt



def computeCost(X,y,theta):
    hx = np.dot(X,theta)
    m = len(y)
    t = hx - y
    J = np.dot(np.transpose(t),t)/(m*2)
    return J

def gradDescent(X,y,theta,alpha, iterations):
    m = len(y)
    for i in range(1500):
        hx = np.dot(X, theta)
        beta = np.dot(np.transpose(X), (hx - y))
        theta = theta - (alpha * beta) / m
    return theta

#Initialize X and y

with open('Data.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]
X=[i.split(',', 1)[0] for i in content]
y=[i.split(',', 1)[-1] for i in content]
y = np.array([float(i) for i in y]).reshape(len(y),1)
X = np.array([float(i) for i in X]).reshape(len(X),1)

#Plot graph between X and y

print('Plotting data.......')
plt.plot(X,y,'rx')
plt.xlabel('Population of City on 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()
plt.close()
print('Press Enter to continue........')
input()

#Calculate the cost function

print('\nTesting the cost function ...\n')
X1 = np.concatenate((np.ones((len(X),1)),X),axis = 1)
theta = np.zeros((2,1))
iterations = 50
alpha = 0.01
print('\nTesting the cost function ...\n')
J = computeCost(X1,y,theta)
print('With theta = [0 ; 0]\nCost computed = ', J)
print('Expected cost value (approx) 32.07\n')
J = computeCost(X1, y, [[-1] ,[2]])
print('\nWith theta = [-1 ; 2]\nCost computed = ', J)
print('Expected cost value (approx) 54.24\n')

#gradient descent

print('\nRunning Gradient Descent\n')
theta = gradDescent(X1,y,theta,alpha,iterations)
print('Theta found by gradient descent:\n')
print(theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')
plt.plot(X,y,'rx',label='Training Data')
plt.plot(X,np.dot(X1,theta),'b',label='Linear Regression')
plt.legend(loc='upper right')
plt.show()

#NormakmEquations

#theta = np.dot(np.dot(inv(np.dot(np.transpose(X1),X1)),np.transpose(X1)),y)