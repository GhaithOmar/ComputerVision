import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow_batchNorm import ANN 
from scipy.io import loadmat
from sklearn.utils import shuffle


def error_rate(p,y):
	return np.mean(p != y)

def score(p, y):
	return np.mean(p == y)

def relu(a):
	return a * (a>0)

def flatten(X):
	N = X.shape[-1]
	flat = np.zeros((N, 3072))
	# 3072 is 32*32*3
	for i in range(N):
		flat[i] = X[:,:,:,i].reshape(3072)
	return flat

def get_data():
	train = loadmat('..\\lg_files\\train_32x32.mat')
	test = loadmat('..\\lg_files\\train_32x32.mat')
	# print(train['X'].shape)
	Xtrain = flatten(train['X'] / 255)
	Ytrain = train['y'].flatten() - 1 
	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

	Xtest = flatten(test['X'] / 255)
	Ytest = test['y'].flatten() - 1

	return Xtrain, Xtest, Ytrain, Ytest



if __name__ == '__main__':
	Xtrain, Xtest,Ytrain, Ytest = get_data()
	model = ANN([1000,500])
	model.fit(Xtrain,Ytrain,Xtest, Ytest, batch_sz=500, epochs=20, show_fig=True, lr=0.1)

