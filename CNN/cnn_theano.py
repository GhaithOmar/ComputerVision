import numpy as np 
import theano 
import theano.tensor as T 
import matplotlib.pyplot as plt 

from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool 

from scipy.io import loadmat 
from sklearn.utils import shuffle

from datetime import datetime

def score(p,Y):
	return np.mean(p == Y)


def relu(a):
	return a * (a>0)

def convpool(X, W, b, poolsize=(2,2)):
	conv_out = conv2d(input=X, filters=W)
	pooled_out = pool.pool_2d(
		input= conv_out,
		ws= poolsize,
		ignore_border=True
	)
	return relu(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))



def rearrange(X):
	N = X.shape[-1]
	out = np.zeros((N, 3, 32, 32), dtype=np.float32)

	for i in range(N):
		for j in  range(3):
			out[i, j, :, :] = X[:,:,j,i]

	return out/255


def init_filter(shape, poolsz):
	w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:]/ np.prod(poolsz)))
	return w.astype(np.float32)
	# * to indicate that this variable should be a list containing the elements from 
	# the iterable that weren't explicitly assigned to another variable.
	#https://www.python.org/dev/peps/pep-3132/

def main():
	print('loading data ... ')
	train = loadmat('..\\lg_files\\train_32x32.mat')
	test = loadmat('..\\lg_files\\train_32x32.mat')

	Xtrain = rearrange(train['X'])
	Ytrain = train['y'].flatten() - 1

	del train 

	Xtest = rearrange(test['X'])
	Ytest = test['y'].flatten() -1 
	del test 

	print('End loading data')

	max_iter = 10 
	print_period = 10 

	lr = np.float32(1e-2)

	reg = np.float32(0.01)
	mu = np.float32(0.99)

	N = Xtrain.shape[0]
	batch_sz = 250 
	n_batches = N // batch_sz

	M = 500 
	K = 10 
	poolsz = (2,2)

	W1_shape = (20, 3, 5, 5)
	W1_init = init_filter(W1_shape, poolsz)
	b1_init = np.zeros(W1_shape[0], dtype=np.float32)

	W2_shape = (50,20,5,5)
	W2_init = init_filter(W2_shape, poolsz)
	b2_init = np.zeros(W2_shape[0], dtype=np.float32)

	W3_init = np.random.randn(W2_shape[0]* 5 *5, M) / np.sqrt(W2_shape[0]*5*5 + M)
	b3_init = np.zeros(M, dtype=np.float32)
	W4_init = np.random.randn(M,K)/np.sqrt(M + K)
	b4_init = np.zeros(K, dtype=np.float32)

	X = T.tensor4('X', dtype='float32')
	Y = T.ivector('Y')
	W1 = theano.shared(W1_init, 'W1')
	b1 = theano.shared(b1_init, 'b1')
	W2 = theano.shared(W2_init, 'W2')
	b2 = theano.shared(b2_init, 'b2')
	W3 = theano.shared(W3_init.astype(np.float32), 'W3')
	b3 = theano.shared(b3_init, 'b3')
	W4 = theano.shared(W4_init.astype(np.float32), 'W4')
	b4 = theano.shared(b4_init, 'b4')

	dW1 = theano.shared(np.zeros(W1_init.shape, dtype=np.float32), 'dW1')
	db1 = theano.shared(np.zeros(b1_init.shape, dtype=np.float32), 'db1')
	dW2 = theano.shared(np.zeros(W2_init.shape, dtype=np.float32), 'dW2')
	db2 = theano.shared(np.zeros(b2_init.shape, dtype=np.float32), 'db2')
	dW3 = theano.shared(np.zeros(W3_init.shape, dtype=np.float32), 'dW3')
	db3 = theano.shared(np.zeros(b3_init.shape, dtype=np.float32), 'db3')
	dW4 = theano.shared(np.zeros(W4_init.shape, dtype=np.float32), 'dW4')
	db4 = theano.shared(np.zeros(b4_init.shape, dtype=np.float32), 'db4')

	Z1 = convpool(X, W1, b1)
	Z2 = convpool(Z1, W2, b2)
	Z3 = T.nnet.relu(Z2.flatten(ndim=2).dot(W3) + b3)
	pY = T.nnet.softmax(Z3.dot(W4) + b4)

	params = (W1, b1 ,W2, b2, W3, b3 ,W4, b4)
	reg_cost = reg*np.sum((param*param).sum() for param in params)
	cost = -T.mean(T.log(pY[T.arange(Y.shape[0]), Y]))
	prediction = T.argmax(pY, axis=1)

	update_W1 = W1 + mu*dW1 - lr*T.grad(cost, W1)
	update_b1 = b1 + mu*db1 - lr*T.grad(cost, b1)
	update_W2 = W2 + mu*dW2 - lr*T.grad(cost, W2)
	update_b2 = b2 + mu*db2 - lr*T.grad(cost, b2)
	update_W3 = W3 + mu*dW3 - lr*T.grad(cost, W3)
	update_b3 = b3 + mu*db3 - lr*T.grad(cost, b3)
	update_W4 = W4 + mu*dW4 - lr*T.grad(cost, W4)
	update_b4 = b4 + mu*db4 - lr*T.grad(cost, b4)

	update_dW1 = mu*dW1 - lr*T.grad(cost, W1)
	update_db1 = mu*db1 - lr*T.grad(cost, b1)
	update_dW2 = mu*dW2 - lr*T.grad(cost, W2)
	update_db2 = mu*db2 - lr*T.grad(cost, b2)
	update_dW3 = mu*dW3 - lr*T.grad(cost, W3)
	update_db3 = mu*db3 - lr*T.grad(cost, b3)
	update_dW4 = mu*dW4 - lr*T.grad(cost, W4)
	update_db4 = mu*db4 - lr*T.grad(cost, b4)

	train = theano.function(
		inputs= [X, Y],
		updates=[
			(W1, update_W1),
			(b1, update_b1),
			(W2, update_W2),
			(b2, update_b2),
			(W3, update_W3),
			(b3, update_b3),
			(W4, update_W4),
			(b4, update_b4),
			(dW1, update_dW1),
			(db1, update_db1),
			(dW2, update_dW2),
			(db2, update_db2),
			(dW3, update_dW3),
			(db3, update_db3),
			(dW4, update_dW4),
			(db4, update_db4)
		]
	)


	get_prediction = theano.function(
		inputs=[X, Y],
		outputs= [cost, prediction]
	)

	t0 = datetime.now()
	LL = [] 

	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
			Ybatch = Ytrain[j*batch_sz:(j*batch_sz+batch_sz)]

			train(Xbatch, Ybatch)

			if j % print_period ==0:
				c, y_hat = get_prediction(Xtest, Ytest)
				LL.append(c)
				acc = score(y_hat, Ytest)

				print(f'Cost : {c}   	 acc :{acc}')

	print(f'Elapsed time : {datetime.now() - t0} ')
	plt.plot(LL)
	plt.show()

	# Visualize W1 
	W1_val = W1.get_value()
	grid = np.zeros((8*5, 8*5))
	m =0 
	n =0

	for i in range(20):
		for j in range(3):
			filt = W1_val[i,j]
			grid[m*5:(m+1)*5, n*5:(n+1)*5] = filt 
			m +=1
			if m>= 8:
				m = 0
				n +=1

	plt.imshow(grid, cmap='gray')
	plt.title('W1')
	plt.show()


	# Visualize W2 

	W2_val = W2.get_value()
	grid = np.zeros((32*5, 32* 5))
	m = 0 
	n = 0
	for i in range(50):
		for j in range(20):
			filt = W1_val[i,j]
			grid[m*5:(m+1)*5, n*5:(n+1)*5] = filt 
			m +=1
			if m >= 32:
				m = 0 
				n += 1 

	plt.imshow(grid, cmap='gray')
	plt.title('W2')
	plt.show()


if __name__ == '__main__':
	main()