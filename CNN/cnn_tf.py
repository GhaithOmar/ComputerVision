import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from scipy.io import loadmat

def score(y,t):
	return np.mean(y == t)

def convpool(X, W,b):
	conv_out = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='SAME')
	conv_out = tf.nn.bias_add(conv_out, b)
	pool_out = tf.nn.max_pool(
		conv_out,
		ksize=[1,2,2,1],
		strides=[1,2,2,1],
		padding='SAME'
	)
	return tf.nn.relu(pool_out)

def rearrange(X):
	return (X.transpose(3, 0, 1, 2) / 255).astype(np.float32)

def init_filter(shape, poolsz):
	w = np.random.randn(*shape)/np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:]/ np.prod(poolsz)))
	return w.astype(np.float32)


def main():
	print('loading data ... ')
	train = loadmat('..\\lg_files\\train_32x32.mat')
	test = loadmat('..\\lg_files\\train_32x32.mat')

	Xtrain = rearrange(train['X'])
	Ytrain = train['y'].flatten() - 1

	Xtrain = Xtrain[:73000,]
	Ytrain = Ytrain[:73000]
	del train 

	Xtest = rearrange(test['X'])
	Ytest = test['y'].flatten() -1
	Xtest = Xtest[: 26000,]
	Ytest = Ytest[: 26000] 
	del test 

	print('End loading data')
	
	print('Initializing Weights')
	N = Xtrain.shape[0]
	pr = 10 
	epoches = 10
	lr = 1e-2
	batch_sz = 500
	n_batches = N// batch_sz
	M = 500 
	K = len(set(Ytrain))
	poolsz= (2,2)

	W1_shape = (5,5,3, 20)
	W1_init = init_filter(W1_shape, poolsz)	
	b1_init = np.zeros(W1_shape[-1], dtype=np.float32)

	W2_shape = (5,5,20,50)
	W2_init = init_filter(W2_shape, poolsz)
	b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

	D = W2_shape[-1] * 8 *8 
	W3_init = np.random.randn(D , M) / np.sqrt(D + M)
	b3_init = np.zeros(M, dtype=np.float32)
	W4_init = np.random.randn(M, K)
	b4_init = np.zeros(K, dtype=np.float32)	

	W1 = tf.Variable(W1_init.astype(np.float32), 'W1')
	b1 = tf.Variable(b1_init, 'b1')
	W2 = tf.Variable(W2_init.astype(np.float32), 'W2')
	b2 = tf.Variable(b2_init, 'b2')
	W3 = tf.Variable(W3_init.astype(np.float32), 'W3')
	b3 = tf.Variable(b3_init, 'b3')
	W4 = tf.Variable(W4_init.astype(np.float32), 'W4')
	b4 = tf.Variable(b4_init, 'b4')

	X = tf.placeholder(tf.float32,shape=(batch_sz, 32, 32, 3), name='X')
	Y = tf.placeholder(tf.int32,shape=(batch_sz,), name='Y')
	print('Strating CNN')
	Z1 = convpool(X, W1, b1)
	Z2 = convpool(Z1, W2, b2)
	Z2_shape = Z2.get_shape().as_list()
	Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
	Z3 = tf.nn.relu(tf.matmul(Z2r, W3)+  b3)
	logits = tf.matmul(Z3, W4) + b4 

	cost = tf.reduce_mean(
		tf.nn.sparse_softmax_cross_entropy_with_logits(
			logits=logits, 
			labels=Y
		)
	)


	train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)

	predict_op = tf.argmax(logits, 1)

	init = tf.global_variables_initializer()
	ll = []
	with tf.Session() as session:
		session.run(init)
		print('Start training')
		for i in range(epoches):
			for j in range(n_batches):
				Xbatch = Xtrain[j*batch_sz:(j*batch_sz+batch_sz)]
				Ybatch = Ytrain[j*batch_sz:(j*batch_sz+batch_sz)]
				session.run(train_op,feed_dict={X:Xbatch, Y:Ybatch})

				if j == pr:    
					# due to RAM limitations we need to have a fixed size input
					# so as a result, we have this ugly total cost and prediction computation
					print('Calculating cost and acc')
					test_cost = 0
					prediction = np.zeros(len(Xtest))
					for k in range(len(Xtest) // batch_sz):
						Xtestbatch = Xtest[k*batch_sz:(k*batch_sz + batch_sz),]
						Ytestbatch = Ytest[k*batch_sz:(k*batch_sz + batch_sz),]
						test_cost += session.run(cost, feed_dict={X: Xtestbatch, Y: Ytestbatch})
						prediction[k*batch_sz:(k*batch_sz + batch_sz)] = session.run(
							predict_op, feed_dict={X: Xtestbatch}
						)
					acc = score(prediction, Ytest)
					print(f'epochs: {i} 	Cost: {test_cost}	acc:  {acc}')


	plt.plot(ll)
	plt.show()



if __name__=='__main__':
	main()