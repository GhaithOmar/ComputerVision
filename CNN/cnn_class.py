import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from util import getImageData
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def init_weight_and_bias(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, poolsz):
	w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
	return w.astype(np.float32)

class ConvLayers(object):
	def __init__(self, mi, mo, wt=5,ht=5, poolsz=(2,2)):
		self.poolsz = poolsz
		sz = (wt,ht, mi,mo)
		w_init = init_filter(sz, poolsz)
		self.W = tf.Variable(w_init)
		self.b = tf.Variable(np.zeros(mo, dtype=np.float32))
		self.params=[self.W, self.b]

	def forward(self, X):
		conv_out = tf.nn.conv2d(X, self.W, strides=[1,1,1,1], padding='SAME')
		conv_out = tf.nn.bias_add(conv_out, self.b)
		p1, p2 = self.poolsz
		pool_out = tf.nn.max_pool(
			conv_out,
			ksize=[1,p1,p2,1],
			strides=[1,p1,p2,1],
			padding='SAME'
		)
		return tf.nn.relu(pool_out)


class HiddenLayer(object):
	def __init__(self, M1, M2, f):
		self.M1 = M1
		self.M2 = M2 
		self.f  = f 

		w_init, b_init= init_weight_and_bias(M1, M2)
		self.W = tf.Variable(w_init)
		self.b = tf.Variable(b_init)
		self.params= [self.W, self.b]

	def forward(self,X):
		a = tf.matmul(X, self.W) + self.b 
		return self.f(a)

class CNN():
	def __init__(self, conv_sz, hidden_layer_sz):
		self.conv_sz = conv_sz 
		self.hidden_layer_sz = hidden_layer_sz

	def fit(self, X, Y, Xvalid, Yvalid,activation=tf.nn.relu, lr=1e-2, decay=0.999, mu=0.9, epochs=10, batch_sz=1, show_fig=True):
		X, Xvalid = X.astype(np.float32), Xvalid.astype(np.float32)
		Y, Yvalid = Y.astype(np.int32) ,  Yvalid.astype(np.int32)

		N, width, height, fe = X.shape
		n_batches = N// batch_sz
		dw = width
		dh = height

		self.conv_layer = [] 
		mi = fe 
		for mo, w, h in self.conv_sz:
			conv = ConvLayers(mi,mo,w,h)
			self.conv_layer.append(conv)
			mi=mo 
			dw = dw//2
			dh = dh//2

		M1 = self.conv_sz[-1][0] * dw * dh
		self.hidden_layer  = [] 
		for M2 in self.hidden_layer_sz:
			h = HiddenLayer(M1, M2, activation)
			M1 = M2 
			self.hidden_layer.append(h)

		K= len(set(Y))
		self.hidden_layer.append(HiddenLayer(M1,K,lambda x: x))

		self.params=[] 
		for c in self.conv_layer:
			self.params += c.params
		for h in self.hidden_layer:
			self.params += h.params

		tfX = tf.placeholder(tf.float32, shape=(None, width, height, fe), name='X')
		T = tf.placeholder(tf.int32, shape=(None,),name='T')
		self.tfX = tfX
		logits = self.forward(tfX)

		cost = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits,
				labels=T
			)
		)

		train_op = tf.train.AdamOptimizer(lr).minimize(cost)
		self.predict_op = tf.argmax(logits, 1)

		init = tf.global_variables_initializer()
		ll =[]
		with tf.Session() as self.session:
			self.session.run(init)
			print('Start Training........')
			for i in range(epochs):
				X , Y = shuffle(X, Y)
				for j in range(n_batches):
					Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
					Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

					self.session.run(
						train_op,
						feed_dict={tfX:Xbatch, T:Ybatch}
					)

					if j%10 ==0:
						c = self.session.run(cost, feed_dict={tfX: Xvalid, T: Yvalid})
						acc = self.score(Xvalid, Yvalid)
						ll.append(c)
						print(f'epoch #{i}		Cost: {c}		acc: {acc}')

		if show_fig:
			plt.plot(ll)
			plt.show()




	def forward(self,X):
		Z =X 
		for c in self.conv_layer:
			Z =c.forward(Z)

		Z_shape = Z.get_shape().as_list()
		Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
	
	
		for h in self.hidden_layer:
			Z = h.forward(Z)
		return Z	

	def score(self,X ,Y):
		pY = self.predict(X)
		return np.mean( pY == Y)

	def predict(self, X):
		return self.session.run(self.predict_op, feed_dict={self.tfX: X})



if __name__=='__main__':
	print('Loading data ... ... ......')
	X, Y = getImageData()
	X = X.transpose((0, 2, 3, 1))
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.15)

	model = CNN([(50, 6, 6), (50, 6, 6)],[750,500])

	model.fit(Xtrain, Ytrain, Xtest, Ytest, batch_sz=100, epochs=20)

