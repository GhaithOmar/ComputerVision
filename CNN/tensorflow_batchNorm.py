import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 

class HiddenLayerBatchNorm(object):
	def __init__(self, M1, M2, f):
		self.M1 = M1 
		self.M2 = M2 
		self.f  = f 

		W = np.random.randn(M1, M2)/np.sqrt(M1)
		self.W = tf.Variable(W.astype(np.float32))

		gamma = np.ones(M2).astype(np.float32)
		beta  = np.zeros(M2).astype(np.float32)
		self.gamma = tf.Variable(gamma)
		self.beta = tf.Variable(beta)

		self.rn_var  = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)
		self.rn_mean = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)

	def forward(self,X, is_training,decay=0.9):
		a = tf.matmul(X,self.W)
		if is_training:
			batch_mean, batch_var = tf.nn.moments(a,[0])
			update_rn_mean = tf.assign(
				self.rn_mean,
				self.rn_mean * decay + batch_mean*(1-decay)
			)
			update_rn_var = tf.assign(
				self.rn_var,
				self.rn_var * decay + batch_var*(1-decay)
			)

			with tf.control_dependencies([update_rn_mean, update_rn_var]):
				out = tf.nn.batch_normalization(
					a, 
					batch_mean, 
					batch_var, 
					self.beta, 
					self.gamma,
					1e-4
				)
		else:
			out = tf.nn.batch_normalization(
				a, 
				self.rn_mean, 
				self.rn_var, 
				self.beta, 
				self.gamma,
				1e-4
			)
		return self.f(out) 


class HiddenLayer(object):
	def __init__(self, M1, M2, f):
		self.f  = f 
		self.M1 = M1
		self.M2 = M2 
		W = np.random.randn(M1,M2)/np.sqrt(M1)
		b = np.zeros(M2)
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))

		self.params = [self.W, self.b]

	def forward(self,X):
		return self.f(tf.matmul(X, self.W) + self.b)


class ANN(object):
	def __init__(self, hiddden_layers_sz):
		self.hiddden_layers_sz = hiddden_layers_sz

	def fit(self, X, Y, Xvalid, Yvalid, activation=tf.nn.relu,lr=1e-3,batch_sz=1, epochs=20,show_fig=False ):
		X = X.astype(np.float32)
		Y = Y.astype(np.int64)
		Xvalid = Xvalid.astype(np.float32)
		Yvalid = Yvalid.astype(np.int64)

		N, D = X.shape
		K = len(set(Y))

		n_batches = N // batch_sz

		self.hidden_layer = []
		M1 = D 
		for M2 in self.hiddden_layers_sz:
			h = HiddenLayerBatchNorm(M1, M2, activation)
			self.hidden_layer.append(h)
			M1 = M2

		h = HiddenLayer(M1,K, lambda X: X)
		self.hidden_layer.append(h)
		


		thX = tf.placeholder(tf.float32, shape=(None,D), name='X')
		T   = tf.placeholder(tf.int64, shape=(None,), name='T')
		self.thX = thX
		logits = self.forward(thX, is_training=True)
		cost = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=logits,
				labels=T
			)
		)

		test_logits = self.forward(thX, is_training=False)
		self.prediction = tf.argmax(test_logits, 1)
		train_op = tf.train.AdamOptimizer(lr).minimize(cost)
		

		init = tf.global_variables_initializer()

		ll = []

		with tf.Session() as self.s:
			self.s.run(init)
			for i in range(epochs):
				for j in range(n_batches):
					Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
					Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

					self.s.run(train_op, feed_dict={thX:Xbatch, T: Ybatch})
					# we pass the RUN/

					if j %50 ==0:
						c = self.s.run(cost, feed_dict={thX: Xvalid , T: Yvalid })
						
						ll.append(c)
						acc = self.score(Xvalid, Yvalid)

						print(f'epoch number {i} batch {j}  cost= {c}    , acc= {acc}')

		if show_fig:
			plt.plot(ll)
			plt.title('cost')
			plt.show()


	def forward(self,X,is_training):
		Z = X 
		for h in self.hidden_layer[:-1]:
			Z = h.forward(Z, is_training)

		return self.hidden_layer[-1].forward(Z)

	def predict(self, X):
		return self.s.run(self.prediction, feed_dict={self.thX: X})

	def score(self, X, T):
		pY = self.predict(X)
		return np.mean(pY == T)



