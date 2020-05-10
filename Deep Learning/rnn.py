import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hm_epocs = 3
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

#height x width
x = tf.compat.v1.placeholder('float',[None, n_chunks, chunk_size])	#if second parameter id not consistent with data, tf throws error. If 2nd param is not written, tf loads the data whatever is
y = tf.compat.v1.placeholder('float')

def recurrent_neural_network(x):
	layer = {'weights':tf.Variable(tf.random.normal([rnn_size,n_classes])),
					  'biases': tf.Variable(tf.random.normal([n_classes]))}
	
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1,chunk_size])
	x = tf.split(x, n_chunks, 0)

	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)

	output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

	return output 

def train_neural_network(x):
	prediction = recurrent_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
	optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)

	with tf.compat.v1.Session() as sess:
		sess.run(tf.compat.v1.global_variables_initializer())

		for epoch in range(hm_epocs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
				_, c = sess.run([optimizer,cost], feed_dict = {x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epocs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))


train_neural_network(x)



