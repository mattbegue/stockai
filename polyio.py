#conda activate stockai

import numpy as np
import pandas as pd
import datetime
import time
import os
from polygon import RESTClient
import tensorflow as tf 
from matplotlib import pyplot as plt
import holidays

apikey = 'Js_pNztNnUBY8gHcZyl1h5ttjjYuy3p_'

#####################################
#build training data
#####################################
trainingdatadir = '/Users/mba0330/MLprojects/StockAI/trainingdata/AAPL/'

starttime = '2019-04-23'
d = datetime.datetime.strptime(starttime,'%Y-%m-%d')
with RESTClient(apikey) as client:
	for k in range(100):
		d0str = d.strftime('%Y-%m-%d')
		d1str = d.strftime('%Y-%m-%d')
		if (d.weekday()<5) &  (d.date() not in holidays.US()):
			resp = client.stocks_equities_aggregates('AAPL',1,'minute',d0str, d1str)
			x = pd.DataFrame(resp.results)
			x.to_csv(os.path.join(trainingdatadir,d0str+'.csv'))
		time.sleep(12)
		d = d+datetime.timedelta(days=1)



with RESTClient(apikey) as client:
	resp = client.stocks_equities_aggregates('AAPL',1,'minute',d0str, d1str)
	x = pd.DataFrame(resp.results)


for k in range(10):
 	print((x[k+1]['t']-x[k]['t'])/1000/60)





sequence_length = 200



class LSTM_for_reconstruction:
	def __init__(self, batch_size=32, vocab_size=300):
		self.batch_size = batch_size
		self.hidden_dims = [100,400]
		#lstm_cell = [tf.nn.rnn_cell.LSTMCell(hidden) for hidden in self.hidden_dims]   #Without dropout
		lstm_cell = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=.8, output_keep_prob=.6, state_keep_prob=1) for cell in lstm_cell]  
		self.multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
		self.vocab_size = vocab_size


	def push(self, inputs, sequence_length, reuse=False, is_training=True):
		logits, state = tf.nn.dynamic_rnn(cell=self.multi_lstm_cell, inputs=inputs,  sequence_length=sequence_length,  dtype=tf.float64)
		logits = tf.layers.dense(inputs = logits, units = self.vocab_size, activation=tf.nn.relu, trainable=is_training, reuse=reuse, name='FC2', kernel_initializer = tf.initializers.he_uniform())		#probability = tf.nn.softmax(logits)
		#outputs_maxlabel = tf.argmax(probability,axis=-1)
		return logits

	def reconstruction_loss(self, pairwise_distance_matrix):
		return tf.reduce_sum(pairwise_distance_matrix)

	def triplet_loss(self, anchor, pos, neg, margin = 10):
		loss = tf.math.maximum((tf.norm(anchor-pos, axis=1) - tf.norm(anchor-neg, axis=1) + margin),0)
		loss = tf.reduce_mean(loss)
		return loss



np.array(x[['v','o','c','h','l']])

xxx = tf.placeholder(tf.float64, shape=(batch_size, sequence_length,2))






