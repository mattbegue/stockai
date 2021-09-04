import numpy as np 
import tensorflow as tf 
import datetime

class Environment:
	def __init__(self):
		self.shares = 1  #So that we don't sit in the local min of doing nothing
		self.current_price=np.random.random()
		self.avg_cost = self.current_price
		self.cash = 100 - self.current_price
		self.next_price = np.random.random()




		self.memory_length = 1
		self.memory_list = np.zeros(self.memory_length)


	def query_stock(self, training=False):
		if training==True:
			self.current_price = self.next_price
			self.memory_list = np.hstack([self.memory_list[1:], self.next_price])
			self.next_price = np.random.random()
		if training == False:
			self.current_price = np.random.random()
			self.memory_list = np.hstack([self.memory_list[1:], self.current_price])

	def make_action(self, decision):
		#0 Sell;  1 Buy;  2 Hold
		if decision==0:
			#if self.shares!=0:
				self.avg_cost = self.current_price if self.shares<=1 else ((self.avg_cost * self.shares) - self.current_price ) / (self.shares - 1)
				self.shares -= 1
				self.cash += self.current_price
		if decision == 1:
			#if self.cash> self.current_price:
				self.shares += 1
				self.avg_cost =  self.current_price if self.shares<1 else ((self.avg_cost * self.shares) + self.current_price ) / (self.shares + 1)
				self.cash -= self.current_price



class Model_LSTM(tf.keras.Model):
	def __init__(self,  vocab_size=3):
		super(Model, self).__init__()
		self.hidden_dims = [3]
		self.lstm_model = tf.keras.layers.LSTM(self.hidden_dims[-1], return_sequences=False, return_state=True)
		#lstm_cell = [tf.nn.rnn_cell.LSTMCell(hidden) for hidden in self.hidden_dims]
		#self.multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
		#self.vocab_size = vocab_size
		self.denses = tf.keras.Sequential([tf.keras.layers.Dense(5, activation='relu', input_shape=(1,self.hidden_dims[-1]+2)), 
			tf.keras.layers.Dense(3, activation='relu')
			])


	def call(self, env, training=False):
		a,b,c = self.LSTM_push(np.reshape(env.memory_list,[1,len(env.memory_list),1]))
		intermediate_input = tf.concat([a,np.reshape([env.shares/10, env.avg_cost],[1,2])],axis=1)  #/10 to normalize?
		logits = self.denses(intermediate_input)
		logits = tf.reshape(tf.nn.softmax(logits),[-1])
		return logits


	def LSTM_push(self, inputs):
		seq_output, final_mem_state, final_carry_state = self.lstm_model(inputs)
		return seq_output, final_mem_state, final_carry_state

class Model(tf.keras.Model):
	def __init__(self):
		super(Model, self).__init__()
		#self.hidden_dims = [3]
		#self.lstm_model = tf.keras.layers.LSTM(self.hidden_dims[-1], return_sequences=False, return_state=True)
		#lstm_cell = [tf.nn.rnn_cell.LSTMCell(hidden) for hidden in self.hidden_dims]
		#self.multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
		#self.vocab_size = vocab_size
		self.denses = tf.keras.Sequential([tf.keras.layers.Dense(5, activation='relu', input_shape=(1,1+2)), 
			tf.keras.layers.Dense(2, activation='relu')
			])


	def call(self, env, training=False):
		#a,b,c = self.LSTM_push(np.reshape(env.memory_list,[1,len(env.memory_list),1]))
		#intermediate_input = tf.concat([env.memory_list,env.shares/10, env.avg_cost],axis=1)  #/10 to normalize?
		intermediate_input = tf.reshape(tf.concat([env.memory_list, [env.shares/10, env.avg_cost]],axis=-1),(1,-1))  #/10 to normalize?

		logits = self.denses(intermediate_input)
		#logits = tf.reshape(tf.nn.softmax(logits),[-1])
		logits = tf.nn.softmax(logits)

		return logits


	def LSTM_push(self, inputs):
		seq_output, final_mem_state, final_carry_state = self.lstm_model(inputs)
		return seq_output, final_mem_state, final_carry_state



class Agent:
	def loss_function(self, env, action_logits, hold_penalty= 50):
		opportunity_delta_value =  (env.next_price - env.current_price)  #*env.shares
		long_delta_value =  (env.next_price - env.avg_cost)  #* env.shares
		sell_loss =  tf.maximum(opportunity_delta_value * action_logits[0],0)
		hold_loss =  np.abs(opportunity_delta_value * action_logits[1]) * (1+hold_penalty)  #delta_value * action_logits[1] * loss_epsilon  #Do we need a penalty for noop?
		buy_loss = tf.maximum(-opportunity_delta_value * action_logits[2],0)
		#sell_loss =  (opportunity_delta_value * action_logits[0])
		#hold_loss = (opportunity_delta_value * action_logits[1]) * (1+hold_penalty) #delta_value * action_logits[1] * loss_epsilon  #Do we need a penalty for noop?
		#buy_loss = -(opportunity_delta_value * action_logits[2])

		long_sell_loss = long_delta_value * action_logits[0] 
		long_hold_loss =  np.abs(long_delta_value * action_logits[1]) * (1+hold_penalty)  #delta_value * action_logits[1] * loss_epsilon  #Do we need a penalty for noop?
		long_buy_loss = - long_delta_value * action_logits[2] 

		opportunity_loss = sell_loss + hold_loss + buy_loss
		long_loss = long_sell_loss + long_hold_loss + long_buy_loss
		return opportunity_loss 

	def xe_loss(self, env, action_logits):
		opportunity_delta_value =  (env.next_price - env.current_price)  #*env.shares
		long_delta_value =  (env.next_price - env.avg_cost)  #* env.shares
		#oracle_decision = tf.reshape([int(long_delta_value<0), int(long_delta_value>0)],(1,2))
		oracle_decision = tf.reshape([int(opportunity_delta_value<0), int(opportunity_delta_value>0)],(1,2))
		xe = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=.1)
		return xe(oracle_decision, action_logits)


	def invalid_loss(self, env, action_logits, invalid_penalty=100):
		invalid_loss=0
		if env.shares<1:
			invalid_loss=invalid_penalty *(-env.shares + 1) * action_logits[0]
		return invalid_loss

	def make_decision(self, action_logits, action_threshold=.6):
		if np.max(logits) > action_threshold:
			decision = np.argmax(action_logits)
		else:
			decision=2
		return decision




env = Environment()
optimizer = tf.keras.optimizers.Adam()#learning_rate=0.1)

writer = tf.summary.create_file_writer("./logs/gaussian/%d"%int(datetime.datetime.now().timestamp()))



#Burn in 
for _ in range(env.memory_length):
	env.query_stock(training=True)

a = Agent()



m = Model()
m.compile(loss=a.loss_function, optimizer='adam')

for k in range(1000000):
	if k%100==0:
		env=Environment()
		#Burn in 
		for _ in range(env.memory_length):
			env.query_stock(training=True)
	with tf.GradientTape() as tape:
		logits = m(env,training=True)
		loss = a.xe_loss(env, logits) #+ a.invalid_loss(env,logits)
	grads = tape.gradient(loss, m.trainable_weights)
	optimizer.apply_gradients(zip(grads, m.trainable_weights))	
	#decision = tf.argmax(tf.reshape(logits,[-1]))
	decision = a.make_decision(logits)
	if k%99==0:
		with writer.as_default():
			tf.summary.scalar('net_worth', env.cash + env.shares * env.current_price, step=k//100)


	env.make_action(decision)
	print('step:  %d;  net_worth: %f; shares: %d;  cash: %f;  avg share cost: %f;  current share price:   %f;  decision: %d;   loss: %f'%(k%100, env.cash + env.shares * env.current_price, env.shares, env.cash, env.avg_cost, env.current_price, decision, loss))
	env.query_stock(training=True)





