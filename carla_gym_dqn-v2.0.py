'''
Créditos do código base da DQN ao autor: ??, 2018
DQN modificada por Wenderson Souza
Ambiente preparado por Wenderson Souza
'''
import argparse
import glob
import os
import sys
from time import sleep
import math 
from random import randrange
try:
	sys.path.append(glob.glob('../../carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg')[0])
	sys.path.append(glob.glob('../../PythonAPI/carla/')[0])
except IndexError:
	pass

import carla
import random
import time
import logging
import cv2
import numpy as np
import pdb
from matplotlib import pyplot as plt
import traceback
from carla import ColorConverter as cc

import tensorflow as tf
from tensorflow import keras
from collections import deque, Counter
from datetime import datetime

camera_size_x = 800
camera_size_y = 600
qtd_acoes = 7
####

epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000

buffer_len = 20000
exp_buffer = deque(maxlen=buffer_len) #limita a quantidade de experiencias. quando encher, retira as ultimas experiencias

num_episodes = 800
batch_size = 48
learning_rate = 0.001
discount_factor = 0.97

input_shape = (None, camera_size_x, camera_size_y, 1)

global_step = 0
copy_steps = 100
steps_train = 4
start_steps = 2000
####

logdir = 'logs'
tf.reset_default_graph()

mainQ = generateNetwork('mainQ')
targetQ = generateNetwork('targetQ')

action = tf.Variable(0, dtype=uint8)

y = tf.placeholder(tf.float32, shape=(None,1))

loss = tf.reduce_mean(tf.square(y - Q_action))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

def epsilon_greedy(action, step):
	p = np.random.random(1).squeeze()
	epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
	if np.random.rand() < epsilon:
		return np.random.randint(qtd_acoes)
	else:
		return action

def main():
	world = None

	try:
		print("Tentando conectar ao servidor Carla...")
		client = carla.Client('127.0.0.1', 2000)
		client.set_timeout(1.0)

		print("Conectado com sucesso.")
		
		world = World(client.get_world())        
		env = Env(world)
		print("Iniciando episodios...")
		
		env.reset()

	init.run()
	# for each episode
	for i in range(num_episodes):
		env.world.tick()
		done = 0
		info = None
		obs = env.observation #env.reset()
		epoch = 0
		episodic_reward = 0
		actions_counter = Counter()
		episodic_loss = []

		while not done:

			# get the preprocessed game screen
			obs = env.observation
			# feed the game screen and get the Q values for each action
			actions = mainQ.predict(obs)

			# get the action
			action.assign(np.argmax(actions)) #escolhe a posicao com maior probabilidade
			actions_counter[str(action)] += 1 

			# select the action using epsilon greedy policy
			action = epsilon_greedy(action, global_step)

			# now perform the action and move to the next state, next_obs, receive reward
			next_obs, reward, done, info = env.step(action)

			# Store this transistion as an experience in the replay buffer
			exp_buffer.append([obs, action, next_obs, reward, done])

			# After certain steps, we train our Q network with samples from the experience replay buffer
			if global_step % steps_train == 0 and global_step > start_steps:

				# sample experience
				obs, act, next_obs, reward, done = sample_memories(batch_size)

				#TALVEZ PRECISE USAR POR CONTA DO SHAPE DA OBS::
				#obs = np.expand_dims(obs, axis=0) #transforma as observacoes em um array
				#next_obs= np.expand_dims(obs, axis=0) #transforma as observacoes em um array
				next_act = targetQ.predict(next_obs) #valor de probabilidade da ação mais provável
				
				y = reward + discount_factor * np.max(next_act) * (1 - done) 
				# now we train the network and calculate loss
				#train mode 
				
				#gradient descent
				#diferença entre das obse
				
				#gradient_descent(y - mainQ.predict(obs))**2

				mainQ.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
				train_loss = mainQ.fit(obs, y)
				episodic_loss.append(train_loss) #historico

			# after some interval we copy our main Q network weights to target Q network
			if (global_step+1) % copy_steps == 0 and global_step > start_steps:
				targetQ.set_weights(mainQ.get_weights()) #Copy networks weights

			obs = next_obs
			epoch += 1
			global_step += 1
			episodic_reward += reward

		print('Epoch', epoch, 'Reward', episodic_reward,)		

	except RuntimeError:
		print("\n\ntreta: RuntimeError")
		#traceback.print_exc()
		pass

	finally:

		if world is not None:
			world.destroy()
	

if __name__ == '__main__':
	main()

def gradient_descent():
	return 0

def sample_memories(batch_size):
	perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
	mem = np.array(exp_buffer)[perm_batch]
	return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]


def generateNetwork(name,scope):
	with tf.variable_scope(name_scope) as scope:
		model = models.Sequential()
		model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_camera_x, image_camera_y, 1))) #grayscale image
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.Flatten())
		model.add(layers.Dense(128, activation='softmax'))
		model.add(layers.Dense(qtd_acoes))
		#model.summary() #display architecture
		return model




class Env(object):
	def __init__(self, world):
		self.observation = None
		self.reward = 0
		self.done = 0
		self.info = None
		self.world = world
		self.player = None
	
	def reset(self):
		self.reward = 0
		self.done = 0
		self.world.restart()
		self.player = self.world.player
		self.world.camera_sensor.listen(lambda img: self.world.getCameraImage(img, self))
		print("Ator resetado...")

	
	def applyReward(self):
		#se houve colisão, negativa em X pontos e termina
		if self.world.colission_history > 0:
			self.reward = self.reward - 10
			self.done = 1
			return

		#se esta perto do objetivo, perde menos pontos
		dist_atual = self.world.destiny_dist()
		self.reward = self.reward + (dist_atual - self.world.last_distance)/dist_atual 
		self.world.last_distance = dist_atual
		
		vel = self.world.velocAtual()
		self.reward = self.reward + (vel/vel+1)
	

	def step(self, action):
				
		self.info = self.applyAction(action)
		
		#self.observation = getObservation(self.world.getObservation())
		self.applyReward()

		return self.observation, self.reward, self.done, self.info

	def applyAction(self, action):
		speed_limit = 10
		if self.world.velocAtual() >= speed_limit:
			throttle=0.0
			print("\nAcima do limite de velocidade de", speed_limit, "km/h...")
		else:
			throttle=0.5
		actions = (
				(0.0, 0.0, False), #sem acao

				(throttle, 0.0, False), #frente
				(throttle, -0.5, False), #frente-direita
				(throttle, 0.5, False), #frente-esquerda

				(throttle, 0.0, True), #ré/freio
				(throttle, -0.5, True), #ré-direita
				(throttle, 0.5, True) #ré-esquerda
				) #7 acoes

		print("\tAção: ", actions[action])

		self.player.apply_control(carla.VehicleControl(actions[action][0], actions[action][1], reverse=actions[action][2]))
		return None


class World(object):
	def __init__(self, carla_world):
		self.world = carla_world
		self.map = self.world.get_map()
		self.player = None
		self.collision_sensor = None
		self.camera_sensor = None
		self.destiny = carla.Location(x=9.9, y=0.3, z=20.3)
		self.last_distance = 0
		self.colission_history = 0
		#restart do world é feito pelo Enviroment

	def spawnPlayer(self):
		#Cria um audi tt no ponto primeiro waypoint dos spawns
		blueprint = (self.world.get_blueprint_library().find('vehicle.audi.tt'))
		if blueprint.has_attribute('color'):
			color = random.choice(blueprint.get_attribute('color').recommended_values)
			blueprint.set_attribute('color', color)
		# Spawn the player.
		spawn_point = (self.world.get_map().get_spawn_points())[0]
		self.player =  self.world.try_spawn_actor(blueprint, spawn_point)
		time.sleep(2) # para nao comecar as ações sem ter iniciado adequadamente

	def restart(self):
		# Set up the sensors.
		self.destroy()
		self.spawnPlayer()
		self.config_camera()
		self.config_collision_sensor()
		print("Iniciando componentes do ator...")
		
	def config_collision_sensor(self):
		bp = self.world.get_blueprint_library().find('sensor.other.collision')
		self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(carla.Location(x=2.0, z=1.0)), attach_to=self.player)	
		self.collision_sensor.listen(lambda event: self.on_collision(event))

	def on_collision(self, event):
		self.colission_history += 1
		print("\tMais uma colisão...")

	def config_camera(self):
		camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
		camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
		camera_bp.set_attribute('image_size_x', str(camera_size_x))
		camera_bp.set_attribute('image_size_y', str(camera_size_y))
		camera_bp.set_attribute('sensor_tick', '0.02') # Captura uma imagem a cada 50hz = 0.02
		camera_bp.set_attribute('fov', '110') # angulo horizontal de 110graus
		self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.player)
		

	def tick(self):
		time.sleep(0.5)
	
	def destroy(self):
		print("Destruindo ator e sensores...")
		actors = [
			self.camera_sensor,
			self.collision_sensor,
			self.player]
		for actor in actors:
			if actor is not None:
				actor.destroy()
 
	def getCameraImage(self, image, env): 
		'''
		é realizado automaticamento 
		quando há uma nova imagem disponível pelo sensor camera_sensor.listen
		'''
		image.convert(cc.Raw)
		array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
		#array = np.reshape(array, (image.height, image.width, 4))
		
		array = array[:, :, :3]
		array = array[:, :, ::-1]
		array = np.dot(array[...,:3], [0.299, 0.587, 0.144]) #to grayscale

		##TODO: TESTAR
		array = np.reshape(array, (image.height, image.width, 1)) #unidimensional 
		env.observation = array #repassa para o ambinte uma nova imagem da camera



	def defineDestiny(self, d):
		self.destiny = d

	def destiny_dist(self):
		pos = self.player.get_location()
		distance = pos.distance(self.destiny)
		return distance

	def velocAtual(self):
		v = self.player.get_velocity()
		vel = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
		return vel