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

camera_size_x = 800
camera_size_y = 600

class Env(object):
	def __init__(self, world):
		self.observation = None
		self.reward = 0
		self.done = False
		self.info = None
		self.world = world
		self.player = None
	
	def reset(self):
		self.reward = 0
		self.done = False
		self.world.restart()
		self.player = self.world.player
		self.world.camera_sensor.listen(lambda img: self.world.getCameraImage(img, self))
		print("Ator resetado...")

	
	def applyReward(self):
		#se houve colisão, negativa em X pontos e termina
		if self.world.colission_history > 0:
			self.reward = self.reward - 10
			self.done = True
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
		#self.done = True #pelo fato de ser uma função assíncrona, pode ser que dê problema...

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
		array = np.reshape(array, (image.height, image.width, 4))
		
		array = array[:, :, :3]
		array = array[:, :, ::-1]
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
		
		qtd_acoes = 7
		env.reset()

		observation = env.observation #primeira observ
		
		ep = 0
		qtd_episodios = 1000
		
		while ep < qtd_episodios:
			print("\nEpisodio", ep+1)
			env.world.tick()
			#action = dqn.predict(observation)
			action = randrange(qtd_acoes)
			#print("Acao:", action)
			observation, reward, done, info = env.step(action)

			
			''' DQN 
			--definir as taxas e variaveis como epsilon-greedy
			--
			
			preprocessar a primeira imagem ...
			repetir ate acabar os episodios
				cnn para processar a observation (alterar para receber 4 observations); as saidas da cnn são as ações possiveis
				com os valores de saída:
					escolher ação tirando um numero aleatório. se for menor, usar ação aleatório, senão, usa a melhor ação Q
				realizar a ação
				com o novo estado e recompensa, preprocessar a imagem
				salvar a nova transição (imagem, acao, recompensa, proxima imagem resultante da ação) no buffer de experiencias [substituir a mais antiga, caso tenha ultrapassado a capacidade %]

				tirar algumas experiencias aleatorias do buffer para calcular o loss
					--calculo loss com target network
					--realizar o gradiente em nossa rede atual
				apos K passos, atualizar a targe network com os valores de pesos da rede atual 
			





				



			'''
			
			

			print("\tReward: % 3.3f " % reward)
			if done:
				print("Fim do agente #colisao?")
				break
			ep = ep + 1
		

	except RuntimeError:
		print("\n\ntreta: RuntimeError")
		#traceback.print_exc()
		pass

	finally:

		if world is not None:
			world.destroy()
	

if __name__ == '__main__':
	main()