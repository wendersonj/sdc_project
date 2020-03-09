import argparse
import glob
import os
import sys
from time import sleep
import math 
from random import randrange
try:
	sys.path.append(glob.glob('/home/wenderson/Downloads/CARLA_0.9.7.3/PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg')[0])
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

image_size_x = 800
image_size_y = 600


def process_img(image):
	i = np.array(image.raw_data)
	i2 = i.reshape((image_size_y, image_size_x, 4))
	i3 = i2[:, :, :3]
	cv2.imshow("W1", i3)
	cv2.waitKey(1)
	
	return i3/255.0


def velocAtual(player):
	v = player.get_velocity()
	vel = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
	#print(vel)
	return vel


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================
class World(object):
	def __init__(self, carla_world, dqn):
		self.world = carla_world
		self.map = self.world.get_map()
		self.dqn = dqn
		self.player = None
		self.collision_sensor = None
		self.camera_sensor = None
		self.restart()


	def spawnPlayer(self):
		#Cria um audi tt no ponto primeiro waypoint dos spawns
		blueprint = (self.world.get_blueprint_library().find('vehicle.audi.tt'))
		if blueprint.has_attribute('color'):
			color = random.choice(blueprint.get_attribute('color').recommended_values)
			blueprint.set_attribute('color', color)
		# Spawn the player.
		spawn_point = (self.world.get_map().get_spawn_points())[0]
		self.player =  self.world.try_spawn_actor(blueprint, spawn_point)
		time.sleep(2)


	def restart(self):
		# Set up the sensors.
		#self.collision_sensor = CollisionSensor(self.player, self.hud)
		self.spawnPlayer()
		self.config_camera()


	def config_camera(self):
		camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
		camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
		camera_bp.set_attribute('image_size_x', str(image_size_x))
		camera_bp.set_attribute('image_size_y', str(image_size_y))
		camera_bp.set_attribute('sensor_tick', '0.02') # Captura uma imagem a cada 50hz = 0.02
		camera_bp.set_attribute('fov', '110') # angulo horizontal de 110graus
		self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.player)


	def tick(self):
		#realiza a dqn
		self.dqn.tick(world=self.world, player=self.player)
		#print(velocAtual(self.player))
		

	def destroy(self):
		actors = [
			self.camera_sensor,
			#self.collision_sensor.sensor,
			#self.lane_invasion_sensor.sensor,
			#self.gnss_sensor.sensor,
			self.player]
		for actor in actors:
			if actor is not None:
				actor.destroy()
	
	
	def showImg(self):
		self.camera_sensor.listen(lambda img: process_img(img))

	def applyReward(self):
		#se esta perto do objetivo, perde menos pontos
		self.reward = self.reward - self.destiny_dist(player)/10000 #possivel problema: ele pode querer subir em calçadas ou colidir a chegar no objetivo

		#vetor de direção na direção do objetivo aumenta pontos
		#+proximo de 0, melhor...

		#velocidade da recomenpensa (/360)
		self.reward = self.reward + (velocAtual(player)/360)

		#limitar velocidade a 10km/h
		return self.reward
		reward = 0
		done = False
		return reward, done	
	
# ==============================================================================
# -- DQN ---------------------------------------------------------------
# ==============================================================================
class DQN(object):
	def __init__(self):
		self.reward = 0
		self.x = 0
		self.destiny = carla.Location(x=9.9, y=0.3, z=20.3)


	def destiny_dist(self, player):
		pos = player.get_location()
		distance = pos.distance(self.destiny)
		#print("Posição: ", pos)
		#print("Distancia objetivo: ", distance )
		return distance
	def tick(self, world, player):
		self.applyControl(player, action=2)
		print("Reward: ", self.applyReward(player))

		

	def applyControl(self, player, action=0, random=False):
		qtd_actions = 7
		if random:
			setAction(player, action=randrange(qtd_actions))
		else:
			setAction(player, action=action)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
	world = None

	try:
		client = carla.Client(args.host, args.port)
		client.set_timeout(1.0)
		dqn = DQN()
		world = World(client.get_world(), dqn)        
		i_max = 10000000 #clock
		i = i_max
		#world.showImg()	
		
		while True:
			if(i == 0):
				world.tick()
				i = i_max
			else:
				i = i - 1

			#nova movimentacao ... DQN step         

	except RuntimeError:
		print("treta: RuntimeError")
		pass

	finally:

		if world is not None:
			world.destroy()

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
	argparser = argparse.ArgumentParser(
		description='CARLA Manual Control Client')
	argparser.add_argument(
		'-v', '--verbose',
		action='store_true',
		dest='debug',
		help='print debug information')
	argparser.add_argument(
		'--host',
		metavar='H',
		default='127.0.0.1',
		help='IP of the host server (default: 127.0.0.1)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port to listen to (default: 2000)')
	argparser.add_argument(
		'--res',
		metavar='WIDTHxHEIGHT',
		default='1280x720',
		help='window resolution (default: 1280x720)')

	args = argparser.parse_args()

	args.width, args.height = [int(x) for x in args.res.split('x')]
	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
	logging.info('listening to server %s:%s', args.host, args.port)

	try:

		game_loop(args)

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')

if __name__ == '__main__':
	main()

def setAction(player, action=0):
	speed_limit = 10
	if velocAtual(player) >= speed_limit:
		throttle=0.0
	else:
		throttle=0.5
	actions = [
			(player.apply_control(carla.VehicleControl(throttle=throttle, steer=0.0))), #sem acao
			(player.apply_control(carla.VehicleControl(throttle=throttle, steer=0.0))), #frente
			(player.apply_control(carla.VehicleControl(throttle=throttle, steer=-0.5))), #frente-direita
			(player.apply_control(carla.VehicleControl(throttle=throttle, steer=-1.0))), #frente-esquerda
			(player.apply_control(carla.VehicleControl(throttle=throttle, steer=0.0))), #ré/freio
			(player.apply_control(carla.VehicleControl(throttle=throttle, steer=-0.5))), #ré-direita
			(player.apply_control(carla.VehicleControl(throttle=throttle, steer=-1.0))) #ré-esquerda
			] #7 acoes

	return actions[action]


def episodes(n=1000):
	for i in n:
		observation = env.reset()
		#action = dqn.predict(observation)

		observation, reward, done, info = env.step(action=action)
	    

def reset():
	#return imagem da camera
	return 0

def getObservation():
	obs = 0
	return obs

def step(self, action):
	 """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
	observation = None
	reward = None
	done = False
	info = None
       
    return observation, applyReward(), done, info