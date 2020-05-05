#!/usr/bin/env python
# coding: utf-8

# In[12]:


#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[3]:


'''
Créditos do código base da DQN ao autor: ??, 2018
DQN modificada por Wenderson Souza
Ambiente preparado por Wenderson Souza
'''

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import glob
import os
import sys
from time import sleep
import math
from random import randrange
try:
    sys.path.append(
        glob.glob('../../carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg')[0])
    sys.path.append(glob.glob('../../PythonAPI/carla/')[0])
except IndexError:
    pass

import carla
import random
import time
#import logging
import cv2
import numpy as np
import pdb
from matplotlib import pyplot as plt
import traceback
from carla import ColorConverter as cc
from collections import deque, Counter
from datetime import datetime


try:
    import queue
except ImportError:
    import Queue as queue


# In[6]:


camera_size_x = 800
camera_size_y = 600
qtd_acoes = 7
####


class Env(object):
    def __init__(self, world):
        self.observation = None
        self.reward = 0
        self.done = 0
        self.info = None
        self.world = world
        self.player = None
        self.dict_act=["sem acao","frente","frente-direita", "frente-esquerda", "ré/freio", "ré-direita", "ré-esquerda"]

    def reset(self):
        self.reward = 0
        self.done = 0
        self.world.restart()
        self.player = self.world.player
        #self.world.camera_sensor.listen(
        #    lambda img: self.world.getCameraImage(img, self))
        #self.world.camera_sensor.listen(self.world.image_queue.put)
        
        print("Ator resetado...")

    def applyReward(self):
        # se houve colisão, negativa em X pontos e termina
        if self.world.colission_history > 0:
            self.reward = self.reward - 10
            self.done = 1
            return
        
        # se esta perto do objetivo, perde menos pontos
        dist_atual = self.world.destiny_dist()
        self.reward = self.reward +             (dist_atual - self.world.last_distance)/dist_atual
        self.world.last_distance = dist_atual

        if dist_atual <= 10: #se esta a menos de 10unidades do destino, fim.
            self.reward = self.reward + 100
            self.done = 1
            return

        vel = self.world.velocAtual()
        #if(vel >= 0):
        #self.reward = self.reward + (vel/(vel+1)) #[RETIRADO para verificar se o veículo irá se locomover mais]

    def step(self, action):

        self.info = self.applyAction(action)

        #self.observation = getObservation(self.world.getObservation())
        self.applyReward()

        #return self.observation, self.reward, self.done, self.info #next_observation vem do tick no main
        return self.reward, self.done, self.info

    def applyAction(self, action):
        speed_limit = 10
        if self.world.velocAtual() >= speed_limit:
            throttle = 0.0
            print("\nAcima do limite de velocidade de", speed_limit, "km/h...")
        else:
            throttle = 0.5
        actions = (
            (0.0, 0.0, False),  # sem acao

            (throttle, 0.0, False),  # frente
            (throttle, -0.5, False),  # frente-direita
            (throttle, 0.5, False),  # frente-esquerda

            (throttle, 0.0, True),  # ré/freio
            (throttle, -0.5, True),  # ré-direita
            (throttle, 0.5, True)  # ré-esquerda
        )  # 7 acoes
        
        #print("\tAção: ", self.dict_act[action] ,"(",actions[action], ")")
        #print("\tAção: ", self.dict_act[action])

        self.player.apply_control(carla.VehicleControl(
            actions[action][0], actions[action][1], reverse=actions[action][2]))
        return None


class World(object):
    def __init__(self, carla_world):
        self.world = carla_world
        self.is_no_rendering = True #False or True
        
        settings = self.world.get_settings()
        settings.no_rendering_mode = self.is_no_rendering
        self.world.apply_settings(settings)
        
        self.map = self.world.get_map()
        self.player = None
        self.collision_sensor = None
        self.camera_sensor = None
        self.destiny = carla.Location(x=9.9, y=0.3, z=20.3)
        self.last_distance = 0
        self.colission_history = 0
        # restart do world é feito pelo Enviroment
        
        

    def spawnPlayer(self):
        # Cria um audi tt no ponto primeiro waypoint dos spawns
        blueprint = (
            self.world.get_blueprint_library().find('vehicle.audi.tt'))
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        spawn_point = (self.world.get_map().get_spawn_points())[0]
        self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # para nao comecar as ações sem ter iniciado adequadamente
        

    def restart(self):
        # Set up the sensors.
        self.destroy()
        self.spawnPlayer()
        self.config_camera()
        self.config_collision_sensor()
        time.sleep(2)

        print("Iniciando componentes do ator...")

    def config_collision_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            bp, carla.Transform(carla.Location(x=2.0, z=1.0)), attach_to=self.player)
        self.collision_sensor.listen(lambda event: self.on_collision(event))

    def on_collision(self, event):
        self.colission_history += 1
        print("\n|| Mais uma colisão... || ")

    def config_camera(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_bp.set_attribute('image_size_x', str(camera_size_x))
        camera_bp.set_attribute('image_size_y', str(camera_size_y))
        # Captura uma imagem a cada 50hz = 0.02
        #camera_bp.set_attribute('sensor_tick', '1')
        #camera_bp.set_attribute('fov', '110')  # angulo horizontal de 110graus
        self.camera_sensor = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.player)

    def destroy(self):
        print("Destruindo ator e sensores...")
        actors = [
            self.camera_sensor,
            self.collision_sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    #def getCameraImage(self, image):
    def convertImage(self, image):
        '''
        é realizado automaticamento 
        quando há uma nova imagem disponível pelo sensor camera_sensor.listen
        '''
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        #array = np.dot(array[..., :3], [0.299, 0.587, 0.144])  #to_grayscale:transforma a matriz 3d em 1d por meio da multiplicação
        # TODO: TESTAR
        # unidimensional
        array = np.expand_dims(array, axis=0)
        return array
        
        #env.observation = array  # repassa para o ambinte uma nova imagem da camera

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

class CarlaSyncMode(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20) #default value de fps=20
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=True, #trocar para testar
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick() #manda o mundo atualizar
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        #debug
        #print(data)
        #sleep(10)
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000

buffer_len = 20000
# limita a quantidade de experiencias. quando encher, retira as ultimas experiencias
exp_buffer = deque(maxlen=buffer_len)

num_episodes = 100
batch_size = 32
learning_rate = 0.001
discount_factor = 0.97

def epsilon_greedy(action, step):
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(qtd_acoes)
    else:
        return action


def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:, 0], mem[:, 1], mem[:, 2], mem[:, 3], mem[:, 4]


def generateNetwork(scope):
    #with tf.variable_scope(name_scope) as scope:
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(
        camera_size_y, camera_size_x, 3)))  # para usar imagem gray, tem que trocar o shape da imagem na layer. X e Y trocados
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='softmax'))
    model.add(layers.Dense(qtd_acoes))
    # model.summary() #display architecture
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model


# In[3]:


mainQ = generateNetwork('mainQ')
targetQ = generateNetwork('targetQ')

action = 0
y=0


# In[4]:


def connect(world, client, end='localhost'):
    try:
        print("Tentando conectar ao servidor Carla...")
        client = carla.Client(end, 2000)
        client.set_timeout(1.0)
        world = World(client.get_world())
    except RuntimeError:
        print("Falha na conexão.")
        return None, None
    print("Conectado com sucesso.")
    return world, client
    
    


# In[5]:


def main(world, client):
    fps = 60
    world=world
    client=client
    try:
        env = Env(world)
        print("Iniciando episodios...")
        sleep(2)
        env.reset()
        print("Inicializando DQN...")
        with CarlaSyncMode(world.world, world.camera_sensor, fps=fps) as sync_mode:    
            global_step = 0
            copy_steps = 10 #a cada x passos irá copiar a main_network para a target_network
            steps_train = 5 #a cada x passos irá treinar a main_network
            start_steps = 20 #passos inicias . default=200

            # for each episode
            for i in range(num_episodes):
                print("%-- Episodio", num_episodes)
                #env.world.tick()
                done = 0
                info = None
                #obs = env.observation  # env.reset()
                snapshot, first_img = sync_mode.tick(timeout=2.0) #snapshot nao esta sendo usado aqui
                obs = world.convertImage(first_img)
                epoch = 0
                episodic_reward = 0
                actions_counter = Counter()
                episodic_loss = []

                while not done:
                    print("|-step(total)", global_step, "begin ; ")
                    #snapshot, image_rgb = sync_mode.tick(timeout=2.0) #atualiza o mundo e retorna as informações
                    # get the preprocessed game screen
                    #obs = 
                    # feed the game screen and get the Q values for each action
                    actions = mainQ.predict(obs)

                    # get the action
                    # escolhe a posicao com maior probabilidade
                    #action.assign(np.argmax(actions))
                    action = np.argmax(actions)
                    actions_counter[str(action)] += 1

                    # select the action using epsilon greedy policy
                    action = epsilon_greedy(action, global_step)

                    # now perform the action and move to the next state, next_obs, receive reward
                    reward, done, info = env.step(action)
                    snapshot, next_obs = sync_mode.tick(timeout=2.0) #atualiza o mundo e retorna as informações
                    next_obs = world.convertImage(next_obs)
                    
                    # Store this transistion as an experience in the replay buffer
                    exp_buffer.append([obs, action, next_obs, reward, done])
                    print(" Reward step: ",reward, "-|")

                    # After certain steps, we train our Q network with samples from the experience replay buffer
                    if global_step % steps_train == 0 and global_step > start_steps:
                        print("--atualização da Q-Network %100passos--")
                        # sample experience
                        obs, act, next_obs, reward, done = sample_memories(
                            batch_size)

                        # TALVEZ PRECISE USAR POR CONTA DO SHAPE DA OBS::
                        # obs = np.expand_dims(obs, axis=0) #transforma as observacoes em um array
                        # next_obs= np.expand_dims(obs, axis=0) #transforma as observacoes em um array
                        # valor de probabilidade da ação mais provável
                        targetValues = targetQ.predict(next_obs)
                        bestAction = np.argmax(targetValues)

                        y = reward + discount_factor *                             np.max(targetValues) * (1 - done)
                        targetValues[bestAction] = y
                        # now we train the network and calculate loss
                        # train mode

                        # gradient descent (x=obs e y=recompensa)

                        train_loss = mainQ.fit(obs, targetValues)
                        episodic_loss.append(train_loss)  # historico
                        print("|| Episodic Loss: ", episodic_loss)

                    # after some interval we copy our main Q network weights to target Q network
                    if (global_step+1) % copy_steps == 0 and global_step > start_steps:
                        # Copy networks weights
                        targetQ.set_weights(mainQ.get_weights())

                    obs = next_obs #troca a u
                    epoch += 1
                    global_step += 1
                    episodic_reward += reward
                    print('Epoch(passos)', epoch, ' and step Reward: ', reward)
                print('Episode', num_episode, ' and episodic(total) Reward: ', episodic_reward, "--%")
                

    except RuntimeError:
        print("\n\ntreta: RuntimeError")
        # traceback.print_exc()
        pass
    except Exception:
        traceback.print_exc()
        pass
    
    finally:

        if world is not None:
            world.destroy()


print("Ready to begin.")


# In[7]:


world=None
client=None
world, client=connect(world, client)#, end='35.197.43.29')


# In[7]:


if __name__ == '__main__' and world!=None and client != None :
    main(world, client)
else:
    print("Erro. Verifique a conexão.")


# In[ ]:




