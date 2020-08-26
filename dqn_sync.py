"""
Créditos do código base da DQN ao autor: ??, 2018
DQN modificada por Wenderson Souza
Ambiente preparado por Wenderson Souza, baseado no código base da OpenGymIA
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import glob
import os
import sys
from time import sleep
import math
from random import randrange

try:
    sys.path.append(glob.glob("../../carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg")[0])
    sys.path.append(glob.glob("../../PythonAPI/carla/")[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import pdb
import traceback
from carla import ColorConverter as cc
from collections import deque, Counter
from datetime import datetime
import queue

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
'''
Para consertar o erro CUDNN_STATUS_INTERNAL_ERROR
'''

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


'''
To-do:
Receber três imagens da câmera por tick (percepção de movimento)
'''


# Variáveis Simulacao
camera_size_x = 800
camera_size_y = 600
qtd_acoes = 7
# Variáveis DQN
epsilon = 0.5
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000
buffer_len = 20000  # limita a quantidade de experiencias. quando encher, retira as ultimas experiencias
exp_buffer = deque(maxlen=buffer_len)
QTD_EPISODIOS = 2 #100
BATCH_SIZE = 32
learning_rate = 0.001
discount_factor = 0.97
# Variáveis de mundo e cliente
world = None
client = None
SPEED_LIMIT = 40

historico = []

class Env(object):
    def __init__(self, world):
        #self.observation = None
        self.reward = 0
        self.done = 0
        self.info = None
        self.player = None
        self.dict_act = [
            "sem acao",
            "frente",
            "frente-direita",
            "frente-esquerda",
            "ré/freio",
            "ré-direita",
            "ré-esquerda",
        ]
        self.frameObs = None
        self.image_queue = queue.Queue()

    def reset(self):
        print("Reiniciando ambiente...")
        self.reward = 0
        self.done = 0
        world.restart()
        print("Iniciando componentes do ator...")
        self.player = world.player
        print("is player None? ", self.player)
        print("Câmera iniciada")
        world.camera_sensor.listen(lambda img: self.convertImage(img))
        print("Ator resetado...")

    def applyReward(self):
        # se houve colisão, negativa em X pontos e termina
        if world.colission_history > 0:
            self.reward = self.reward - 5
            self.done = 1
            return 

        vel = world.velocAtual()

        #não tenho noção de velocidade...
            
        if vel > SPEED_LIMIT and vel < 20:
            print("\nAcima do limite de velocidade de ", SPEED_LIMIT, "km/h. \nVelocidade Atual: ", vel)
            self.reward = self.reward + 0.02
        else:
            self.reward = self.reward + 0.1
        
        


        # se esta perto do objetivo, perde menos pontos
        '''
        #não tenho noção de posição / localização
        dist_atual = world.destiny_dist()
        self.reward = self.reward + (dist_atual - world.last_distance) / dist_atual
        world.last_distance = dist_atual
        

        if dist_atual <= 10:  # se esta a menos de 10unidades do destino, fim.
            self.reward = self.reward + 100
            self.done = 1
            return
        
        '''
        
        #distancia do centro da faixa ?

    def step(self, action):
        self.info = self.applyAction(action)
        world.tick()  # atualiza o mundo
       #print("Frame recebido (step): ", self.frameObs)
        self.applyReward()
        return self.getObservation(), self.reward, self.done, self.info

    def applyAction(self, action):
        throttle = 0.5
        actions = (
            (0.0, 0.0, False),  # sem acao
            (throttle, 0.0, False),  # frente
            (throttle, -0.5, False),  # frente-direita
            (throttle, 0.5, False),  # frente-esquerda
            (throttle, 0.0, True),  # ré/freio
            (throttle, -0.5, True),  # ré-direita
            (throttle, 0.5, True),  # ré-esquerda
        )  # 7 acoes

        print("\tAção: ", actions[action] , '(', self.dict_act[action],')') 
        self.player.apply_control(
            carla.VehicleControl(
                actions[action][0], actions[action][1], reverse=actions[action][2]
            )
        )
        return self.dict_act[action]

    def getObservation(self):
        print("Esperando imagem ...")
        return self.image_queue.get()    

    def convertImage(self, image):
        """
        é realizado automaticamente 
        quando há uma nova imagem disponível pelo sensor camera_sensor.listen
        """
        # image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = np.expand_dims(array, axis=0)
        print("Colocando imagem na fila")    
        self.image_queue.put(array)
        print("Imagem colocada na fila")        

class World(object):
    def __init__(self, carla_world):
        """
        Incialização das variáveis
        """
        self.carla_world = carla_world
        self.player = None
        self.collision_sensor = None
        self.camera_sensor = None
        self.destiny = carla.Location(x=9.9, y=0.3, z=20.3)
        self.last_distance = 0
        self.colission_history = 0
        # restart do world é feito pelo Enviroment

    def spawnPlayer(self):
        self.player = None
        # Cria um audi tt no ponto primeiro waypoint dos spawns
        blueprint = self.carla_world.get_blueprint_library().find("vehicle.audi.tt")
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        # Spawn the player.
        print("Convocando carro-EGO...")
        spawn_point = (self.carla_world.get_map().get_spawn_points())[0]
        while(self.player == None):
            self.player = self.carla_world.try_spawn_actor(blueprint, spawn_point)
        print("Carro-EGO convocado no mapa")
        # para nao comecar as ações sem ter iniciado adequadamente

        """
        prepara o carro
        """
        print("Esquentando o carro-ego ...") #
        for i in range(20):
            self.tick()

    def tick(self):
        #fixed_delta_seconds indica que são 20 frames
        frame = self.carla_world.tick()

    def restart(self):
        # Set up the sensors.
        print("Reiniciando mundo ...")
        self.destroy()
        self.config_carla_world()
        self.spawnPlayer()
        self.config_camera()
        self.config_collision_sensor()
        print("Fim do reinício do mundo.")

    def config_carla_world(self):
        """
        Configuração do World
        -Será síncrono e quando atualizar, atualizará 20 frames.
        """
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # quando tick for usado, a simulação irá simular 1/fixed_delta_seconds frames. Em 0.05 serão 20 frames por tick; 0.5 serão 2 frames por tick. Usar sempre tempo < 0.1
        self.carla_world.apply_settings(settings)

    def config_collision_sensor(self):
        print("Configurando sensor colisão ...")
        bp = self.carla_world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.carla_world.spawn_actor(
            bp, carla.Transform(carla.Location(x=2.0, z=1.0)), attach_to=self.player
        )
        self.collision_sensor.listen(lambda event: self.on_collision(event))

    def on_collision(self, event):
        self.colission_history += 1
        print("\n|| COLISÃO ! || ")

    def config_camera(self):
        print("Configurando câmera 1...")
        camera_bp = self.carla_world.get_blueprint_library().find("sensor.camera.rgb")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_bp.set_attribute("image_size_x", str(camera_size_x))
        camera_bp.set_attribute("image_size_y", str(camera_size_y))
        self.camera_sensor = self.carla_world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.player
        )

    def destroy(self):
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = False
        self.carla_world.apply_settings(
            settings
        )  # Perde o controle manual da simulação
        print("Destruindo atores e sensores...")
        actors = [self.camera_sensor, self.collision_sensor, self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def defineDestiny(self, d):
        self.destiny = d

    def destiny_dist(self):
        pos = self.player.get_location()
        distance = pos.distance(self.destiny)
        return distance

    def velocAtual(self):
        v = self.player.get_velocity()
        vel = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        return vel

def epsilon_greedy(action, step):
    p = np.random.random(1).squeeze()
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(qtd_acoes)
    else:
        return action

def sample_memories(BATCH_SIZE):
    perm_batch = np.random.permutation(len(exp_buffer))[:BATCH_SIZE]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:, 0][0], mem[:, 1][0], mem[:, 2][0], mem[:, 3][0], mem[:, 4][0]

def generateNetwork(scope):
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            filters=64, kernel_size=3, activation="relu", input_shape=(600, 800, 3)
        )
    )  # para usar imagem gray, tem que trocar o shape da imagem na layer. Y, X.
    model.add(layers.MaxPooling2D((2, 2))) #usa um filtro de max-pool 2x2: camera_size/2
    model.add(layers.Conv2D(32, kernel_size=3, activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, kernel_size=3, activation="relu")) #64
    model.add(layers.Flatten())
    model.add(layers.Dense(8, activation="relu")) #128
    model.add(layers.Dense(qtd_acoes, activation="softmax"))
    model.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
    return model

def connect(world, client, ip="localhost"):
    try:
        print("Tentando conectar ao servidor Carla...")
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        carla_world = World(client.get_world())
    except RuntimeError:
        print("Falha na conexão. Verifique-a.")
        return None, None
    print("Conectado com sucesso.")
    return carla_world, client

###

filepath1="rede.{epoch:02d}-{val_loss:.2f}.hdf5"
filepath2="rede.{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint1 = ModelCheckpoint(filepath1, monitor='val_loss', save_best_only=True, verbose=1, mode='min', save_weights_only=False)
checkpoint2 = ModelCheckpoint(filepath2, monitor='val_accuracy', save_best_only=True, verbose=1, mode='max', save_weights_only=False)
callbacks_list = [checkpoint1, checkpoint2]

def make_graph(x, y, xlabel="", ylabel="", title="", save_img=False, show_img=False, file_name="graphimg.jpg"):
    if type(x) is int:
        xi = list(range(x))
    else: #array
        xi = list(range(len(x)))
        plt.xticks(xi, x)
    plt.plot(xi, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_img:
        plt.savefig(file_name)
    if show_img:
        plt.show()
    plt.close(fig)

###

def main():
    try:
        print("Iniciando mundo Carla...")
        env = Env(world)
        #env.reset()
        print("Inicializando DQN...")

        global_step = 0  # passos totais (somatorio dos passos em cada episodio)
        copy_steps = 100 # a cada x passos irá copiar a main_network para a target_network
        steps_train = 50  # a cada x passos irá treinar a main_network
        start_steps = 100  # passos inicias. (somente após essa qtd de passo irá treinar começar a rede)

        print("Gerando rede DQN principal...")
        mainQ = generateNetwork('mainQ')
        #mainQ.summary()
        print("Gerando rede DQN alvo...")
        targetQ = generateNetwork('targetQ')
        print("Redes geradas.")

        action = 0 #ação escolhida
        y = 0 #y do dqn [?]

        print("Iniciando episodios...")
        for episodio in range(QTD_EPISODIOS):
            env.reset() 
            
            print("\n%Episodio", episodio, "%")
            world.tick()  # atualiza o mundo
            obs = env.getObservation()
            passos_ep = 0  # quantidade de passos realizados em um episodio
            actions_counter = Counter()
            episodic_loss = []

            while not env.done:
                print("=> Passo ", global_step, " - Início ")
                # Prediz uma ação (com base no que possui treinada) com base na observação - por enquanto, apenas uma imagem de câmera
                actions = mainQ.predict(obs)
                # A rede produz um resultado em %. Logo, escolhe a posição do vetor(ação) com maior probabilidade
                action = np.argmax(
                    actions
                )  # neste caso, argmax retorna a posição com maior probabilidade
                actions_counter[str(action)] += 1 #?

                """ 
                Por mais que a rede tenha escolhido uma ação mais "adequada"(maior probabilidade), o método da DQN usa a política de ganância, que verifica se deve usar a própria experiência (rede DQN) ou se explora o ambiente com uma nova ação aleatória.
                """
                action = epsilon_greedy(action, global_step)

                # Realiza a ação escolhida e recupera o próximo estado, a recompensa, info e se acabou(se houve colisão).
                next_obs, reward, done, info = env.step(action)
                #print("next_obs:", next_obs)
                #print("done:", done)
                #print("info:", info)
                

                # Armazenamento das experiências no buffer de replay
                exp_buffer.append([obs, action, next_obs, reward, done])
                print("Terminou o PASSO com recompensa: ", reward, "")

                # Treino da rede principal, após uma qtd de passos, usando as experiências do buffer de replay.
                if global_step % steps_train == 0 and global_step > start_steps:
                    print("\n-- Atualização da Q-Network %", steps_train, "passos --")
                    # sample experience
                    obs, act, next_obs, reward, done = sample_memories(BATCH_SIZE)

                    # valor de probabilidade da ação mais provável
                    targetValues = targetQ.predict(next_obs)
                    print("Valores alvo: ", targetValues)

                    bestAction = np.argmax(targetValues)
                    print("Melhor Ação (prediction): ", bestAction)

                    y = reward + discount_factor * np.max(targetValues) * (1 - done)
                    print("Y atualizado: ", y)
                    print(
                        "Antes (melhor ação - rede): ",
                        targetValues[0][bestAction],
                        "| Depois (y): ",
                        y,
                    )

                    targetValues[0][bestAction] = y
                    """
                    now we train the network and calculate loss
                    gradient descent (x=obs e y=recompensa)
                    """
                    train_loss = mainQ.fit(obs, targetValues, batch_size=BATCH_SIZE, epochs=1, shuffle=True, callbacks=[checkpoint1, checkpoint2])
                    #episodic_loss.append(train_loss)  # historico [REVISAR ISSO AQUI]
                    #print("|| Perda do episódio (episodic loss): ", episodic_loss.result())

                if (global_step + 1) % copy_steps == 0 and global_step > start_steps:
                    # Cópia dos pesos da rede principal para a rede alvo.
                    targetQ.set_weights(mainQ.get_weights())

                obs = next_obs  # troca a obs
                passos_ep += 1
                global_step += 1
                
                if passos_ep == 10:
                    env.done = 1
            print(
                "=> Fim do Episódio ",
                episodio,
                "com ",
                passos_ep,
                " passos e recompensa total de: ",
                env.reward,
                " pontos.\n",
            )
            historico.append([episodio, passos_ep, env.reward, actions_counter])
            historico.append([episodio, passos_ep, env.reward, actions_counter])
            #print(historico[:, 0])

            x = [ep for ep in historico[ep][0]]
            print(x)
            #make_graph(x=historico[:][0], y=historico[:][2], ylabel="", title="historico de recompensas", save_img=True, file_name="historico_recompensas.jpg")
            #make_graph(x=historico[:][0], y=historico[:][1], ylabel="", title="passos por episódio", save_img=True, file_name="passos_por_ep.jpg")
            #histograma do actions counter

    except RuntimeError:
        print("\nRuntimeError. Trace: ")
        traceback.print_exc()
        pass
    except Exception:
        traceback.print_exc()
        pass
    finally:
        if world is not None:
            world.destroy()


"""
Conexão com o simulador
"""
world, client = connect(world, client , ip='192.168.1.103')

if __name__ == "__main__" and world != None and client != None:
    main()
