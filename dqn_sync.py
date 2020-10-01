'''
Créditos do código base da DQN retirado do livro : 'Hands-On Reinforcement Learning with Python: Master reinforcement 
    and deep reinforcement learning using OpenAI Gym and TensorFlow', 2018. Ravichandiran, Sudharsan.

DQN modificada por Wenderson Souza para utilizar a framework Keras.
Ambiente preparado por Wenderson Souza, baseado no código base da OpenGymIA e códigos de exemplos fornecidos pela 
    documentação do simulador Carla
'''

'''
To-do:
Receber três imagens da câmera por tick (percepção de movimento).
problemas ao receber 3 imagens... verificar a classe observation
'''

import glob
import os
import sys
from time import sleep
import math
from random import randrange
from datetime import datetime

try:
    sys.path.append(glob.glob('../../carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg')[0])
    sys.path.append(glob.glob('../../PythonAPI/carla/')[0])
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


import tensorflow as tf
print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#from tensorflow import summary as tf.summary


from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt

'''
Para consertar o erro CUDNN_STATUS_INTERNAL_ERROR
'''

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# Variáveis Simulacao
CAMERA_SIZE = (800, 600)
QTD_ACOES = 7
# Variáveis DQN
epsilon = -1 #apenas inicializada
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000
#
buffer_len = 20000  # limita a quantidade de experiencias. quando encher, retira as ultimas experiencias
exp_buffer = deque(maxlen=buffer_len)
QTD_EPISODIOS = 2 #100
BATCH_SIZE = 32
learning_rate = 0.001
discount_factor = 0.9
# Variáveis de mundo e cliente
world = None
client = None
SPEED_LIMIT = 40

# Define the Keras TensorBoard callback.
logdir='logs/' + datetime.now().strftime('%d-%m-%Y--%H:%M:%S')
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

filepath='redeCheckpoint.{epoch:02d}-{accuracy:.2f}.hdf5'
checkpoint1 = ModelCheckpoint(filepath, monitor='accuracy', save_best_only=True, verbose=1, mode='max', save_weights_only=False)
callbacks_list = [checkpoint1]

#grafico de recompensa por época personalizado
historico_episodio_summary = tf.summary.create_file_writer()

global_training_history = [] #armazena os resultados do fit.

class Env(object):
    def __init__(self, world):
        #self.observation = None
        self.reward = 0
        self.done = 0
        self.info = None
        self.player = None
        self.DICT_ACT = [
            'sem acao',
            'frente',
            'frente-direita',
            'frente-esquerda',
            'ré/freio',
            'ré-direita',
            'ré-esquerda',
        ]
        self.frameObs = None
        self.image_queue = queue.Queue() #jackpot: toda vez que ocorre um .get(), ocorre um tick
        self.passos_ep = 0
        self.coord_faixas = []

        '''
        Ações
        throttle=aceleração
        '''
        self.throttle = 0.5
        self.ACTIONS = (
            (0.0, 0.0, False),  # sem acao
            (self.throttle, 0.0, False),  # frente
            (self.throttle, -0.5, False),  # frente-direita
            (self.throttle, 0.5, False),  # frente-esquerda
            (self.throttle, 0.0, True),  # ré/freio
            (self.throttle, -0.5, True),  # ré-direita
            (self.throttle, 0.5, True),  # ré-esquerda
        )  # 7 acoes

    def reset(self):
        print('< Reiniciando ambiente...')
        self.reward = 0
        self.done = 0
        self.passos_ep = 0
        world.restart()
        print('< Iniciando componentes do ator...')
        self.player = world.player
        print('< is player None? ', self.player)
        print('< Câmera iniciada')
        world.camera_sensor.listen(lambda img: self.convertImage(img))
        print('< Ator resetado.')

    def applyReward(self):
        # se houve colisão, negativa em X pontos e termina
        if world.colission_history > 0:
            self.reward = self.reward - 100
            self.done = 1
            return 

        #velocidade do veiculo
        vel = world.velocAtual()
        #não tenho noção de velocidade...
        print('> Velocidade atual: ', vel, 'km/h.')
        #vel <=1: reward += 0.01
        
        if vel > 5 and vel < SPEED_LIMIT:
            self.reward = self.reward + 0.5
        elif vel > SPEED_LIMIT:
            self.reward = self.reward - 0.3
        #se não for, nao ganha recompensa, mas tb nao perde

        #self.reward = self.reward + (1 - (vel/SPEED_LIMIT)**(0.4*vel))
        
        '''Se demora muito para se locomover e juntar recompensa, perde do mesmo jeito.
        O objetivo aqui é fazer o veículo se locomover para acumular recompensa.
        '''
       
        if self.coord_faixas not None:
            self.reward = self.reward + (-(((CAMERA_SIZE[1]/2)-self.coord_faixas[3][1]) ** 4)+1)

        #limite de tempo (passos)
        if self.passos_ep >= 1000:
            self.reward = self.reward - 10
            self.done = 1
            return

        
        '''
        # se esta perto do objetivo, perde menos pontos
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
        self.passos_ep = self.passos_ep + 1
        self.info = self.applyAction(action)
        world.tick()  # atualiza o mundo
        '''
        O que acontece se eu der 3 ticks ?  verificar o frame e as imagens recebidas
        -verificar se consigo plotar
        '''
        #print('Frame recebido (step): ', self.frameObs)
        self.applyReward()
        return self.getObservation(), self.reward, self.done, self.info

    def applyAction(self, action):
        print('> Ação: ', self.ACTIONS[action] , '(', self.DICT_ACT[action],')') 
        self.player.apply_control(
            carla.VehicleControl(
                self.ACTIONS[action][0], self.ACTIONS[action][1], reverse=self.ACTIONS[action][2]
            )
        )
        return self.DICT_ACT[action]

    def getObservation(self):
        print('< Esperando imagem ...')
        '''
        deve receber as três imagens e concatenar
        '''
        obs = Observation()
        for i in range(3):
            obs.append(self.image_queue.get())
        return obs

    def convertImage(self, image):
        '''
        é realizado automaticamente 
        quando há uma nova imagem disponível pelo sensor camera_sensor.listen
        '''
        # image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1] #inverte a ordem das camadas RGB
        array = np.expand_dims(array, axis=0)
        #self.extractROI(array)
        print('< Colocando imagem na fila')    
        self.image_queue.put(array)
        print('< Imagem colocada na fila')      
    
    def extractROI(image):
        roi = 0
        return roi

class Observation(object):
    #def __init__(self, obs1, obs2, obs3, veloc, coord_faixas):
    def __init__(self, img, veloc, coord_faixas):
        self.img = img
        self.coord_faixas = coord_faixas #última faixa ?
        self.veloc = veloc #veloc media entre as 3 obs ?
    
    def calculaCentroFaixa(coord_faixas)
        return None

class World(object):
    def __init__(self, carla_world):
        '''
        Incialização das variáveis
        '''
        self.carla_world = carla_world
        self.player = None
        self.collision_sensor = None
        self.camera_sensor = None
        #self.destiny = carla.Location(x=9.9, y=0.3, z=20.3)
        #self.last_distance = 0
        self.colission_history = 0
        # restart do world é feito pelo Enviroment

    def spawnPlayer(self):
        self.player = None
        # Cria um audi tt no ponto primeiro waypoint dos spawns
        blueprint = self.carla_world.get_blueprint_library().find('vehicle.audi.tt')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        print('< Convocando carro-EGO...')
        spawn_point = (self.carla_world.get_map().get_spawn_points())[0]
        while(self.player == None):
            self.player = self.carla_world.try_spawn_actor(blueprint, spawn_point)
        print('< Carro-EGO convocado no mapa')
        # para nao comecar as ações sem ter iniciado adequadamente

        '''
        prepara o carro
        '''
        print('Esquentando o carro-ego ...') #
        for i in range(20):
            self.tick()
        print('Carro-ego preparado.')

    def tick(self):
        #fixed_delta_seconds indica que são 20 frames
        frame = self.carla_world.tick()

    def restart(self):
        # Set up the sensors.
        print('< Reiniciando mundo ...')
        self.destroy()
        self.config_carla_world()
        self.spawnPlayer()
        self.config_camera()
        self.config_collision_sensor()
        print('< Fim do reinício do mundo.')

    def config_carla_world(self):
        '''
        Configuração do World
        -Será síncrono e quando atualizar, atualizará 20 frames.
        '''
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # quando tick for usado, a simulação irá simular 1/fixed_delta_seconds frames. Em 0.05 serão 20 frames por tick; 0.5 serão 2 frames por tick. Usar sempre tempo < 0.1
        self.carla_world.apply_settings(settings)

    def config_collision_sensor(self):
        print('< Configurando sensor colisão ...')
        bp = self.carla_world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.carla_world.spawn_actor(
            bp, carla.Transform(carla.Location(x=2.0, z=1.0)), attach_to=self.player
        )
        self.collision_sensor.listen(lambda event: self.on_collision(event))

    def on_collision(self, event):
        self.colission_history += 1
        print('> || COLISÃO ! || ')

    def config_camera(self):
        print('< Configurando câmera 1...')
        camera_bp = self.carla_world.get_blueprint_library().find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=0.2, z=1.5))
        camera_bp.set_attribute('image_size_x', str(CAMERA_SIZE[0]))
        camera_bp.set_attribute('image_size_y', str(CAMERA_SIZE[1]))
        self.camera_sensor = self.carla_world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.player
        )

    def destroy(self):
        settings = self.carla_world.get_settings()
        settings.synchronous_mode = False
        self.carla_world.apply_settings(
            settings
        )  # Perde o controle manual da simulação
        print('< Destruindo atores e sensores...')
        actors = [self.camera_sensor, self.collision_sensor, self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def defineDestiny(self,g d):
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
        return np.random.randint(QTD_ACOES)
    else:
        return action

def sample_memories(BATCH_SIZE):
    perm_batch = np.random.permutation(len(exp_buffer))[:BATCH_SIZE]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:, 0][0], mem[:, 1][0], mem[:, 2][0], mem[:, 3][0], mem[:, 4][0]

def func_erro(y_true, y_pred):
    #https://keras.io/api/losses/
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

def generateNetwork(nome='rede'):

    #considerar o cenário de colisão: caso ocorra,  
    camera = keras.Input(shape=(CAMERA_SIZE[0],CAMERA_SIZE[1],CAMERA_SIZE[2],), name='img1')
    #
    mid= layers.Conv2D(filters=256, kernel_size=3, activation='relu')(camera)
    mid=layers.MaxPooling2D((2,2))(mid) #400,300
    mid=layers.Conv2D(filters=256, kernel_size=3, activation='relu')(mid)
    mid=layers.MaxPooling2D((2,2))(mid) #200,150
    mid=layers.Conv2D(filters=128, kernel_size=3, activation='relu')(mid)
    mid=layers.MaxPooling2D((2,2))(mid) #100,75
    mid=layers.Conv2D(filters=64, kernel_size=3, activation='relu')(mid)
    mid=layers.MaxPooling2D((2,2))(mid) #50,32
    mid=layers.Flatten()(mid)
    dense1 = layers.Dense(32, activation='relu')(mid) #128
    #
    coord_faixas = keras.Input(shape=(4,), name='coordenadas-faixas-esq.-dir.')
    dense3 = layers.Dense(64, activation='relu')(coord_faixas)
    #
    velocidade = keras.Input(shape=(1,), name='velocidade')
    #
    x = layers.concatenate([dense1, dense3, velocidade]) #concatena, sem processar
    #
    outputs = layers.Dense(QTD_ACOES, activation='softmax')(x)
    #
    model = keras.Model(inputs=[camera, coord_faixas, velocidade], outputs=outputs, name=nome)
    model.compile(optimizer='adam', loss=func_erro, metrics=['accuracy'])
    return model

def connect(world, client, ip='localhost'):
    try:
        print('< Tentando conectar ao servidor Carla...')
        client = carla.Client(ip, 2000)
        client.set_timeout(10.0)
        print('< Conectado com sucesso.')
        #print('Carregando mapa Town05 [1/2] ...')        
        #client.load_world('Town05')
        #sleep(5)
        print('< Carregando mapa ...')        
        carla_world = World(client.get_world())
    except RuntimeError:
        print('Falha na conexão. Verifique-a.')
        return None, None
    os.system('clear')
    print('< Mapa carregado com sucesso !')
    return carla_world, client

def main():
    try:
        print('< Iniciando mundo Carla...')
        env = Env(world)
        #env.reset()
        print('< Inicializando DQN...')

        steps_train = 5  # a cada x passos irá treinar a main_network #50
        copy_steps = 1 # a cada x passos irá copiar a main_network para a target_network  #100
        global_step = 0  # passos totais (somatorio dos passos em cada episodio)
        start_steps = 1  # passos inicias. (somente após essa qtd de passo irá treinar começar a rede) #100

        print('< Gerando rede DQN principal...')
        mainQ = generateNetwork('mainQ')
        #mainQ.summary()
        print('< Gerando rede DQN alvo...')
        targetQ = generateNetwork('targetQ')
        print('< Redes geradas.')

        #inicializa variaveis
        action = 0 #ação escolhida
        print('< Iniciando episodios...')
        for episodio in range(QTD_EPISODIOS):
            env.reset() 
            
            print('>> Episodio', episodio, '%')
            world.tick()  # atualiza o mundo
            obs = env.getObservation()
            passos_ep = 0  # quantidade de passos realizados em um episodio
            actions_counter = Counter()
            episodic_loss = []

            while not env.done:
                print('> Passo ', global_step, ' - Início ')
                # Prediz uma ação (com base no que possui treinada) com base na observação - por enquanto, apenas uma imagem de câmera
                actions = mainQ.predict(obs)
                # A rede produz um resultado em %. Logo, escolhe a posição do vetor(ação) com maior probabilidade (argmax())
                action = np.argmax(actions)  # neste caso, argmax retorna a posição com maior probabilidade
                actions_counter[str(action)] += 1 #soma +1 em um histórico de ações realizadas por episódio

                ''' 
                Por mais que a rede tenha escolhido uma ação mais 'adequada'(maior probabilidade),
                o método da DQN usa a política de ganância, que verifica se deve usar a própria 
                experiência (rede DQN) ou se explora o ambiente com uma nova ação aleatória.
                '''

                action = epsilon_greedy(action, global_step)

                # Realiza a ação escolhida e recupera o próximo estado, a recompensa, info e se acabou(se houve colisão).
                next_obs, reward, done, info = env.step(action)
                
                # Armazenamento das experiências no buffer de replay
                exp_buffer.append([obs, action, next_obs, reward, done])
                print('> Terminou o PASSO com recompensa: ', reward, '')

                # Treino da rede principal, após uma qtd de passos, usando as experiências do buffer de replay.
                if global_step % steps_train == 0 and global_step > start_steps:
                    print('> Atualização da Q-Network %', steps_train, 'passos.')
                    # Retira uma amostra de experiências
                    obs, act, next_obs, reward, done = sample_memories(BATCH_SIZE)

                    # valor de probabilidade da ação mais provável
                    targetValues = targetQ.predict(next_obs.obs[0], next_obs.obs[1], \
                        next_obs.obs[2], next_obs.veloc, next_obs.coord_faixas)
                    print('> Valores alvo: ', targetValues)

                    bestAction = np.argmax(targetValues)
                    
                    print('> Melhor Ação (prediction): ', env.DICT_ACT[bestAction], '(',bestAction,').')

                    '''
                    se terminar o episódio, y=reward. (multiplica por 0 e sobra reward);
                    senão, reward + discount_factor * ...
                    '''
                    y = reward + discount_factor * np.max(targetValues) * (1 - done)

                    '''
                    print('Y atualizado: ', y)
                    print(
                        'Antes (melhor ação - rede): ',
                        targetValues[0][bestAction],
                        '| Depois (y): ',
                        y,
                    )
                    '''

                    #targetValues[0][bestAction] = y ###isso está errado
                    '''
                    now we train the network and calculate loss
                    gradient descent (x=obs e y=recompensa)
                    
                    o y vai aumentar a dimensão, passando a ser um vetor.
                    terá formato: [ 0,Y,0].

                    ao treinar, a rede dará foco na alteração dos pesos, focando o resultado na 'saída' Y. 
                    Que, na verdade, é a ação que melhor representa as observações obtidas.
                    ex.: uma imagem de um 2, é como resultado [0,0,1,0,...]
                    '''
                    training_history = mainQ.fit(x=[obs.obs[0], obs.obs[1], obs.obs[2], obs.veloc, obs.coord_faixas], \ 
                        y=np.expand_dims(y, axis=-1), batch_size=BATCH_SIZE, epochs=1, \
                        shuffle=True, callbacks=[checkpoint1, tensorboard_callback])
                    episodic_loss.append(training_history)  # historico
                    global_training_history.append(training_history)
                    #print(training_history.history.keys())
                    #print('Loss:', episodic_loss['loss'])
                    #print('Accuracy:', training_history.history['accuracy'])

                if (global_step + 1) % copy_steps == 0 and global_step > start_steps:
                    # Cópia dos pesos da rede principal para a rede alvo.
                    targetQ.set_weights(mainQ.get_weights())

                obs = next_obs  # troca a obs
                passos_ep += 1
                global_step += 1
            
            #colocar no lugar certo::
            with historico_episodio_summary.as_default():
                tf.summary.scalar('recompensa', env.reward, step=episodio)
                tf.summary.scalar('passos', passos_ep, step=episodio)
                tf.summary.scalar('contador de ações', actions_counter, step=episodio)


            print(
                '=> Fim do Episódio ',
                episodio,
                'com ',
                passos_ep,
                ' passos e recompensa total de: ',
                env.reward,
                ' pontos.\n',
            )
           

    except RuntimeError:
        print('\nRuntimeError. Trace: ')
        traceback.print_exc()
        pass
    except Exception:
        traceback.print_exc()
        pass
    finally:
        if world is not None:
            world.destroy()

'''
Conexão com o simulador
'''
world, client = connect(world, client , ip='192.168.100.11')

if __name__ == '__main__' and world != None and client != None:
    main()
