'''
Créditos do código base da DQN retirado do livro : 'Hands-On Reinforcement Learning with Python: Master reinforcement 
    and deep reinforcement learning using OpenAI Gym and TensorFlow', 2018. Ravichandiran, Sudharsan.

DQN modificada por Wenderson Souza para utilizar a framework Keras.
Ambiente preparado por Wenderson Souza, baseado no código base da OpenGymIA e códigos de exemplos fornecidos pela 
    documentação do simulador Carla

Fontes:
    https://carla.readthedocs.io/en/latest/python_api/
    https://carla.readthedocs.io/en/latest/core_concepts/
    https://pythonprogramming.net/reinforcement-learning-agent-self-driving-autonomous-cars-carla-python/
'''


import io
import glob
import os
import sys
from time import sleep
import math
from random import randrange
import random
#import time
import numpy as np
import pdb
import traceback
from carla import ColorConverter as cc
from collections import deque, Counter
from datetime import datetime
import queue

import carla
try:
    sys.path.append(glob.glob('../../carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg')[0])
    sys.path.append(glob.glob('../../PythonAPI/carla/')[0])
except IndexError:
    pass

#print('< Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))

import tensorflow as tf
from tensorflow import keras
from tensorflow import summary

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')

import time

'''
inicio = time.time()
time.sleep(0.1)
fim = time.time()
print(fim - inicio)
exit(0)
'''

'''
Para consertar o erro CUDNN_STATUS_INTERNAL_ERROR
'''

config = ConfigProto()
config.gpu_options.allow_growth = True
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus is not None:
    tf.config.experimental.set_memory_growth(gpus[0], True)
session = InteractiveSession(config=config)


# Variáveis Simulacao
CAMERA_SIZE = (800, 600, 3)
QTD_ACOES = 8
# Variáveis DQN
epsilon = -1 #apenas inicializada
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 1000 #a cada 2 episodios (500*5)
#
buffer_len = 20000  # limita a quantidade de experiencias. quando encher, retira as ultimas experiencias
exp_buffer = deque(maxlen=buffer_len)
QTD_EPISODIOS = 25 #100
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

filepath='checkpoints/redeCheckpoint.{epoch:02d}-{accuracy:.2f}.hdf5'
checkpoint1 = ModelCheckpoint(filepath, monitor='accuracy', save_best_only=True, verbose=1, mode='max', save_weights_only=False)

#grafico de recompensa por época personalizado
historico_episodio_summary = summary.create_file_writer(logdir+'/historico_ep')

global_training_history = [] #armazena os resultados do fit.

def salvarModeloReward(acc, reward, model, ep):
    print("> Avaliando recompensa..")
    #print("acc: ", acc[0])
    files = glob.glob('/home/wenderson/projeto_sdc/checkpoints/*')
    reward_last_model = -9999
    model_file = ''
    #print(files)
    print(files)
    for f in files:
        if 'reward:' in f:
            print('file f:', f)
            reward_model = float(f.split('--')[-1].split(':')[-1].split('.')[0])
            print('last reward :', reward_model)
            if reward_model > reward_last_model:
                print('trocou os modelos > melhor recompensa do ultimo modelo:', reward_model)
                reward_last_model = reward_model
                model_file = f
                break
            
    
    #print('recompensa atual:', reward)
    if float(reward_last_model) < reward:
        if model_file:
            os.remove(model_file)
        filepath='checkpoints/redeCheckpoint--ep:{}--acc:{:.2f}--reward:{}.hdf5'.format(ep, acc[0], reward)
        model.save(filepath, overwrite=True)
        print('> Modelo com melhor recompensa re-escrito.')
    else:
        print('> Modelo com melhor recompensa ainda é o último escrito.')

    if ep == QTD_EPISODIOS-1: #salva o útlimo modelo treinado, antes de acabar o treino
        filepath='checkpoints/LastredeCheckpoint--ep:{}--acc:{:.2f}--reward:{}.hdf5'.format(ep, acc[0], reward)
        model.save(filepath)


def gerarGrafico(y, x, linear=False):
    #https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    figure = plt.figure()

    if(not linear):
        print("< Gerando gráfico de ações...")    
        ind = np.arange(len(x)) 
        plt.bar(ind, y, label='Ações')

        plt.ylabel('Qtd. vezes realizada')
        plt.title('Contador de Ações')

        plt.xticks(ind, rotation = 45, labels=x)
    else:
        print("< Gerando gráfico de velocidade...")
        plt.plot(x, y, color='orange')
        #plt.scatter(x, y, color='blue')
        plt.ylabel('Velocidade')
        plt.title('Tacógrafo')

    plt.tight_layout()

    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    #print('salvando imagem')
    plt.savefig(buf, format='png')
    #print('imagem salva')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    
    plt.close(figure)
    #print('imagem fechada')
    buf.seek(0)
    #print("usando o buffer")
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)
    print('< Gráfico gerado.')
    return image

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
            'freio',
            'ré',
            'ré-direita',
            'ré-esquerda',
        ]
        #self.frameObs = None
        self.image_queue = queue.Queue() #jackpot: toda vez que ocorre um .get(), ocorre um tick
        #self.radar_queue = queue.Queue() #jackpot: toda vez que ocorre um .get(), ocorre um tick
        self.passos_ep = 0
        self.actions_counter = None
        self.tacografo = None
        self.dist_percorrida = None
        #self.coord_faixas = []
        self.ultima_posicao = None

        '''
        Ações
        throttle=aceleração
        '''
        self.acelerador = 0.5
        self.ACTIONS = (
            (throttle=0.0, steer=0.0, brake=0.0, reverse=False), #sem acao
            (throttle=self.acelerador, steer=0.0, brake=0.0, reverse=False), #frente
            (throttle=self.acelerador, steer=-0.5, brake=0.0, reverse=False), #frente-direita
            (throttle=self.acelerador, steer=0.5, brake=0.0, reverse=False), #frente-esquerda
            (throttle=0.0, steer=0.0, brake=0.0, reverse=True), #freio
            (throttle=self.acelerador, steer=0.0, brake=0.0, reverse=True), #ré
            (throttle=self.acelerador, steer=-0.5, brake=0.0, reverse=True), #ré-direta
            (throttle=self.acelerador, steer=0.5, brake=0.0, reverse=True), #ré-esquerda
        )  # 8 acoes

    def reset(self):
        print('< Reiniciando ambiente...')
        self.reward = 0
        self.done = 0
        self.passos_ep = 0
        world.restart()
        print('< Iniciando componentes do ator...')
        self.player = world.player
        print('< Info inicial player (None?): ', self.player)
        print('< Câmera iniciada.')
        #world.camera_sensor.listen(lambda img: self.convertImage(img))
        world.camera_sensor.listen(self.image_queue.put)
        print('< Ator resetado.')
        self.actions_counter = np.zeros(shape=(QTD_ACOES), dtype=int)
        self.tacografo = []
        self.dist_percorrida = 0.0
        self.ultima_posicao = self.player.get_location() #posicao de spawn

    def applyReward(self, vel):
        # se houve colisão, negativa em X pontos e termina
        if world.colission_history > 0:
            print('> || OCORREU UMA COLISÃO ! || ')
            self.reward = self.reward - 100
            self.done = 1
            return 

        #não tenho noção de velocidade...
        print('> Velocidade atual: ', vel, '\tkm/h.')
        #vel <=1: reward += 0.01
        
        if vel > 3 and vel < SPEED_LIMIT:
            self.reward = self.reward + 1
        elif vel > SPEED_LIMIT:
            self.reward = self.reward - 1

        #limite de tempo (passos)
        if self.passos_ep >= 501: #1001
            self.reward = self.reward - 10
            self.done = 1
            return
        
        #self.reward = self.reward + (1 - (vel/SPEED_LIMIT)**(0.4*vel))
        
        '''Se demora muito para se locomover e juntar recompensa, perde do mesmo jeito.
        O objetivo aqui é fazer o veículo se locomover para acumular recompensa.
        '''
       
        '''
        if self.coord_faixas not None:
            self.reward = self.reward + (-(((CAMERA_SIZE[1]/2)-self.coord_faixas[3][1]) ** 4)+1)

        #(esq0, esq1, dir0, dir1)
        if self.coord_faixas is not None:
            #se o X/2 (metade da tela), está entre as linhas, +1. senão, -1
            if CAMERA_SIZE[1]/2: 
                0
            #  self.reward = self.reward + (-(((CAMERA_SIZE[1]/2)-self.coord_faixas[3][1]) ** 4)+1)
        '''

        '''
        #se der a mesma quantidade de passos para frente e para trás, perde ponto
        #NÃO ESTÁ IMPLEMENTADO
        
        for acao in range(self.actions_counter.size) 
            frente += self.actions_counter[acao]
            tras += self.actions_counter[acao]

        '''
        

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
    def calcularDistPercorrida(self):
        pos = self.player.get_location()
        dist_perc = pos.distance(self.ultima_posicao)
        self.ultima_posicao = pos
        #atualiza a distancia percorrida 
        self.dist_percorrida += dist_perc     
        
    def step(self, action):
        vel = world.velocAtual() #velocidade do veiculo
        self.passos_ep = self.passos_ep + 1 #atualiza os passos do ep
        self.tacografo.append(vel) #tacografo
        self.actions_counter[action] += 1 #soma +1 em um histórico de ações realizadas por episódio
        #
        self.info = self.applyAction(action) #guarda o nome da ação realizada
        world.tick()  # atualiza o mundo
        '''
        ao dar 3 ticks, são salvas 3 imagens na queue
        ''' 
        #print('Frame recebido (step): ', self.frameObs)
        self.applyReward(vel=vel)
        self.calcularDistPercorrida()
        
        return self.getObservation(), self.reward, self.done, self.info

    def applyAction(self, action):
        print('> Ação: \t', self.DICT_ACT[action]) 
        self.player.apply_control(carla.VehicleControl(self.ACTIONS[action]))
        return self.DICT_ACT[action]

    def getObservation(self):
        print('< Esperando imagem ...')
        
        #return Observation(self.image_queue.get(), veloc = world.velocAtual(), coord_faixas=None)
        #return Observation(self.image_queue.get(), veloc = world.velocAtual())
        return Observation(self.getFila(), veloc = world.velocAtual())
        
        #for i in range(3):
        '''
        #nao está usando --
        deve receber as três imagens e concatenar
        '''
        #    obs.append(self.image_queue.get())
        #return obs

    def convertImage(self, image):
        '''
        é realizado automaticamente 
        quando há uma nova imagem disponível pelo sensor camera_sensor.listen
        '''
        # image.convert(cc.Raw)
        print('< Processado e convertendo imagem.')
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1] #inverte a ordem das camadas RGB
        array = np.expand_dims(array, axis=0)
        #print('< Colocando imagem na fila ...')    
        #self.image_queue.put(array)
        #print('< Imagem colocada na fila')
        print('< Imagem convertida. Retornando.')
        return array
    
    def getFila(self):
        print('< Retirando imagem da fila...')
        img = self.image_queue.get()
        return self.convertImage(img)
    
class Observation(object):
    #def __init__(self, obs1, obs2, obs3, veloc, coord_faixas):
    #def __init__(self, img, coord_faixas, veloc):
    def __init__(self, img, veloc):
        print('< Criando nova Observation.')
        if img is not None:
            self.img1 = img
        if veloc is not None:
            self.veloc = np.expand_dims([veloc], axis=0) #veloc media entre as 3 obs ?
        '''
        print(self.img1.shape)
        if self.img1 is not None and coord_faixas is None:
            self.coord_faixas = self.calculaCoordFaixa(self.img1) 
        else: 
            print('fail coord faixas')
            self.coord_faixas = None
        print(self.coord_faixas.shape)
        
    def calculaCoordFaixa(self, img):
        #extract roi etc
        return np.expand_dims([0,0,0,0], axis=0)

    def calculaCentroFaixa(self, coord_faixas):
        return None
        
    def printShape(self):
        return '{0}\n{1}\n{2}'.format(self.img1.shape, self.coord_faixas.shape, self.veloc.shape)
        '''

    def retornaObs(self):
        return [self.img1, self.veloc]

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
        print('< Esquentando o carro-ego ...') #
        for i in range(20):
            self.tick()
        print('< Carro-ego preparado.')

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
        self.colission_history = 0
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
        print('< Ação aleatória.')
        return np.random.randint(QTD_ACOES)
    else:
        print('< Ação predita/prevista.')
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
    camera = keras.Input(shape=(CAMERA_SIZE[1],CAMERA_SIZE[0],CAMERA_SIZE[2]), name='img1')
    #
    mid= layers.Conv2D(filters=64, kernel_size=3, activation='relu')(camera)
    mid=layers.MaxPooling2D((2,2))(mid) #400,300
    mid=layers.Conv2D(filters=32, kernel_size=3, activation='relu')(mid)
    mid=layers.MaxPooling2D((2,2))(mid) #200,150
    mid=layers.Conv2D(filters=16, kernel_size=3, activation='relu')(mid)
    mid=layers.MaxPooling2D((2,2))(mid) #100,75
    mid=layers.Conv2D(filters=8, kernel_size=3, activation='relu')(mid)
    mid=layers.MaxPooling2D((2,2))(mid) #50,32
    mid=layers.Flatten()(mid)
    dense1 = layers.Dense(32, activation='relu')(mid) #128
    #
    #coord_faixas = keras.Input(shape=(4), name='coordenadas-faixas-esq.-dir.')
    #dense3 = layers.Dense(64, activation='relu')(coord_faixas)
    #
    velocidade = keras.Input(shape=(1), name='velocidade')
    #
    #x = layers.concatenate([dense1, dense3, velocidade]) #concatena, sem processar
    x = layers.concatenate([dense1, velocidade]) #concatena, sem processar
    #
    outputs = layers.Dense(QTD_ACOES, activation='softmax')(x)
    #
    #model = keras.Model(inputs=[camera, coord_faixas, velocidade], outputs=outputs, name=nome)
    model = keras.Model(inputs=[camera, velocidade], outputs=outputs, name=nome)
    #
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
        print('<< Falha na conexão. Verifique-a.')
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

        media_recompensa = 0
        total_recompensa = 0

        steps_train = 250  # a cada x passos irá treinar a main_network #250
        copy_steps = 250 # a cada x passos irá copiar a main_network para a target_network  #250
        global_step = 0  # passos totais (somatorio dos passos em cada episodio) 
        start_steps = 100  # passos inicias. (somente após essa qtd de passo irá treinar começar a rede) #100

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
            
            print('>> Episodio', episodio, '<<')
            world.tick()  # atualiza o mundo
            obs = env.getObservation()
            passos_ep = 0  # quantidade de passos realizados em um episodio
            
            episodic_loss = []

            while not env.done:
                print( '\n> Passo ', env.passos_ep,' (ep ',episodio,'):')
                # Prediz uma ação (com base no que possui treinada) com base na observação - por enquanto, apenas uma imagem de câmera
                actions = mainQ.predict(x=obs.retornaObs())
                # A rede produz um resultado em %. Logo, escolhe a posição do vetor(ação) com maior probabilidade (argmax())
                action = np.argmax(actions)  # neste caso, argmax retorna a posição com maior probabilidade
                
                ''' 
                Por mais que a rede tenha escolhido uma ação mais 'adequada'(maior probabilidade),
                o método da DQN usa a política de ganância, que verifica se deve usar a própria 
                experiência (rede DQN) ou se explora o ambiente com uma nova ação aleatória.
                '''

                action = epsilon_greedy(action, global_step)
                
                # Realiza a ação escolhida e recupera o próximo estado, a recompensa, info e se acabou(se houve colisão).
                next_obs, reward, done, info = env.step(action)
                
                # Armazenamento das experiências no buffer de replay
                print('> Guardando experiências buffer de replay...')
                exp_buffer.append([obs, action, next_obs, reward, done])
                print('> Terminou o PASSO com recompensa: ', reward, '')

                # Treino da rede principal, após uma qtd de passos, usando as experiências do buffer de replay.
                if global_step % steps_train == 0 and global_step > start_steps:
                    print('> Atualização da Q-Network %', steps_train, 'passos.')
                    # Retira uma amostra de experiências
                    obs, act, next_obs, reward, done = sample_memories(BATCH_SIZE)

                    # valor de probabilidade da ação mais provável
                    targetValues = targetQ.predict(x=next_obs.retornaObs())
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

                    '''
                    now we train the network and calculate loss
                    gradient descent (x=obs e y=recompensa)
                    
                    o y vai aumentar a dimensão, passando a ser um vetor.
                    terá formato: [ 0,Y,0].

                    ao treinar, a rede dará foco na alteração dos pesos, focando o resultado na 'saída' Y. 
                    Que, na verdade, é a ação que melhor representa as observações obtidas.
                    ex.: uma imagem de um 2, é como resultado [0,0,1,0,...]
                    '''
                    training_history = mainQ.fit(x=next_obs.retornaObs(), y=np.expand_dims(y, axis=-1), batch_size=BATCH_SIZE, epochs=1, \
                        shuffle=True, callbacks=[checkpoint1, tensorboard_callback])
                                        
                    episodic_loss.append(training_history)  # historico
                    global_training_history.append(training_history)
                    #print(training_history.history.keys())
                    #print('Loss:', episodic_loss['loss'])
                    #print('Accuracy:', training_history.history['accuracy'])

                if (global_step + 1) % copy_steps == 0 and global_step > start_steps:
                    # Cópia dos pesos da rede principal para a rede alvo.
                    print('< Copiando pesos da rede principal para a rede alvo.')
                    targetQ.set_weights(mainQ.get_weights())

                obs = next_obs  # troca a obs
                passos_ep += 1
                global_step += 1
                
            with historico_episodio_summary.as_default():
                summary.scalar('Recompensa', env.reward, step=episodio)
                summary.scalar('Passos', passos_ep, step=episodio)
                summary.scalar('Distância Percorrida', env.dist_percorrida, step=episodio)
                #summary.histogram('contador de ações1(histograma)', actions_counter, step=episodio)
                summary.image('Contador de Ações', gerarGrafico(x=env.DICT_ACT, y=env.actions_counter), step=episodio)
                summary.image('Tacógrafo', gerarGrafico(x=[x for x in range(passos_ep)], y=env.tacografo, linear=True), step=episodio)
                print('< Fim da escrita do historico')

            #avalia recomepnsa e verifica se precisa substituir o modelo
            if len(global_training_history) > 0:
                salvarModeloReward(acc=global_training_history[-1].history['accuracy'], reward=env.reward, model=mainQ, ep=episodio) 

            print(
                '=> Fim do Episódio ',
                episodio,
                'com ',
                passos_ep,
                ' passos e recompensa total de: ',
                env.reward,
                ' pontos.\n',
            )
        print("\n< Fim do treinamento. Acabaram os episódios.")

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


