'''
SCRIPT AVALIAÇÃO
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
import os
os.system('clear')
import io
import glob
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
from statistics import mean

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
CAMERA_SIZE = (150, 150, 3)
QTD_ACOES = 8
#
QTD_EPISODIOS = 100 #100
BATCH_SIZE = 32
learning_rate = 0.001
discount_factor = 0.9
# Variáveis de mundo e cliente
world = None
client = None
SPEED_LIMIT = 30
#MAX_PASSOS = 501
episodio = 0#apenas para salvar

#variaveis globais de histórico
historico_recompensa=[]
historico_lanes=[[],[]] #0:solid, 1:broken

# Define the Keras TensorBoard callback.
logdir='logs/' + datetime.now().strftime('%d-%m-%Y--%H:%M:%S')
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

#grafico de recompensa por época personalizado
log1 = summary.create_file_writer(logdir+'/historico_ep/log1')
recomp_media = summary.create_file_writer(logdir+'/historico_ep/recomp_media')
log_faixa = summary.create_file_writer(logdir+'/historico_ep/faixas')
log_colisao = summary.create_file_writer(logdir+'/historico_ep/colisoes')

def gerarGrafico(y, x, linear=False, grafico_greedy=False):
    #https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    figure = plt.figure()

    if(not linear):
        #print("< Gerando gráfico de ações...")    
        N = np.arange(len(x)) 
        plt.bar(N, y, label='Ações')

        plt.ylabel('Qtd. vezes realizada')
        plt.title('Contador de Ações')

        plt.xticks(N, rotation = 45, labels=x)
    elif grafico_greedy:
        N = np.arange(len(x)) 
        plt.bar(N, y, label='Escolha por episódio')

        plt.ylabel('Aleatória ou Predita')
        plt.title('Ação aleatória ou predita por episódio')
        plt.yticks(['Aleatória', 'Predita'])
    else:
        #print("< Gerando gráfico de velocidade...")
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

    #print('< Gráfico gerado.')
    return image

class Env(object):
    def __init__(self, world):
        self.reward = 0
        self.done = 0
        self.info = None
        self.player = None
        self.DICT_ACT = [
            'sem acao',
            'frente',
            'frente-esquerda',
            'frente-direita',
            'freio',
            'ré',
            'ré-esquerda',
            'ré-direita',
        ]
        self.image_queue = queue.Queue() 
        self.passos_ep = 0
        self.actions_counter = None
        self.tacografo = None
        self.ultima_posicao = None
        self.dist_percorrida = None
        self.distancia_re = 0
        self.distancia_re_total = 0
        self.freio_steps = 0

        '''
        Ações
        throttle=aceleração
        '''
        self.acelerador = 0.5
        self.ACTIONS = [
            #(trottle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False)
            (0.0, 0.0, 0.0, False, False), #sem acao
            (self.acelerador, 0.0, 0.0, False, False), #frente
            (self.acelerador, -0.5, 0.0, False, False), #frente-esquerda
            (self.acelerador, 0.5, 0.0, False, False), #frente-direita
            (0.0, 0.0, 1.0, False, False), #freio
            (self.acelerador, 0.0, 0.0, False,True), #ré
            (self.acelerador, -0.5, 0.0, False, True), #ré-esquerda
            (self.acelerador, 0.5, 0.0, False, True), #ré-direita
        ]  # 8 acoes

    def reset(self):
        print('< Reiniciando ambiente...')
        self.reward = 0
        self.done = 0
        self.passos_ep = 0
        world.restart()
        print('< Iniciando componentes do ator...')
        self.player = world.player
        print('< Info inicial player (None?): ', self.player)
        print('< Resetando a fila de imagens')
        del self.image_queue
        self.image_queue = queue.Queue()
        print('< Câmera iniciada.')
        world.camera_sensor.listen(self.image_queue.put)
        print('< Ator resetado.')
        self.actions_counter = np.zeros(shape=(QTD_ACOES), dtype=int)
        self.tacografo = []
        self.dist_percorrida = 0
        self.ultima_posicao = self.player.get_location() #posicao de spawn
        self.distancia_re = 0
        self.distancia_re_total = 0
        self.freio_steps = 0

    def applyReward(self, vel):
        print('> Aplicando recompensa...')
        # se houve colisão, negativa em X pontos e termina
        print('> Velocidade atual: ', vel, '\tkm/h.')
        #self.reward = self.reward + (1 - (vel/SPEED_LIMIT)**(0.4*vel)) #antiga função       
        if vel > 2 and vel < SPEED_LIMIT:
            self.reward = self.reward + 1
        elif vel > SPEED_LIMIT and vel <= 2:
            self.reward = self.reward - 1

        '''
        #limite de tempo (passos)
        if self.passos_ep >= MAX_PASSOS:
            self.reward = self.reward - 1 #evitar de ficar parado
            self.done = 1
        '''

        if self.distancia_re >= 3:
            print('> Andou muito de ré !')
            self.reward = self.reward - 1
            self.done = 1
        
        centroFaixa = world.map.get_waypoint(self.player.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving))
        dst = self.player.get_location().distance(centroFaixa.transform.location)
        
        if dst > 0.4 and dst < 2    :
            print('> Distância do centro da faixa: Fora.')
            self.reward = self.reward - 1
        elif dst > 2:
            print('> Distância do centro da faixa: Muito fora.', )
            self.reward = self.reward - 1
            self.done = 1
        else:
            print('> Distância do centro da faixa: Dentro')
            self.reward = self.reward + 1
            
        if world.lane_invasion[0] > 0: #solid lane
            self.reward = self.reward - 1
            self.done = 1
        
        if world.colission_history > 0:
            self.reward = self.reward - 1
            self.done = 1
        
        if self.freio_steps >= 100:
            self.reward = self.reward - 1
            self.done = 1

    def calcularDistPercorrida(self, re):
        pos = self.player.get_location()
        dist_perc = pos.distance(self.ultima_posicao)
        self.ultima_posicao = pos
        #atualiza a distancia percorrida
        self.dist_percorrida += dist_perc
        if re:
            self.distancia_re += dist_perc #distancia em re do momento
            self.distancia_re_total += dist_perc #distancia em re total
            
    def step(self, action):
        vel = world.velocAtual() #velocidade do veiculo
        self.passos_ep = self.passos_ep + 1 #atualiza os passos do ep
        self.tacografo.append(vel) #tacografo
        self.actions_counter[action] += 1 #soma +1 em um histórico de ações realizadas por episódio
        #
        self.info = self.applyAction(action) #guarda o nome da ação realizada
        world.tick()  # atualiza o mundo
        
        if action == 0 or action == 4:
            self.freio_steps += 1
        else:
            self.freio_steps = 0

        if action == 5 or action == 6 or action == 7:
            dist = self.calcularDistPercorrida(re=True)
        else:
            self.distancia_re = 0
            dist = self.calcularDistPercorrida(re=False)
        print('< Distância percorrida TOTAL: ', self.dist_percorrida)
        print('< Distância percorrida RÉ (momento): ', self.distancia_re)
        print('< Distância percorrida RÉ (total): ', self.distancia_re_total)
        self.applyReward(vel=vel)    

        return self.getObservation(), self.reward, self.done, self.info

    def applyAction(self, action):
        print('> Ação: \t', self.DICT_ACT[action], '\t', self.ACTIONS[action])
        
        self.player.apply_control(carla.VehicleControl(*(self.ACTIONS[action]))) #* é para fazer o unpack da tuple para parametros
        return self.DICT_ACT[action]

    def getObservation(self):
        print('< Esperando imagem ...')
        return Observation(self.getFila(p=1), self.getFila(p=2), self.getFila(p=3), veloc=world.velocAtual())

    def convertImage(self, image):
        global episodio
        '''
        é realizado automaticamente 
        quando há uma nova imagem disponível pelo sensor camera_sensor.listen
        '''
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        image.save_to_disk('imagens/ep[%03d]-frame[%08d]' % (episodio ,image.frame))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1] #inverte a ordem das camadas RGB
        array = np.expand_dims(array, axis=0)
        return array
    
    def getFila(self, p=1):
        world.tick()
        img = self.image_queue.get()
        return self.convertImage(img)
    
class Observation(object):
    def __init__(self, img1, img2, img3, veloc):
        print('< Criando nova Observation.')
        if img1 is not None and img2 is not None and img3 is not None:
            self.img1 = img1
            self.img2 = img2
            self.img3 = img3
        if veloc is not None:
            self.veloc = np.expand_dims([veloc], axis=0) #veloc media entre as 3 obs ? Não.

    def retornaObs(self):
        return [self.img1, self.img2, self.img3, self.veloc]

class World(object):
    def __init__(self, carla_world):
        '''
        Incialização das variáveis
        '''
        self.carla_world = carla_world
        self.player = None
        self.map = carla_world.get_map()
        self.collision_sensor = None
        self.lane_sensor = None
        self.camera_sensor = None
        #self.destiny = carla.Location(x=9.9, y=0.3, z=20.3)
        #self.last_distance = 0
        self.colission_history = 0
        self.lane_invasion = [0,0]
        # restart do world é feito pelo Enviroment

    def spawnPlayer(self):
        self.player = None
        # Cria um audi tt no ponto primeiro waypoint dos spawns
        blueprint = self.carla_world.get_blueprint_library().find('vehicle.audi.tt')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'ego')
        # Spawn the player.
        print('< Convocando carro-EGO...')
        while(self.player == None):
            print("< Procurando ponto de spawn para o veículo...")
            spawn_point = random.choice((self.carla_world.get_map().get_spawn_points()))
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
        self.config_lane_sensor()
        self.colission_history = 0
        self.lane_invasion = [0,0]
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
        print('< Sensor detetou uma colisão.')
        print('> || OCORREU UMA COLISÃO ! || ')
        self.colission_history += 1
        #adicionar subir na calçada como colisao
        
    def config_lane_sensor(self):
        print('< Configurando sensor invasão de faixa ...')
        bp = self.carla_world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_sensor = self.carla_world.spawn_actor(bp, carla.Transform(), attach_to=self.player)
        self.lane_sensor.listen(lambda event: self.on_invasion(event))

    def on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        print(lane_types)
        event_type = [str(x).split()[-1] for x in lane_types]
        print(event_type)

        for i in event_type:
            if 'Solid' in event_type:
                self.lane_invasion[0] += 1
                historico_lanes[0][-1]+= 1 #soma invasão de faixa contínua
                print('< Invadiu uma linha contínua.')

            if 'Broken' in event_type and 'Solid' not in event_type:
                self.lane_invasion[1] += 1
                historico_lanes[1][-1]+= 1 #soma invasão de faixa seccionada
                print('< Invadiu uma linha seccionada.')

        
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
        actors = [self.camera_sensor, self.collision_sensor, self.player, self.lane_sensor]
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

def connect(world, client, ip='localhost'):
    try:
        print('< Tentando conectar ao servidor Carla...')
        client = carla.Client(ip, 2000)
        client.set_timeout(1000.0)
        print('< Conectado com sucesso.')
        #print('Carregando mapa Town05 [1/2] ...')        
        #client.load_world('Town05')
        #sleep(5)
        print('< Carregando mapa ...')        
        carla_world = World(client.get_world())
    except RuntimeError:
        print('<< Falha na conexão. Verifique-a.')
        return None, None
    
    print('< Mapa carregado com sucesso !')
    return carla_world, client

def func_erro(y_true, y_pred):
    #https://keras.io/api/losses/
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

def main():
    mainQ = None
    carregarModelo = True
    global episodio
    try:
        print('< Iniciando mundo Carla...')
        env = Env(world)
        print('< Inicializando DQN...')

        global_step = 0  # passos totais (somatorio dos passos em cada episodio) 

        if carregarModelo:
            print('< Carregando modelo ...')
            #mainQ = keras.models.load_model('redetcc.hdf5',custom_objects={'loss': func_erro})
            mainQ = keras.models.load_model('redetcc.hdf5', compile=False)

        else:
            print("> Sem modelo !")
            exit()     
        

        #inicializa variaveis
        action = 0 #ação escolhida
        print('< Iniciando episodios...')
        while episodio < QTD_EPISODIOS:
        #for episodio in range(QTD_EPISODIOS):
            env.reset()
            
            print('>> Episodio', episodio, '<<')
            print('< Iniciando gravação.')
            client.start_recorder("gravacao_ep_{0}.log".format(episodio))
           
            #world.tick()  # atualiza o mundo
            obs = env.getObservation()
            passos_ep = 0  # quantidade de passos realizados em um episodio
            
            #novas posicoes para gravar o historico de faixas
            historico_lanes[0].append(0)
            historico_lanes[1].append(0)

            while not env.done:
                print( '\n> Passo ', env.passos_ep,' (ep ',episodio,')  [Passo Global:', global_step, ']:')
                # Prediz uma ação (com base no que possui treinada) com base na observação - por enquanto, apenas uma imagem de câmera
                actions = mainQ.predict(x=obs.retornaObs())
                # A rede produz um resultado em %. Logo, escolhe a posição do vetor(ação) com maior probabilidade (argmax())
                action = np.argmax(actions)  # neste caso, argmax retorna a posição com maior probabilidade
                              
                # Realiza a ação escolhida e recupera o próximo estado, a recompensa, info e se acabou(se houve colisão).
                next_obs, reward, done, info = env.step(action)
                
                print('> Terminou o PASSO com recompensa: ', reward, '')
                
                obs = next_obs  # troca a obs
                passos_ep += 1
                global_step += 1
                
            historico_recompensa.append(env.reward)
            with log1.as_default():
                summary.scalar('Recompensa', historico_recompensa[-1], step=episodio)
                summary.image('Contador de Ações', gerarGrafico(x=env.DICT_ACT, y=env.actions_counter), step=episodio)
                summary.image('Tacógrafo', gerarGrafico(x=[x for x in range(passos_ep)], y=env.tacografo, linear=True), step=episodio)
                summary.image('Última imagem do episódio', obs.img1,  step=episodio)
                
            with log_faixa.as_default():
                summary.scalar('Faixa Contínua', historico_lanes[0][-1], step=episodio)
                summary.scalar('Faixa Seccionada', historico_lanes[1][-1], step=episodio)

            with recomp_media.as_default():
                summary.scalar('Recompensa', mean(historico_recompensa), step=episodio) #recompensa média
                
            with log_colisao.as_default():
                summary.scalar('Quantidade de Passos do Episódio', passos_ep, step=episodio)
                summary.scalar('Colisões', world.colission_history, step=episodio)
                summary.scalar('Distância Percorrida TOTAL (metros por episódio)', env.dist_percorrida, step=episodio)
                summary.scalar('Distância Percorrida de RÉ-TOTAL', env.distancia_re_total, step=episodio, description='metros por episódio')
            
            print('< Fim da escrita do histórico geral')
            print(
                '=> Fim do Episódio ',
                episodio,
                'com ',
                passos_ep,
                ' passos e recompensa total de: ',
                env.reward,
                ' pontos.\n',
            )

            print('< Finalizando a gravação atual...')
            client.stop_recorder()
            print('< Fim da gravação atual.')
            episodio += 1

        print("\n< Fim do treinamento. Acabaram os episódios.")

    except RuntimeError:
        print('\nRuntimeError. Trace: ')
        traceback.print_exc()
        pass
    except Exception:
        traceback.print_exc()
        pass
    finally:
        print('< Finalizando o programa...')
        print('< Parando gravação.')
        client.stop_recorder()
        if world is not None:
            world.destroy()
        print('<< \t Programa finalizado. >>')
        
'''
Conexão com o simulador
'''

world, client = connect(world, client , ip='192.168.100.11')

if __name__ == '__main__' and world != None and client != None:
    main()


