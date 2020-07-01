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
num_episodes = 100
batch_size = 32
learning_rate = 0.001
discount_factor = 0.97
# Variáveis de mundo e cliente
world = None
client = None


class Env(object):
    def __init__(self, world):
        self.observation = None
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

    def reset(self):
        self.reward = 0
        self.done = 0
        world.restart()
        self.player = world.player
        world.camera_sensor.listen(lambda img: world.convertImage(img, env=self))
        print("Ator resetado...")

    def applyReward(self):
        # se houve colisão, negativa em X pontos e termina
        if world.colission_history > 0:
            self.reward = self.reward - 10
            self.done = 1
            return

        # se esta perto do objetivo, perde menos pontos
        dist_atual = world.destiny_dist()
        self.reward = self.reward + (dist_atual - world.last_distance) / dist_atual
        world.last_distance = dist_atual

        if dist_atual <= 10:  # se esta a menos de 10unidades do destino, fim.
            self.reward = self.reward + 100
            self.done = 1
            return
        vel = world.velocAtual()

    def step(self, action):
        self.info = self.applyAction(action)
        world.carla_world.tick()  # atualiza o mundo
        self.applyReward()
        return self.observation, self.reward, self.done, self.info

    def applyAction(self, action):
        speed_limit = 10
        if world.velocAtual() >= speed_limit:
            throttle = 0.0
            print("\nAcima do limite de velocidade de ", speed_limit, "km/h...")
        else:
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

        print("\tAção: ", actions[action])
        self.player.apply_control(
            carla.VehicleControl(
                actions[action][0], actions[action][1], reverse=actions[action][2]
            )
        )
        return None

    def getObservation(self):
        return self.observation


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
        # Cria um audi tt no ponto primeiro waypoint dos spawns
        blueprint = self.carla_world.get_blueprint_library().find("vehicle.audi.tt")
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        # Spawn the player.
        spawn_point = (self.carla_world.get_map().get_spawn_points())[0]
        self.player = self.carla_world.try_spawn_actor(blueprint, spawn_point)
        # para nao comecar as ações sem ter iniciado adequadamente

        """
        prepara o carro
        """
        for i in range(100):
            self.carla_world.tick()

    def restart(self):
        # Set up the sensors.
        print("Reiniciando ...")
        self.destroy()
        self.config_carla_world()
        self.spawnPlayer()
        self.config_camera()
        self.config_collision_sensor()
        print("Fim do Reinício.")
        print("Iniciando componentes do ator...")

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
        bp = self.carla_world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.carla_world.spawn_actor(
            bp, carla.Transform(carla.Location(x=2.0, z=1.0)), attach_to=self.player
        )
        self.collision_sensor.listen(lambda event: self.on_collision(event))

    def on_collision(self, event):
        self.colission_history += 1
        print("\n|| Mais uma colisão... || ")

    def config_camera(self):
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

    def convertImage(self, image, env):
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
        env.observation = array  # repassa para o ambinte uma nova imagem da camera

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


def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:, 0][0], mem[:, 1][0], mem[:, 2][0], mem[:, 3][0], mem[:, 4][0]


def generateNetwork(scope):
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(camera_size_y, camera_size_x, 3)
        )
    )  # para usar imagem gray, tem que trocar o shape da imagem na layer. X e Y trocados
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="softmax"))
    model.add(layers.Dense(qtd_acoes))
    model.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
    return model


def connect(world, client, ip="localhost"):
    try:
        print("Tentando conectar ao servidor Carla...")
        client = carla.Client(ip, 2000)
        client.set_timeout(1.0)
        carla_world = World(client.get_world())
    except RuntimeError:
        print("Falha na conexão.")
        return None, None
    print("Conectado com sucesso.")
    return carla_world, client


def main():
    try:
        env = Env(world)
        print("Iniciando episodios...")
        env.reset()
        print("Inicializando DQN...")

        global_step = 0
        copy_steps = (
            10  # a cada x passos irá copiar a main_network para a target_network
        )
        steps_train = 5  # a cada x passos irá treinar a main_network
        start_steps = 20  # passos inicias . default=200

        print("Gerando rede DQN principal...")
        # mainQ = generateNetwork('mainQ')
        print("Gerando rede DQN alvo...")
        # targetQ = generateNetwork('targetQ')
        print("Redes geradas.")

        action = 0
        y = 0

        for i in range(num_episodes):
            print("\n%Episodio", num_episodes, "%")
            done = 0
            info = None
            world.carla_world.tick()  # atualiza o mundo
            obs = env.getObservation()
            passos_ep = 0  # quantidade de passos realizados em um episodio
            episodic_reward = 0
            actions_counter = Counter()
            episodic_loss = []

            while not done:
                print("| Passo(total)", global_step, " - Início. ")
                actions = mainQ.predict(obs)
                # get the action
                # escolhe a posicao com maior probabilidade
                action = np.argmax(actions)
                actions_counter[str(action)] += 1

                # select the action using epsilon greedy policy
                action = epsilon_greedy(action, global_step)

                # now perform the action and move to the next state, next_obs, receive reward
                next_obs, reward, done, info = env.step(action)

                # Store this transistion as an experience in the replay buffer
                exp_buffer.append([obs, action, next_obs, reward, done])
                print("Terminou o passo com recompensa: ", reward, " .|\n")

                # After certain steps, we train our Q network with samples from the experience replay buffer
                if global_step % steps_train == 0 and global_step > start_steps:
                    print("\n-- Atualização da Q-Network %100 passos --")
                    # sample experience
                    obs, act, next_obs, reward, done = sample_memories(batch_size)

                    # valor de probabilidade da ação mais provável
                    targetValues = targetQ.predict(next_obs)
                    print("\nValores alvo: ", targetValues)

                    bestAction = np.argmax(targetValues)
                    print("\nMelhor Ação: ", bestAction)

                    y = reward + discount_factor * np.max(targetValues) * (1 - done)
                    print("\nY atualizado: ", y)
                    print(
                        "\nAntes (melhor ação): ",
                        targetValues[0][bestAction],
                        "| Depois: ",
                        y,
                    )

                    targetValues[0][bestAction] = y
                    # now we train the network and calculate loss
                    # gradient descent (x=obs e y=recompensa)
                    train_loss = mainQ.fit(obs, targetValues)
                    episodic_loss.append(train_loss)  # historico
                    print("|| Loss Episódica: ", episodic_loss)

                if (global_step + 1) % copy_steps == 0 and global_step > start_steps:
                    # Cópia dos pesos da rede principal para a rede alvo.
                    targetQ.set_weights(mainQ.get_weights())

                obs = next_obs  # troca a u
                passos_ep += 1
                global_step += 1
                episodic_reward += reward
            print(
                "\nEpisódio ",
                num_episode,
                "com ",
                epoch,
                "passos and recompensa total: ",
                episodic_reward,
                ".",
            )

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

"""
Conexão com o simulador
"""
world, client = connect(world, client)  # , ip='35.197.43.29')

print("Pronto para iniciar.")

if __name__ == "__main__" and world != None and client != None:
    main()
else:
    print("Erro. Verifique a conexão.")
