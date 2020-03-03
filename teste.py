import glob
import os
import sys
from time import sleep
import math 
try:
    sys.path.append(glob.glob('/home/wenderson/Downloads/CARLA_0.9.7.3/PythonAPI/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg')[0])
    sys.path.append(glob.glob('../../PythonAPI/carla/')[0])
except IndexError:
    pass

import carla
import random
import time

# ==============================================================================
# -- Main ---------------------------------------------------------------
# ==============================================================================

def main():
	def veloc_atual():
		v = carro_sdc.get_velocity()
		c = carro_sdc.get_control()
		return int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

	def destroy():
		print('destroying actors')
		for actor in actor_list:
		    actor.destroy()
		print('done.')
	try:
		actor_list = []
		client = carla.Client('localhost', 2000)
		client.set_timeout(10.0)
		world = client.get_world()
		#world.set_weather(carla.WeatherParameters.WetCloudySunset)

		blueprint_library = world.get_blueprint_library()
		bp = (blueprint_library.find('vehicle.audi.tt'))
		if bp.has_attribute('color'):
			color = random.choice(bp.get_attribute('color').recommended_values)
			bp.set_attribute('color', color)

		# Now we need to give an initial transform to the vehicle. We choose a
		# random transform from the list of recommended spawn points of the map.
		transform = (world.get_map().get_spawn_points())[0]

		# So let's tell the world to spawn the vehicle.
		carro_sdc = world.spawn_actor(bp, transform)
		actor_list.append(carro_sdc)

		camera_bp = blueprint_library.find('sensor.camera.rgb')
		camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
		camera_bp.set_attribute('image_size_x', '800')
		camera_bp.set_attribute('image_size_y', '600')
		camera_bp.set_attribute('sensor_tick', '0.02') # Captura uma imagem a cada 50hz
		camera_bp.set_attribute('fov', '110') # angulo horizontal de 110graus

		camera = world.spawn_actor(camera_bp, camera_transform, attach_to=carro_sdc)
		actor_list.append(camera)
		print('created %s' % camera.type_id)

		#salvar as informações de velocidade, aceleracao, volante e freio no mesmo nome de imagem        
		#camera.listen(lambda image: image.save_to_disk('_out/%08d' % image.frame))

		transform.location += carla.Location(x=40, y=-3.2)
		transform.rotation.yaw = -180.0

		#cidade com apenas um veículo. mudar o range
		for _ in range(0, 0):
			transform.location.x += 8.0
			bp = random.choice(blueprint_library.filter('vehicle'))

		# This time we are using try_spawn_actor. If the spot is already
		# occupied by another object, the function will return None.
		npc = world.try_spawn_actor(bp, transform)
		if npc is not None:
			actor_list.append(npc)
			npc.set_autopilot()
			print('created %s' % npc.type_id)

		time.sleep(5)
		i = 5000

		pos_obj = carla.Location(x=9.9, y=0.3, z=20.3)

		while(i > 0):
			#as informações precisam ser atualizadas a cada movimento do carro, pois senão a imagem sai incorreta.
			#Entretanto, o certo seria ter uma função que rodasse infinitamente informando as condições do carro. 

			i = i - 1
			v = carro_sdc.get_velocity()
			c = carro_sdc.get_control()
			veloc_atual = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

			
			if veloc_atual >= 10:
				#carro_sdc.apply_control(carla.VehicleControl(throttle=0.2, steer=-1.0))
				carro_sdc.apply_control(carla.VehicleControl(throttle=0.2, steer=-1.0))
			else:
				carro_sdc.apply_control(carla.VehicleControl(throttle=0.5, steer=-1.0))
			

			i = i - 1
			v = carro_sdc.get_velocity()
			c = carro_sdc.get_control()
			veloc_atual = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
			print("passos restantes: ", i)
			print("Velocidade atual: %d km/h" % veloc_atual)
			
			pos = carro_sdc.get_location()
			print("Posição: ", pos)
			print("Distancia objetivo:", pos.distance(pos_obj))
			_ = os.system('clear') 

	except RuntimeError:
		pass
	finally:
		destroy()
		
	def recompensa(self):
		recompensa = recompensa + (veloc_atual/360)


if __name__ == '__main__':
	
	main()