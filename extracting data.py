import glob
import os
import sys
import random
import numpy as np
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import carla
import time 
import cv2
IMG_WIDTH=640
IMG_HEIGHT=480
def process(image):
    frame_n=image.frame_number
    im=np.array(image.raw_data)
    im=im.reshape((480,640,4))
    im=im[:,:,:3]
    cv2.imwrite(f'{frame_n}.png',im)

def process1(image):
    frame_n=image.frame_number
    im=np.array(image.raw_data)
    im=im.reshape((480,640,4))
    im=im[:,:,:3]
    cv2.imwrite(f'lab{frame_n}.png',im)


actor_list=[]
try:
    client=carla.Client('localhost',2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library=world.get_blueprint_library()
    bp=blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point=random.choice(world.get_map().get_spawn_points())
    vehicle=world.spawn_actor(bp,spawn_point)
    vehicle.set_autopilot(True)
    actor_list.append(vehicle)
    cam_bp=blueprint_library.find('sensor.camera.rgb')
    cam_bp1=blueprint_library.find('sensor.camera.semantic_segmentation')
    cam_bp.set_attribute('image_size_x',f"{IMG_WIDTH}")
    cam_bp.set_attribute('image_size_y',f"{IMG_HEIGHT}")
    cam_bp.set_attribute('fov','110')
    cam_bp1.set_attribute('image_size_x',f"{IMG_WIDTH}")
    cam_bp1.set_attribute('image_size_y',f"{IMG_HEIGHT}")
    cam_bp1.set_attribute('fov','110')

    spawn_point=carla.Transform(carla.Location(x=2.5,z=0.7))
    sensor=world.spawn_actor(cam_bp,spawn_point,attach_to=vehicle)
    sensor1=world.spawn_actor(cam_bp1,spawn_point,attach_to=vehicle)
    actor_list.append(sensor)
    actor_list.append(sensor1)
    sensor.listen(lambda image:process(image))
    sensor1.listen(lambda image: process1(image))
    time.sleep(1)

finally:
    for actor in actor_list:
        actor.destroy()
    print('all cleaned up!')
