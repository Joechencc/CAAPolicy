from pathlib import Path
from queue import Queue
from datetime import datetime
import yaml


import carla
import random
import pygame
import time

from utils.ply2voxel import voxelization_save, align_from_path_save


def init_pygame():
    pygame.init()
    size = (200, 200)
    pygame.display.set_mode(size)
def save_unit_data(sensor_data_frame,cur_save_path,ticks,lidar_specs):

    data_frame = sensor_data_frame
    ticks = int(ticks/5-1)
    cur_save_path = Path(cur_save_path)  # 确保 cur_save_path 是 Path 对象

    for sensor in data_frame.keys():
        # if sensor.startswith('rgb'):
        #     # _, image = self.image_process(self.target_parking_goal, cam_id=sensor, image=data_frame[sensor])
        #     # image = Image.fromarray(image)
        #     # image.save(str(cur_save_path / sensor / ('%04d.png' % index)))
        #     data_frame[sensor].save_to_disk(
        #         str(cur_save_path / sensor / ('%04d.png' % index)))
        if sensor.startswith('depth') and not sensor.startswith("depth_e2e"):
            data_frame[sensor].save_to_disk(
                str(cur_save_path / sensor / (f"{ticks:04}.png")))
        if sensor.startswith('camera'):
            data_frame[sensor].save_to_disk(
                str(cur_save_path / sensor / (f"{ticks:04}.png")))
        elif sensor.startswith('lidar'):
            data_frame[sensor].save_to_disk(
                str(cur_save_path / sensor / (f"{ticks:04}.ply")))
    align_from_path_save(cur_save_path,lidar_specs,ticks )


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_name))
def spawn_semantic_lidar(world,vehicle,lidar_id,lidar_specs,sensor_list,sensor_queue):
    blueprint_library = world.get_blueprint_library()
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('rotation_frequency', str(lidar_specs['rotation_frequency']))
    lidar_bp.set_attribute('points_per_second', str(lidar_specs['points_per_second']))
    lidar_bp.set_attribute('channels', str(lidar_specs['channels']))
    lidar_bp.set_attribute('upper_fov', str(lidar_specs['upper_fov']))
    lidar_bp.set_attribute('lower_fov', str(lidar_specs['lower_fov']))
    lidar_bp.set_attribute('range', str(lidar_specs['range']))
    lidar_bp.set_attribute("horizontal_fov", str(lidar_specs['horizontal_fov']))

    lidar_location = carla.Location(x=lidar_specs['x'], y=lidar_specs['y'], z=lidar_specs['z'])
    lidar_rotation = carla.Rotation(pitch=lidar_specs['pitch'], roll=lidar_specs['roll'], yaw=lidar_specs['yaw'])
    lidar_transform = carla.Transform(lidar_location, lidar_rotation)

    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle,
                                       attachment_type=carla.AttachmentType.Rigid)

    lidar.listen(lambda data: sensor_callback(data, sensor_queue, lidar_id))
    sensor_list.append(lidar)

def process_sensor_data(data):
    print(f"处理了数据：{data.frame}")
def configure_traffic_manager(client, global_distance=2.0, global_sensitivity=1.0):
    """
    Configure the traffic manager settings for vehicle behavior in the simulation.

    :param client: Carla client object.
    :param global_distance: Global safe distance to leading vehicle.
    :param global_sensitivity: Global driving sensitivity.
    """
    # 获取交通管理器实例，默认端口8000
    traffic_manager = client.get_trafficmanager(8000)

    # 设置全局车辆间的安全距离
    traffic_manager.set_global_distance_to_leading_vehicle(global_distance)

    # 设置驾驶敏感度（0.0 = 最不敏感，1.0 = 最敏感）
    traffic_manager.global_percentage_speed_difference(global_sensitivity)
    return traffic_manager
def update_spectator_to_vehicle(world, vehicle, offset=carla.Location( z=2)):
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator_transform = carla.Transform(transform.location + offset, transform.rotation)
    spectator.set_transform(spectator_transform)

def check_for_h_key():
    toggle = False
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
            toggle = True
    return toggle

def try_spawn_vehicle(world, blueprint, spawn_point, retries=5):
    for _ in range(retries):
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is not None:
            return vehicle
    return None

def main():
    init_pygame()
    ############ setup world  ###############################
    num_NPC = 10
    proximity_range = 200
    min_npc_distance = 0
    map = "Town05"
    weather = carla.WeatherParameters(
        cloudiness=80.0,  # 云量
        precipitation_deposits =70.0,  # 降水量
        sun_altitude_angle=70.0  # 太阳的高度角
    )
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    tm = configure_traffic_manager(client)
    world = client.load_world(map)
    world.set_weather(weather)
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    # set all traffic lights green
    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Green)
        traffic_light.freeze(True)
    sensor_list = []
    sensor_queue = Queue()
    batch_sensor_data = []
    lidar_specs = {}
    blueprint_library = world.get_blueprint_library()

    all_vehicle_blueprints = list(blueprint_library.filter('vehicle.*'))
    random.shuffle(all_vehicle_blueprints)
    car_bp = blueprint_library.find('vehicle.tesla.model3')

    all_vehicles = []
    if not car_bp:
        print("Car blueprint 'vehicle.tesla.model3' not found.")
        pygame.quit()
        return

    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = try_spawn_vehicle(world, car_bp, spawn_point)
    if not vehicle:
        print("Failed to spawn main vehicle.")
        pygame.quit()
        return
    all_vehicles.append(vehicle)
    print('Created %s' % vehicle.type_id)
    ################# setup sensors ############
    sensor_data_frame = {}
    with open('sensor_setup.yaml', 'r') as file:
        data = yaml.safe_load(file)

        cam_spec = data['cam_specs']
        lidar_specs = data['lidar_specs']

        for key, value in lidar_specs.items():
            spawn_semantic_lidar(world,vehicle,key,value,sensor_list,sensor_queue)

    #################  spawn npc ########
    npc_positions = []
    sum = 0
    while sum < num_NPC:
        nearby_spawn_points = [
            sp for sp in world.get_map().get_spawn_points()
            if sp.location.distance(vehicle.get_location()) <= proximity_range and sp.location.distance(vehicle.get_location()) > 10
        ]

        valid_spawn_points = [
            sp for sp in nearby_spawn_points
            if all(sp.location.distance(npc) > min_npc_distance for npc in npc_positions)
        ]

        if valid_spawn_points:
            npc_spawn_point = random.choice(valid_spawn_points)
            npc_bp = random.choice(all_vehicle_blueprints)
            npc_vehicle = try_spawn_vehicle(world, npc_bp, npc_spawn_point)
            if npc_vehicle:
                all_vehicles.append(npc_vehicle)
                npc_positions.append(npc_vehicle.get_location())
                sum += 1
                print('Spawned NPC %s at %s' % (npc_vehicle.type_id, npc_spawn_point.location))
            else:
                print('Failed to spawn NPC vehicle at %s' % npc_spawn_point.location)
        else:
            print('No suitable spawn points available for NPCs')

    tracking_enabled = True
    for v in all_vehicles:
        v.set_autopilot(True, tm.get_port())

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    ticks = 0
    try:
        print("Start driving！！！")
        now = datetime.now()
        formatted_time = now.strftime('%Y_%m_%d_%H_%M_%S')
        while ticks<200:

            world.tick()
            for i in range(0, len(sensor_list)):
                s_data = sensor_queue.get(block=True, timeout=1.0)
                sensor_data_frame[s_data[1]] = s_data[0]
            ticks += 1
            print("ticked once!")
            if check_for_h_key():
                tracking_enabled = not tracking_enabled
                print('Tracking toggled:', 'On' if tracking_enabled else 'Off')


            if tracking_enabled:
                update_spectator_to_vehicle(world, vehicle)
            if ticks%5 == 0: # 2hz as NuScenes setup
                print("should save data now!!!")
                save_unit_data(sensor_data_frame,"./output"+ "/" + formatted_time + "/task0",ticks,lidar_specs)
                # for sensor in sensor_list:

    except KeyboardInterrupt:
        print('\nSimulation stopped by user.')
    finally:
        print('Destroying actors')
        for actor in world.get_actors():
            if actor.type_id.startswith('vehicle.'):
                actor.destroy()
        for sensor in sensor_list:
            sensor.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
