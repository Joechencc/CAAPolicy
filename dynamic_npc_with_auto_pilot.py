import carla
import random
import pygame
import time

def init_pygame():
    pygame.init()
    size = (200, 200)
    pygame.display.set_mode(size)


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
def update_spectator_to_vehicle(world, vehicle, offset=carla.Location(x=-8, z=5)):
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

    num_NPC = 30
    proximity_range = 200
    min_npc_distance = 10
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    tm = configure_traffic_manager(client)
    world = client.get_world()
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

    tracking_enabled = False
    for v in all_vehicles:
        v.set_autopilot(True, tm.get_port())
    print("Start driving！！！")
    try:
        while True:
            if check_for_h_key():
                tracking_enabled = not tracking_enabled
                print('Tracking toggled:', 'On' if tracking_enabled else 'Off')

            if tracking_enabled:
                update_spectator_to_vehicle(world, vehicle)

            time.sleep(0.05)
    except KeyboardInterrupt:
        print('\nSimulation stopped by user.')
    finally:
        print('Destroying actors')
        for actor in world.get_actors():
            if actor.type_id.startswith('vehicle.'):
                actor.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
