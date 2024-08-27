import carla
import random
import pygame
import time


def init_pygame():
    pygame.init()
    size = (200, 200)
    pygame.display.set_mode(size)


def attach_sensors_to_vehicle(world, vehicle):
    blueprint_library = world.get_blueprint_library()

    # RGB Camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Semantic LiDAR
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('range', '50')
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    return camera, lidar


def setup_sensor_callbacks(camera, lidar):
    def save_camera_image(image):
        image.save_to_disk('output/rgb_%06d.png' % image.frame)

    def save_lidar_data(lidar_data):
        lidar_data.save_to_disk('output/lidar_%06d.ply' % lidar_data.frame)

    camera.listen(save_camera_image)
    lidar.listen(save_lidar_data)


def update_spectator_to_vehicle(world, vehicle, offset=carla.Location(x=-8, z=5)):
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator_transform = carla.Transform(transform.location + offset, transform.rotation)
    spectator.set_transform(spectator_transform)


def check_for_h_key():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
            return True
    return False


def main():
    init_pygame()
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    car_bp = blueprint_library.find('vehicle.audi.a2')

    if not car_bp:
        print("Car blueprint 'vehicle.audi.a2' not found.")
        pygame.quit()
        return

    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(car_bp, spawn_point)
    if vehicle is None:
        print("Vehicle could not be spawned.")
        pygame.quit()
        return
    vehicle.set_autopilot(True)
    print('Created %s' % vehicle.type_id)
    camera, lidar = attach_sensors_to_vehicle(world, vehicle)
    setup_sensor_callbacks(camera, lidar)

    try:
        while True:
            update_spectator_to_vehicle(world, vehicle)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('\nSimulation stopped by user.')
    finally:
        print('Destroying actors')
        camera.destroy()
        lidar.destroy()
        vehicle.destroy()
        pygame.quit()


if __name__ == '__main__':
    main()
