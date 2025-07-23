import argparse
import logging
import carla
import pygame

from data_generation.data_generator import DataGenerator, OODDataGenerator
from data_generation.keyboard_control import KeyboardControl


def game_loop(args):
    pygame.init()
    pygame.font.init()
    data_generator = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        logging.info('Load Map %s', args.map)
        carla_world = client.load_world(args.map)
        carla_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        data_generator = OODDataGenerator(carla_world, args)
        controller = KeyboardControl(data_generator.world)


        # Apply weather AFTER everything is initialized
        import time
        # time.sleep(0.2)
        # weather_0911_default = carla.WeatherParameters(
        #     cloudiness=15.0,
        #     precipitation=0.0,
        #     precipitation_deposits=0.0,
        #     wind_intensity=0.35,
        #     sun_azimuth_angle=0.0,
        #     sun_altitude_angle=75.0,
        #     fog_density=0.0,
        #     fog_distance=0.0,
        #     fog_falloff=0.0,
        #     wetness=0.0,
        #     scattering_intensity=1.0,
        #     mie_scattering_scale=0.03,
        #     rayleigh_scattering_scale=0.0331,
        #     dust_storm=0.0
        # )

        # carla_world.set_weather(weather_0911_default)
        # time.sleep(0.1)
        print("Final weather:", carla_world.get_weather())
        
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        clock = pygame.time.Clock()
        while True:
            data_generator.world_tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, data_generator.world, clock):
                return
            data_generator.tick(clock)
            data_generator.render(display)
            pygame.display.flip()

    finally:

        if data_generator:
            client.stop_recorder()

        if data_generator is not None:
            data_generator.destroy()

        pygame.quit()


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Data Generation')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='860x480',
        help='window resolution (default: 860x480)')
    argparser.add_argument(
        '--gamma',
        default=0.0,
        type=float,
        help='Gamma correction of the camera (default: 0.0)')
    argparser.add_argument(
        '--save_path',
        default='./e2e_parking/',
        help='path to save sensor data (default: ./e2e_parking/)')
    argparser.add_argument(
        '--task_num',
        default=16,
        type=int,
        help='number of parking task (default: 16')
    argparser.add_argument(
        '--map',
        default='Town04_Opt',
        help='map of carla (default: Town04_Opt)',
        choices=['Town04_Opt', 'Town05_Opt'])
    argparser.add_argument(
        '--shuffle_veh',
        default=True,
        type=str2bool,
        help='shuffle static vehicles between tasks (default: True)')
    argparser.add_argument(
        '--shuffle_weather',
        default=False,
        type=str2bool,
        help='shuffle weather between tasks (default: False)')
    argparser.add_argument(
        '--random_seed',
        default=0,
        help='random seed to initialize env; if sets to 0, use current timestamp as seed (default: 0)')
    argparser.add_argument(
        '--bev_render_device',
        default='cpu',
        help='device used for BEV Rendering (default: cpu)',
        choices=['cpu', 'cuda'])
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        logging.info('Cancelled by user. Bye!')


if __name__ == '__main__':
    main()
