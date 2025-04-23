import argparse
import logging
import carla
import pygame

from data_generation.data_generator import DataGenerator
from data_generation.multi_control import MultiControl


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

        data_generator = DataGenerator(carla_world, args)
        controller = MultiControl(data_generator.world)

        # weather = carla_world.get_weather()
        # # print("sun_azimuth_angle", weather.sun_azimuth_angle)  # output: 0. 0-360°，0° is north，90° is east，180° is south
        # # print("sun_altitude_angle", weather.sun_altitude_angle) # output: 75. -90°-90°，0° is horizontal，90° is overhead, -90° is directly below the ground (not visible)
        # # Modify the position of the sun. (sun_azimuth_angle, sun_altitude_angle): (301, 66) or (65, 66)
        # # set (301, 66) to project the shadow on slot 3-8; set (65, 66) to project the shadow on slot 3-10
        # weather.sun_azimuth_angle = 301
        # weather.sun_altitude_angle = 66
        # carla_world.set_weather(weather)

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
        type=int,
        default=0,
        help='random seed to initialize env; if sets to 0, use current timestamp as seed (default: 0)')
    argparser.add_argument(
        '--bev_render_device',
        default='cuda',
        help='device used for BEV Rendering (default: cuda)',
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
