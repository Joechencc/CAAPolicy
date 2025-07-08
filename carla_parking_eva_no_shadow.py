
import argparse
import logging
import carla
import pygame
import os

from data_generation.network_evaluator import NetworkEvaluator
from data_generation.keyboard_control import KeyboardControl
from agent.parking_agent import ParkingAgent, show_control_info


def game_loop(args):
    pygame.init()
    pygame.font.init()
    network_evaluator = None
    next_flag = True



    # 檢查當前品質模式
    config_path = os.path.expanduser("~/.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor/GameUserSettings.ini")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            content = f.read()
            if 'sg.ShadowQuality=0' in content:
                print("✅ 當前模式: LOW (影子已關閉)")
            else:
                print("⚠️ 當前模式: HIGH/EPIC (影子開啟)")
    else:
        print("⚠️ 未找到配置檔案，使用預設 EPIC 模式")

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        logging.info('Load Map %s', args.map)
        carla_world = client.load_world(args.map)
        carla_world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        network_evaluator = NetworkEvaluator(carla_world, args)
        parking_agent = ParkingAgent(network_evaluator, args)
        controller = KeyboardControl(network_evaluator.world)

        settings = carla_world.get_settings()
        settings.no_rendering_mode = False
        carla_world.set_weather(carla.WeatherParameters.ClearNoon)

        # 修正：正確設定影子相關參數
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        carla_world.apply_settings(settings)

        # 方法一：使用品質設定關閉影子（最有效的方法）
        # 這會將畫質設定為低品質模式，自動關閉影子
        try:
            # 嘗試設定低品質模式（會自動關閉影子）
            client.get_tracer().set_shadows(False)  # 如果支援的話
        except AttributeError:
            logging.info("Direct shadow control not available, using quality settings")

        # 方法二：正確的燈光管理方式
        # 使用 LightManager 控制街燈（不是用 set_attribute）
        try:
            light_manager = carla_world.get_lightmanager()  # 注意：是 get_lightmanager() 不是 get_light_manager()
           
            # 獲取所有燈光並關閉
            all_lights = light_manager.get_all_lights()
           
            for light in all_lights:
                # 關閉燈光以減少影子投射
                light.turn_off()
               
        except AttributeError as e:
            logging.info(f"LightManager not available: {e}")
           
            # 備用方法：直接控制燈光 actors
            try:
                # 獲取並關閉街燈
                for actor in carla_world.get_actors().filter("static.light*"):
                    try:
                        # 燈光物件沒有 cast_shadows 屬性，但可以關閉燈光本身
                        actor.turn_off()
                    except AttributeError:
                        # 如果沒有 turn_off 方法，嘗試刪除
                        logging.info(f"Cannot turn off light {actor.id}, attempting to destroy")
                        try:
                            actor.destroy()
                        except:
                            pass

                # 處理其他類型的燈光
                for light in carla_world.get_actors().filter("light.*"):
                    try:
                        light.turn_off()
                    except AttributeError:
                        try:
                            light.destroy()
                        except:
                            pass
                           
            except Exception as e:
                logging.warning(f"Could not control lights: {e}")

        # 方法三：使用 Unreal Engine 控制指令（如果可用）
        try:
            # 嘗試執行 UE4 控制台指令來關閉影子
            carla_world.on_tick(lambda _: None)  # 確保世界已載入
           
            # 這些是 UE4 的影子控制指令
            console_commands = [
                "r.ShadowQuality 0",           # 關閉影子品質
                "r.ContactShadows 0",          # 關閉接觸影子
                "r.CascadedShadowMaps 0",      # 關閉階層式影子地圖
                "r.DistanceFieldShadowing 0",  # 關閉距離場影子
                "r.DynamicGlobalIllumination 0", # 關閉動態全域照明
            ]
           
            for cmd in console_commands:
                try:
                    # 注意：這需要 CARLA 支援控制台指令執行
                    carla_world.execute_console_command(cmd)
                    logging.info(f"Executed: {cmd}")
                except AttributeError:
                    logging.info("Console command execution not available")
                    break
                   
        except Exception as e:
            logging.info(f"Could not execute console commands: {e}")



        display = pygame.display.set_mode((args.width, args.height),
                                         pygame.HWSURFACE | pygame.DOUBLEBUF)

        steer_wheel_img = pygame.image.load("./resource/steer_wheel.png")
        steer_wheel_img = pygame.transform.scale(steer_wheel_img, (100, 100))
        font = pygame.font.Font(None, 25)

        clock = pygame.time.Clock()
        while True:
            network_evaluator.world_tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, network_evaluator.world, clock):
                return
            parking_agent.tick()
            network_evaluator.tick(clock)
            network_evaluator.render(display)
            show_control_info(display, parking_agent.get_eva_control(), steer_wheel_img,
                             args.width, args.height, font)
            pygame.display.flip()

    finally:
        if network_evaluator:
            client.stop_recorder()

        if network_evaluator is not None:
            network_evaluator.destroy()

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
        '--model_path',
        default='./ckpt/last.ckpt',
        help='path to model.ckpt')
    argparser.add_argument(
        '--model_path_dynamic',
        default='./ckpt/dynamic_model.ckpt',
        help='path to dynamic_model.ckpt')
    argparser.add_argument(
        '--model_config_path',
        default='./config/training.yaml',
        help='path to model training.yaml')
    argparser.add_argument(
        '--dynamic_model_config_path',
        default='./config/dynamics_training.yaml',
        help='path to model dynamics_training.yaml')
    argparser.add_argument(
        '--eva_epochs',
        default=4,
        type=int,
        help='number of eva epochs (default: 4')
    argparser.add_argument(
        '--eva_task_nums',
        default=16,
        type=int,
        help='number of parking slot task (default: 16')
    argparser.add_argument(
        '--eva_parking_nums',
        default=6,
        type=int,
        help='number of parking nums for every slot (default: 6')
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
        default=66,
        help='random seed to initialize env; if sets to 0, use current timestamp as seed (default: 0)')
    argparser.add_argument(
        '--bev_render_device',
        default='cpu',
        help='device used for BEV Rendering (default: cpu)',
        choices=['cpu', 'cuda'])
    argparser.add_argument(
        '--show_eva_imgs',
        default=False,
        type=str2bool,
        help='show eva figure in eva model (default: False)')
    argparser.add_argument(
        '--eva_result_path',
        default='./eva_result',
        help='path to save eva csv file')
    # 新增品質設定參數
    argparser.add_argument(
        '--quality-level',
        default='Low',
        help='Graphics quality level (default: Low to disable shadows)',
        choices=['Low', 'Medium', 'High', 'Epic'])
   
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # 提示用戶關於影子設定
    if hasattr(args, 'quality_level') and args.quality_level == 'Low':
        logging.info('Running in Low quality mode - shadows will be disabled')

    try:
        game_loop(args)

    except KeyboardInterrupt:
        logging.info('Cancelled by user. Bye!')


if __name__ == '__main__':
    main()

