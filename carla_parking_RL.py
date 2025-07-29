from data_generation.network_evaluator import NetworkEvaluator, NetworkEvaluatorRL
from agent.parking_agent import ParkingAgent, ParkingAgentRL, show_control_info
from data_generation.keyboard_control import KeyboardControl
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from gym import Env, spaces
import argparse
import logging
import carla
import numpy as np
import torch
import pygame

from carla_parking_eva import str2bool
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CarlaParkingEnv(Env):
    def __init__(self, network_evaluator, parking_agent, controller, encoder, cfg):
        super().__init__()
        self.network_evaluator = network_evaluator
        self.agent = parking_agent
        self.controller = controller
        self.encoder = encoder
        self.cfg = cfg

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(cfg.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(cfg.num_controls)  # or continuous if modified

    def reset(self):
        self.network_evaluator.reset_task()
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        data = self.network_evaluator.get_input_data()
        with torch.no_grad():
            fuse_feature, *_ = self.encoder(data)
            fuse_feature = torch.mean(fuse_feature, dim=2).squeeze(0)  # shape [C]
        return fuse_feature.cpu().numpy()

    def step(self, action_idx):
        network_evaluator.world_tick()
        if controller.parse_events(client, network_evaluator.world, clock):
            return
        parking_agent.tick()
        network_evaluator.tick(clock)

        self.agent.apply_action(action_idx)
        reward = self.agent.compute_reward()
        done = self.agent.check_done()
        obs = self._get_obs()
        return obs, reward, done, {}

def compute_discounted_rewards(rewards, gamma):
    """
    Compute reward-to-go for a trajectory.
    
    Args:
        rewards (List[float]): rewards at each timestep [r0, r1, ..., rT]
        gamma (float): discount factor (0 < gamma ≤ 1)
    
    Returns:
        List[float]: discounted returns G_t for each timestep t
    """
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards

class ValueNet(nn.Module):
    def __init__(self, in_channels=264, hidden_dim=512):
        super(ValueNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),  # (B, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),           # (B, 64, 16, 16)
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output a scalar V(s)
        )

    def forward(self, fuse_feature):  # input shape: (B, 256, 264)
        B, T, C = fuse_feature.shape
        assert T == 256, "Expected 256 spatial tokens (16x16)"
        assert C == 264, f"Expected 264 channels, got {C}"

        # Reshape to spatial map: (B, 264, 16, 16)
        x = fuse_feature.transpose(1, 2).reshape(B, C, 16, 16)  # (B, 264, 16, 16)
        x = self.conv(x)                                       # (B, 64, 16, 16)
        x = F.adaptive_avg_pool2d(x, 1).view(B, -1)            # (B, 64)
        value = self.mlp(x)                                    # (B, 1)
        return value

def ppo_update(parking_model, trajectories, value_net, optimizer, clip_eps=0.2, gamma=0.99):
    all_features, all_actions, all_logits, accum_discount_rews, all_rewards = [], [], [], [], []
    for traj in trajectories:
        all_features += traj[0]
        all_actions += traj[1]
        all_logits += traj[2]
        accum_discount_rews += traj[3]
        all_rewards += traj[4]

    # discounted_returns = compute_discounted_rewards(all_rewards, gamma)
    states = torch.cat(all_features, dim=0).detach()    # [T, 256, 264]
    actions = torch.cat(all_actions, dim=0).long().detach()    # [T, 4]
    logits_old = torch.stack(all_logits, dim=0).detach()    # [T, 3, 204]
    accum_discount_rews = torch.tensor(accum_discount_rews, dtype=torch.float32).cuda().detach()  

    # values = value_net(states).squeeze(-1)


    # advantages = accum_discount_rews - values.detach()

    advantages = accum_discount_rews

    # Compute PPO loss for each autoregressive step
    policy_loss, entropy_loss = 0, 0
    i = np.random.randint(3)
    # for i in range(3):

    logits_i_old = logits_old[:, i, :]  # [T, 204]
    dist_old = torch.distributions.Categorical(logits=logits_i_old)
    log_probs_old = dist_old.log_prob(actions[:, i])

    # Get new logits from model (recomputing)
    pred_multi_controls = actions[:, 0:i+1]
    logits_i_new = parking_model.fuse_feature_to_logits(states, pred_multi_controls)

    # logits_i_new = new_logits[i]
    dist_new = torch.distributions.Categorical(logits=logits_i_new)
    log_probs_new = dist_new.log_prob(actions[:, i])
    entropy_loss += dist_new.entropy().mean()

    # PPO loss
    ratio = torch.exp(log_probs_new - log_probs_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss += -torch.min(surr1, surr2).mean()

    # value_loss = F.mse_loss(values, accum_discount_rews)

    # loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

    loss = policy_loss - 0.01 * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), policy_loss.item(), entropy_loss.item()


if __name__ == "__main__":

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
        choices=['Town03', 'Town04_Opt', 'Town05_Opt'])
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
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # INFO: Start carla

    pygame.init()
    pygame.font.init()
    network_evaluator = None
    next_flag = True

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        logging.info('Load Map %s', args.map)
        carla_world = client.load_world(args.map)
        carla_world.unload_map_layer(carla.MapLayer.ParkedVehicles)


        network_evaluator = NetworkEvaluatorRL(carla_world, args)
        parking_agent = ParkingAgentRL(network_evaluator, args)
        controller = KeyboardControl(network_evaluator.world)

        # Apply weather AFTER everything is initialized

        import time
        time.sleep(0.2)
        weather_0911_default = carla.WeatherParameters(
            cloudiness=35.0,
            precipitation=30.0,
            precipitation_deposits=20.0,
            wind_intensity=0.35,
            sun_azimuth_angle=20.0,
            sun_altitude_angle=25.0,
            fog_density=2.0,
            fog_distance=0.0,
            fog_falloff=0.0,
            wetness=20.0,
        )

        carla_world.set_weather(weather_0911_default)
        time.sleep(0.1)
        print("Final weather:", carla_world.get_weather())

        display = pygame.display.set_mode((args.width, args.height),
                                          pygame.HWSURFACE | pygame.DOUBLEBUF)

        settings = carla_world.get_settings()
        settings.no_rendering_mode = True  # ✅ disables rendering
        carla_world.apply_settings(settings)

        steer_wheel_img = pygame.image.load("./resource/steer_wheel.png")
        steer_wheel_img = pygame.transform.scale(steer_wheel_img, (100, 100))
        font = pygame.font.Font(None, 25)

        clock = pygame.time.Clock()

        fuse_features, actions, all_logits, rewards, trajectories = [], [], [], [], []
        episode_num = 0

        # Freeze all modules except control_predict
        parking_model = parking_agent.model
        for name, param in parking_model.named_parameters():
            if 'control_predict' not in name and 'grad_approx' not in name:
                param.requires_grad = False
        parking_model.eval()  # Freeze mode globally
        parking_model.control_predict.train()  # Make sure this is still trainable
        value_net = ValueNet(in_channels=264, hidden_dim=512).to(DEVICE)
        optimizer = torch.optim.Adam(
            list(parking_model.control_predict.parameters())+list(parking_model.grad_approx.parameters()),
            lr=1e-6,  # adjust as needed
            weight_decay=1e-5  # optional
        )
        # list(value_net.parameters())
        writer = SummaryWriter(log_dir=network_evaluator._eva_result_path)
        global_step = 0

        while True:
            network_evaluator.world_tick()
            clock.tick_busy_loop(60)
            if controller.parse_events(client, network_evaluator.world, clock):
                break # return 
            obs_act = parking_agent.tick()
            new_episode_flag, reward = network_evaluator.tick(clock)
            
            # print("New Episode = ", new_episode_flag)
            # print("Reward = ", reward)
            if obs_act:
                fuse_feature, pred_controls, pred_tgt_logits = obs_act
                fuse_features.append(fuse_feature)
                actions.append(pred_controls)
                all_logits.append(pred_tgt_logits)
                rewards.append(reward)
            if new_episode_flag:
                global_step = global_step + 1
                gamma = 0.97
                accum_discount_rews = compute_discounted_rewards(rewards, gamma)
                # print("accum_rew = ", accum_rew)
                trajectories.append((fuse_features, actions, all_logits, accum_discount_rews, rewards))
            
                # INFO: Update PPO
                if len(trajectories) == 3:
                    for _ in range(3):
                        loss, policy_loss, entropy_loss = ppo_update(parking_model, trajectories, value_net, optimizer, clip_eps=0.2, gamma=gamma)

                    writer.add_scalar("Loss/Total", loss, global_step)
                    writer.add_scalar("Loss/Policy", policy_loss, global_step)
                    # writer.add_scalar("Loss/Value", value_loss, global_step)
                    writer.add_scalar("Loss/Entropy", entropy_loss, global_step)
                    writer.add_scalar("Reward", np.mean(rewards), global_step)
                    success_rate = (network_evaluator._target_success_nums / float(global_step))
                    writer.add_scalar("Success Rate", success_rate, global_step)

                if len(trajectories) > 2:
                    trajectories.pop(0)

                fuse_features, actions, all_logits, rewards = [], [], [], []

                
                
            # network_evaluator.render(display)
            # show_control_info(display, parking_agent.get_eva_control(), steer_wheel_img,
            #                   args.width, args.height, font)
            pygame.display.flip()

            # if dd:
            #     trajectories.append((fuse_feature, actions, all_logits, q_value, reward))

    finally:
        if network_evaluator:
            client.stop_recorder()

        if network_evaluator is not None:
            network_evaluator.destroy()

        pygame.quit()

    # Wrap in gym env
    # rl_env = CarlaParkingEnv(network_evaluator, parking_agent, controller, encoder, args)

    # check_env(rl_env)
    # model = SAC("MlpPolicy", rl_env, verbose=1)
    # model.learn(total_timesteps=200000)

    # torch_red_observation = [Variable(torch.Tensor(red_observation[i]), requires_grad=False).to(device) for i in range(agent_num)]
    # # to_waypt_vec_normalized = torch_red_observation[0][-3:-1] / (torch.linalg.norm(torch_red_observation[0][-3:-1]) + 1e-3)
    # # torch_agent_actions = [to_waypt_vec_normalized]
    # torch_agent_actions = sac.select_action(torch_red_observation)
    # agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions] # agent actions for all robots, each element is an array with dimension 5
    # next_red_observation, rewards, done, i, _, red_detected_flag = env.step(split_red_directions_to_direction_speed((np.concatenate(agent_actions))))
    # replay_buffer.push(red_observation, agent_actions, rewards, next_red_observation, done) 