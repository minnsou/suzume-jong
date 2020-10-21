import sys
import os
import argparse
import re
import pickle

from baselines.ppo2 import ppo2
from baselines.common.cmd_util import parse_unknown_args
from baselines.run import parse_cmdline_kwargs
from baselines import run
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines import logger
import gym
import numpy as np

import suzume_env
import mahjong_utils
import mahjong_networks
import baseline_run
from mahjong_utils import hand2show_str, NUM_SAME_TILE, KIND_TILE, KIND_TILE_WITH_RED

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Suzume2-v0')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('-t', '--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=5, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default='save_models/default.pkl', type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    # for --play
    parser.add_argument('-pn', '--pickle_name', help='name using pickle file(usually opponent gamma)', default='none', type=str)
    parser.add_argument('-dc', "--do_complete_game", help="whether play complete game or not (default False)", default=False, action='store_true')
    parser.add_argument("-ns", "--numpy_seed", help="numpy random seed (default 0)", type=int, default=0)
    return parser

def path2vars(path):
    env_vars = re.findall(r'_(\d+_\d+_\d+_.*).pkl', path)
    if env_vars == []:
        return '0', '0', '0'
    env_vars = env_vars[0].split('_')
    num_players = int(env_vars[0])
    obs_mode = int(env_vars[2])
    name = env_vars[3]
    print('num_players {} obs_mode {} name {}'.format(num_players, obs_mode, name))
    return num_players, obs_mode, name

def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)

def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    print(args, extra_args)
    configure_logger(args.log_path, format_strs=[])

    if args.play:
        args.num_timesteps = 0
        args.num_env = 1
        print("set num_timesteps 0 and num_env 1")
        num_players, obs_mode, name = path2vars(extra_args['load_path'])
    
    model, env = run.train(args, extra_args)
    if not args.play:
        save_path = os.path.expanduser(args.save_path)
        model.save(save_path)
        num_players, obs_mode, name = path2vars(save_path)
    else:
        logger.log("Running trained model")
        obs = env.reset()
        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))
        episode_rew = 0
        for i in range(30):
            print('\033[32m'+'show obs'+'\033[0m')
            if obs_mode == 3:
                hand = mahjong_utils.plane2hand(obs[0], 1) 
                print('hand', hand, '=>', hand2show_str(hand))
            elif obs_mode == 1 or obs_mode == 2:
                offset = 0
                for j in range(num_players):
                    discard_hist = list(map(int, obs[0, offset:offset+20].tolist()))
                    discard_tile = mahjong_utils.hist2hand(discard_hist)
                    discard_tile = hand2show_str(discard_tile)
                    offset += 20
                    print(f'discards{i} "{discard_tile}"')
                dora = np.argmax(obs[0, offset:offset+KIND_TILE_WITH_RED])
                print('dora {} => {}'.format(dora, hand2show_str([dora])))
                if obs_mode == 1:
                    plane = obs[0, -KIND_TILE_WITH_RED*NUM_SAME_TILE:].reshape(4, 20)
                    hand = mahjong_utils.plane2hand(plane, 1)
                    print(f'player0 {hand} => {hand2show_str(hand)}')
                else:
                    plane = obs[0, -KIND_TILE*NUM_SAME_TILE:].reshape(4, 11)
                    hand = mahjong_utils.plane2hand(plane, 2)
                    print(f'player0 {hand} => {hand2show_str(hand)}')
            elif obs_mode == 4:
                obs = obs[0]
                if np.sum(obs[0]) == 0:
                    print('no dealer')
                else:
                    dealer_num = np.argmax(obs[0, 0, :])
                    print('dealer is {}'.format(dealer_num))
                dora = mahjong_utils.plane2hand(obs[1], 2)
                print('dora {} => {}'.format(dora[0], hand2show_str(dora)))
                hand = mahjong_utils.plane2hand(obs[2], 2)
                print('hand', hand, '=>', hand2show_str(hand))
                for j in range(num_players):
                    discard_tile = mahjong_utils.plane2hand(obs[j+3], 2)
                    print(f'player{j} {discard_tile} => {hand2show_str(discard_tile)}')
            elif obs_mode == 5:
                # TODO:
                pass
                    
            print('\033[32m'+'show other info'+'\033[0m')
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs)
            
            obs, rew, done, info = env.step(actions)
            print('act', actions[0], '=>', hand2show_str(actions))
            print('rew', rew[0])
            print('done', done[0])
            # TODO
            print('info')
            if args.do_complete_game:
                for i in range(len(info[0])):
                    print(info[0][i])
            else:
                print(info)
            # if info[0]['is_win']:
            #     print('info ', end='')
            #     print(info[0]['info']['type'])
            #     for i in range(num_players):
            #         hand = info[0]['info'][f'player{i}']
            #         print(f'player{i} hand {hand} => {hand2show_str(hand)}')
            #     win_tile = info[0]['info']['tile']
            #     print(f'win tile {win_tile} => {hand2show_str([win_tile])}')
            #     print(f'episode {info[0]["episode"]}')
            # elif info[0]['info'] == 'no tile':
            #     print('env reset due to no tile')
            #     episode_rew += rew
            #     print('episode_rew={}\n'.format(episode_rew))
            #     episode_rew = 0
            #     obs = env.reset()
            #     continue
            # else:
            #     print(f'info {info[0]}\n')
            episode_rew += rew
            if done:
                print('episode_rew={}\n'.format(episode_rew))
                episode_rew = 0
                
        player_gamma = 'ppo' + name
        total_timesteps = 3000000
        if obs_mode == 5 or obs_mode == 6:
            args.do_complete_game = True
            args.numpy_seed = 0
        count_dict = baseline_run.make_count_dict(
            env=env,
            player_gamma=player_gamma,
            opponent_gamma=args.pickle_name,
            total_timesteps=total_timesteps,
            obs_mode=2, # anything is OK
            num_players=num_players,
            do_complete_game=args.do_complete_game,
            numpy_seed=args.numpy_seed,
            model=model,
            debug=False,
        )
        baseline_run.show_result(count_dict)
        with open('./baseline_results/{}_{}_{}_{}_{}.pickle'.format(
                num_players,
                player_gamma,
                args.pickle_name,
                total_timesteps,
                args.do_complete_game,
        ),
            'wb') as f:
            pickle.dump(count_dict, f)
    env.close()
    
if __name__ == '__main__':
    main(sys.argv)
