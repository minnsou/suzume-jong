import pickle
import random
import argparse
from copy import deepcopy
import json
import pprint
import datetime
import os

import numpy as np

from suzume_env.main import SuzumeEnv
import mahjong_networks
import mahjong_utils

def show_result(count_dict):
    #total_kyoku = 0
    for key in ['win_tumo', 'win_ron', 'lose_tumo', 'lose_ron']:
        re_list = np.array(count_dict[key])
        #total_kyoku += len(count_dict[key])
        if len(re_list) == 0:
            print('{} \tnum 0'.format(key))
        else:
            print('{} \tnum {}\tsum {}\tmax {}\tmin {}\tmean {:.4}\tvar {:.4}'
                  .format(
                      key,
                      len(re_list),
                      int(np.sum(re_list)),
                      int(np.max(re_list)),
                      int(np.min(re_list)),
                      round(np.mean(re_list), 4),
                      round(np.var(re_list), 4)
                  ))
    if len(count_dict['win_turn']) == 0:
        print('win turn ave  None')
    else:
        print('win turn ave {:.3}\twin turn var {:.3f}'.format(
              round(np.mean(count_dict['win_turn']), 3),
              round(np.var(count_dict['win_turn']), 3),
            )
        )
    for key in ['others_ron', 'no wall']:
        print('{} num {}'.format(key, count_dict[key]))
        #total_kyoku += count_dict[key]
    if 'rank' in count_dict:
        print('rank ave {:.4}\t rank var {:.4f}'.format(
              round(np.mean(count_dict['rank']), 4),
              round(np.var(count_dict['rank']), 4),
            )
        )
        print('total game', count_dict['num_game'])
    print('total kyoku', count_dict['num_kyoku'])
    #print('total_kyoku', total_kyoku)

def info_count(count_dict, info, do_complete_game, num_players):
    round_end = False
    for i in range(len(info)):
        if 'is_win' in info[i] and info[i]['is_win']:
            round_end = True
            if info[i]['info']['win_player'] == 'player0':
                count_dict['win_turn'].append(info[i]['info']['turn'])
                key_str = 'win_' + info[i]['info']['type']
                count_dict[key_str].append(info[i]['info']['point'])
            elif info[i]['info']['type'] == 'tumo':
                #print(num_players)
                point = - info[i]['info']['point'] / (num_players - 1)
                count_dict['lose_tumo'].append(point)
            elif info[i]['info']['lose_player'] == 'player0':
                count_dict['lose_ron'].append(-info[i]['info']['point'])
            else:
                count_dict['others_ron'] += 1
        elif 'info' in info[i] and info[i]['info'] == 'no wall':
            round_end = True
            count_dict['no wall'] += 1
        if do_complete_game and 'final_point' in info[i]:
            rank = mahjong_utils.get_rank(
                info[i]['final_point'],
                info[i]['first_dealer_num'],
                'player0',
            )
            count_dict['rank'].append(rank)
            count_dict['num_game'] += 1
            count_dict['num_kyoku'] += 1
    if do_complete_game:
        num_kyoku = info[len(info)-1]['round_num'] - info[0]['round_num']
        count_dict['num_kyoku'] += num_kyoku
    else:
        if round_end:
            count_dict['num_kyoku'] += 1

def make_discards_log(discards, wall, dealer, finish):
    discards_log = {}
    num_players = len(discards)
    num_discards = 0
    for i in range(num_players):
        num_discards += len(discards[f'player{i}'])
    dealer_num =int(dealer[-1:])
    turn_player_num = dealer_num
    dis_idx = 0
    num = 0
    for i in range(num_discards + 1):
        tumo_tile = wall.pop()
        discards_log[f'{num} {turn_player_num}t'] = tumo_tile
        num += 1
        if finish == 'tumo' and i == num_discards:
            break
        discarded_tile = discards[f'player{turn_player_num}'][dis_idx]
        discards_log[f'{num} {turn_player_num}d'] = discarded_tile
        num += 1
        if (finish == 'ron' and i == num_discards - 1) or \
           len(wall) == 0 and finish == 'no wall':
            break
        turn_player_num = (turn_player_num + 1) % num_players
        if turn_player_num == dealer_num:
            dis_idx += 1
    return discards_log

def make_count_dict(
        env,
        player_gamma,
        opponent_gamma,
        total_timesteps,
        num_players,
        obs_mode,
        do_complete_game,
        numpy_seed=0,
        model=None,
        debug=False):
    
    # if do_complete_game:
    #     assert obs_mode == 5, 'please set obs_mode 5'
    # else:
    #     assert obs_mode == 2, 'please set obs_mode 2'
    count_dict = {
        'win_tumo': [],
        'win_ron': [],
        'lose_tumo': [],
        'lose_ron': [],
        'win_turn': [],
        'others_ron': 0,
        'no wall': 0,
        'num_kyoku': 0,
    }
    if do_complete_game:
        count_dict['rank'] = []
        count_dict['num_game'] = 0
    
    done = True
    #rewards = [] # if want to show reward ave, remove #
    for t in range(total_timesteps):
        if done:
            obs = env.reset()
            now = datetime.datetime.today()
            log = {
                "game_info": {
                    "num_players": num_players,
                    "player0": player_gamma,
                    "others": opponent_gamma,
                    "date": str(now.date()),
                    "time": str(now.time()),
                    "numpy_seed": numpy_seed,
                    "timesteps": total_timesteps,
                },
            }
        if player_gamma[:3] == 'ppo':
            actions, _, _, _ = model.step(obs)
            discard_tile = actions[0]
        else:
            dora = np.argmax(obs['dora'])
            hand = mahjong_utils.plane2hand(obs['hand'], obs_mode)
            discard_tile = mahjong_utils.get_discard_tile(hand, dora, player_gamma)
        obs, reward, done, info = env.step(discard_tile)
        if player_gamma[:3] == 'ppo':
            obs, reward, done, info = obs[0], reward[0], done[0], info[0]
        #rewards.append(reward)
        if 'episode' in info:
            del info['episode']
        info_count(count_dict, info, do_complete_game, num_players)
        for i in range(len(info)):
            info_dict = info[i]
            if not 'is_win' in info_dict:
                # this is reset_info
                wall = deepcopy(info_dict['wall'])
                if do_complete_game:
                    round_num = info_dict['round_num']
                    log[f'{round_num} reset'] = deepcopy(info_dict)
                agari_num = 0
                kyoku_dict = {}
            else:
                if info_dict['is_win'] or info_dict['info'] == 'no wall':
                    if 'type' in info_dict['info'] and \
                       info_dict['info']['type'] == 'tumo':
                        finish = 'tumo'
                    elif 'type' in info_dict['info'] and \
                       info_dict['info']['type'] == 'ron':
                        finish = 'ron'
                    else:
                        finish = 'no wall'
                    discards = deepcopy(info_dict['discards'])
                    dealer = info_dict['dealer']                    
                    if kyoku_dict == {}:
                        kyoku_dict = make_discards_log(
                            discards, wall, dealer, finish)
                    if info_dict['is_win']:
                        kyoku_dict[f'agari{agari_num}'] = \
                            deepcopy(info_dict['info'])
                        if 'final_point' in info_dict:
                            kyoku_dict[f'agari{agari_num}']['final_point'] = \
                                deepcopy(info_dict['final_point'])
                        agari_num += 1
                    else:
                        kyoku_dict['no wall'] = {}
                        kyoku_dict['no wall']['discards'] = discards
                        if 'final_point' in info_dict:
                            kyoku_dict['no wall']['final_point'] = \
                                deepcopy(info_dict['final_point'])
                    if do_complete_game:
                        log[f'{round_num} kyoku'] = deepcopy(kyoku_dict)
        if done:
            # save log
            #pprint.pprint(log, width=200)
            now = str(datetime.datetime.today())
            now = now.replace(' ', '_')
            dirname = f'./log_json/{num_players}_{player_gamma}_{opponent_gamma}'
            os.makedirs(dirname, exist_ok=True)
            name_str = '{}/{}_{}_{}_{}.json'.format(
                dirname,
                now,
                num_players,
                player_gamma,
                opponent_gamma
            )
            with open(name_str, 'w') as f:
                json.dump(log, f, indent=4, sort_keys=False)
        
        if debug:
            print('dora', mahjong_utils.hand2show_str([dora]))
            print('obs')
            for i in range(obs['discards'].shape[0]):
                hist = list(map(int, obs['discards'][i].tolist()))
                discarded_tiles = mahjong_utils.hist2hand(hist)
                discarded_tiles = mahjong_utils.hand2show_str(discarded_tiles)
                print(f'player{i} ', discarded_tiles)
            print('hand0', mahjong_utils.hand2show_str(hand))
            print('discard_tile', mahjong_utils.hand2show_str([discard_tile]))
            print('reward', reward, 'done', done)
            if info['is_win']:
                print('info')
                print('is_win', info['is_win'], ' type', info['info']['type'])
                for i in range(num_players):
                    print('hand{}  {}'.format(i, mahjong_utils.hand2show_str(info['info'][f'player{i}'])))
            else:
                print('info', info)
            print()

    # print('rewards \tnum {}\tsum {}\tmax {}\tmin {}\tmean {}  \tvar {}'.format(
    #         len(rewards),
    #         np.sum(rewards),
    #         np.max(rewards),
    #         np.min(rewards),
    #         round(np.mean(rewards), 3),
    #         round(np.var(rewards), 5),
    #     ))
    # print(sorted(list(set(rewards))))

    return count_dict
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-pg", "--player_gamma", help="player discount rate (default '09')", type=str, default='09')    
    parser.add_argument("-t", "--total_timesteps", help="number of total timesteps (defalut 3000000)", type=int, default=int(3 * 1e6))
    parser.add_argument("-l", "--load_path", help="load pickle file and show result", type=str, default=None)
    parser.add_argument('-d', '--debug', default=False, action='store_true')    

    # env 
    parser.add_argument("-p", "--num_players", help="number of players (default 2)", type=int, default=2)
    parser.add_argument("-og", "--opponent_gamma", help="opponent player discount rate (default '09')", type=str, default='09')    
    parser.add_argument("-om", "--obs_mode", help="observation mode (default 2)", type=int, default=2)
    parser.add_argument("-hd", "--has_dealer", help="env has dealer or not", default=False, action='store_true')
    parser.add_argument("-pr", "--pseudo_reward", help="whether add psuedo reward or not (default False)", default=False, action='store_true')
    parser.add_argument("-rs", "--reward_scale", help="reward divide by reward_scale (default 20)", type=int, default=20)
    parser.add_argument('-dc', "--do_complete_game", help="whether play complete game or not (default False)", default=False, action='store_true')
    parser.add_argument("-ns", "--numpy_seed", help="numpy random seed (default 0)", type=int, default=0)

    args = parser.parse_args()
    if args.do_complete_game:
        args.has_dealer = True
    print('args', args)

    if args.load_path:
        file_name = args.load_path
        with open(file_name, mode='rb') as f:
            count_dict = pickle.load(f)
            show_result(count_dict)
    else:
        env = SuzumeEnv(
            process_idx=0,
            num_players=args.num_players,
            gamma_str=args.opponent_gamma,
            obs_mode=args.obs_mode,
            has_dealer=args.has_dealer,
            pseudo_reward=args.pseudo_reward,
            reward_scale=args.reward_scale,
            # reward_mode=args.reward_mode,
            do_complete_game=args.do_complete_game,
            # rank_reward_scale=args.rank_reward_scale,
            numpy_seed=args.numpy_seed,
            debug=False,
        )

        count_dict = make_count_dict(
            env=env,
            player_gamma=args.player_gamma,
            opponent_gamma=args.opponent_gamma,
            total_timesteps=args.total_timesteps,
            obs_mode=args.obs_mode,
            num_players=args.num_players,
            do_complete_game=args.do_complete_game,
            numpy_seed=args.numpy_seed,
            debug=args.debug,
        )
        #print(count_dict)

        print()
        show_result(count_dict)
        with open('./baseline_results/{}_{}_{}_{}_{}.pickle'.format(
                args.num_players,
                args.player_gamma,
                args.opponent_gamma,
                args.total_timesteps,
                args.do_complete_game), 'wb') as f:
            pickle.dump(count_dict, f)
    
if __name__ == '__main__':
    main()
