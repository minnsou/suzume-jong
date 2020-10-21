import math
import sys
import pickle
import itertools
from copy import deepcopy

import numpy as np
import gym
from gym import spaces

import mahjong_utils
from mahjong_utils import KIND_TILE_WITH_RED, MAX_LEN_HAND, NUM_SAME_TILE
from mahjong_utils import hand2show_str, hand2plane, get_rank, KIND_TILE
import reward_predict
from keras.utils import np_utils

class SuzumeEnv(gym.Env):
    def __init__(self, num_players=2, gamma_str='09',
                 obs_mode=1, has_dealer=False, pseudo_reward=False,
                 reward_scale=1, reward_mode=1, do_complete_game=False,
                 rank_reward_scale=1, model_type='',
                 numpy_seed=0, eval_mode=False, debug=True):

        '''
        num_players: number of players
        gamma_str: opponent play style(06, 09 or random)
        obs_mode: observation mode (more details in 'make_obs')
        has_dealer: if False, select dealer randomly
        pseudo_reward: if True, get 0.1 when having 1 meld
        reward_scale: reward is 'point / reward_scale'
        reward_mode: reward mode (more details in 'make_reward')
        do_complete_game: if False, no 'round_num' variable
        rank_reward_scale: rank reward scale
        model_type: global reward prediction's model name(dense, cnn, or gru)
        numpy_seed: numpy random seeed
        eval_mode: if True, when no discard tile in hand, discard random tile
        debug: if True, show variables
        '''

        np.random.seed(numpy_seed)
        #self.process_idx = process_idx
        self.num_players = num_players
        self.action_space = gym.spaces.Discrete(KIND_TILE_WITH_RED)
        self.debug = debug
        self.hands = {}
        self.discarded_tiles = {}
        self.points = {}
        self.wall = []
        self.dora = -1
        self.gamma_str = gamma_str
        self.reward_mode = reward_mode
        self.do_complete_game = do_complete_game
        if self.do_complete_game:
            #assert reward_mode == 2 or reward_mode == 3, 'set reward_mode'
            self.has_dealer = True
            self.reward_scale = 22
            self.round_num = 0
            self.final_round_num = self.num_players * 4 - 1
            self.spec = gym.envs.registration.EnvSpec(f'Suzume{num_players}-v1')
            #self.make_log = make_log
        else:
            self.has_dealer = has_dealer
            self.reward_scale = reward_scale
            self.spec = gym.envs.registration.EnvSpec(f'Suzume{num_players}-v0')
            #self.make_log = False
        if self.has_dealer:
            self.first_dealer_num =  str(np.random.randint(self.num_players))
            self.dealer = 'player' + str(self.first_dealer_num)
            self.reward_range = (-22 / self.reward_scale, 22 / self.reward_scale)
        else:
            self.dealer = None
            self.reward_range = (-20 / self.reward_scale, 20 / self.reward_scale)
        self.pseudo_reward = pseudo_reward
        self.obs_mode = obs_mode
        if obs_mode == 1:
            self.obs_hand_shape = (
                NUM_SAME_TILE,
                KIND_TILE_WITH_RED
            )
        elif obs_mode == 2 or obs_mode == 7:
            self.obs_hand_shape = (NUM_SAME_TILE, KIND_TILE)
        elif obs_mode == 3:
            self.obs_hand_shape = (NUM_SAME_TILE, KIND_TILE_WITH_RED)
        elif obs_mode == 8:
            self.obs_hand_shape = (KIND_TILE, NUM_SAME_TILE)
        elif obs_mode == 9:
            self.obs_hand_shape = (NUM_SAME_TILE+1, KIND_TILE+2)
        elif obs_mode == 10:
            self.obs_hand_shape = (KIND_TILE+2, NUM_SAME_TILE+1)
        elif obs_mode == 11:
            self.obs_hand_shape = (KIND_TILE+2, 2*NUM_SAME_TILE+5)
        elif obs_mode == 12:
            self.obs_hand_shape = (KIND_TILE+2, 27*self.num_players+8)
        if obs_mode == 1 or obs_mode == 2:
            self.observation_space = spaces.Dict({
                'dora': spaces.Box(
                    low=0,
                    high=1,
                    shape=(KIND_TILE_WITH_RED,),
                    dtype=np.int32,
                ),
                'hand': spaces.Box(
                    low=0,
                    high=1,
                    shape=self.obs_hand_shape,
                    dtype=np.int32,
                ),
                'discards': spaces.Box(
                    low=0,
                    high=4,
                    shape=(self.num_players, KIND_TILE_WITH_RED),
                    dtype=np.int32,
                ),
            })
        elif obs_mode == 3 or (7 <= obs_mode <= 12):
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=self.obs_hand_shape,
                dtype=np.int32,
            )
        elif obs_mode == 4 or obs_mode == 5 or obs_mode == 6:
            if obs_mode == 4:
                # dealer, dora, hand => 3, discards => num_players
                num_tile = 3 + num_players
            elif obs_mode == 5:
                # round_info, dora, hand => 3, discards, points => 2 * num_players
                num_tile = 3 + 2 * num_players
            elif obs_mode == 6:
                # dealer, round_num, dora, hand => 5,
                # points, discards => 3 * num_players
                num_tile = 5 + 3 * num_players
            self.obs_shape=(num_tile, NUM_SAME_TILE, KIND_TILE)
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=self.obs_shape,
                dtype=np.int32,
            )
        self.last_move = -1
        self.kyoku_discards = []
        self.players = [f'player{i}' for i in range(self.num_players)]
        self.rank_reward_scale = rank_reward_scale
        self.eval_mode = eval_mode
        if reward_mode == 4 or reward_mode == 5:
            from keras.models import load_model
            model_name = f'./save_rew_pred/{model_type}_{num_players}_09_09_100.h5'
            self.model = load_model(model_name)
            self.model_type = model_type
            self.before_round_num = 0
            self.before_round_point = 0
            self.before_points = {}
            self.before_dealer = 'player0'
            self.after_round_num = 0
            self.after_round_point = 0
            self.after_points = {}
            self.after_dealer = 'player0'
        self.show_proparties()

    def show_proparties(self):
        print('\nshow env proparties')
        for key, item in self.__dict__.items():
            print(key, item)
        print()

    def show_player_hand(self):
        for player in self.players:
            hand = self.hands[player]
            print('{} {} => "{}"'.format(player, hand, hand2show_str(hand)))

    def round_info2plane(self):
        plane = np.zeros((NUM_SAME_TILE, KIND_TILE))
        dealer_num = int(self.dealer[-1:])
        bin_str = format(self.round_num, '06b')
        #print(dealer_num)
        plane[:, dealer_num] = 1
        for i, s in enumerate(bin_str):
            plane[:, i+5] = int(s)
        #print('plane', plane)
        return plane

    def get_remaining_tiles(self, player):
        count_tiles = []
        for tile in range(KIND_TILE_WITH_RED):
            if tile == 18 or tile == 19:
                max_num = 4
            elif tile % 2 == 0:
                max_num = 3
            else:
                max_num = 1
            num_tile = self.hands[player].count(tile)
            for p in self.players:
                num_tile += self.discarded_tiles[p].count(tile)
            if self.dora == tile:
                num_tile += 1
            for i in range(max_num-num_tile):
                count_tiles.append(tile)
        return count_tiles

    def draw_hands(self):
        for player in self.players:
            self.hands[player] = sorted(self.wall[:MAX_LEN_HAND - 1])
            self.wall = self.wall[MAX_LEN_HAND - 1:]
        if self.debug:
            print('draw_hands')
            self.show_player_hand()

    def make_obs(self):
        '''
        obs_mode 1 (dict)
        dora(20), player0 hand(4x20), discards(20xP)

        obs_mode 2 (dict)
        dora(20), player0 hand(4x11), discards(20xP)

        obs_mode 3 (numpy)
        player0 hand(4x20)

        obs_mode 4 (numpy)
        dealer(4x11), dora(4x11), player0 hand(4x11), discards(Px4x11)

        obs_mode 5 (numpy)
        round_info(dealer and round_num(binary), 4x11), dora(4x11), 
        points(binary, Px4x11) player0 hand(4x11), discards(Px4x11)
        
        obs_mode 6 (numpy)
        dealer(4x11), round_num(2x4x11), dora(4x11), points(bucket, 2Px4x11)
        player0 hand(4x11), discards(Px4x11)

        obs_mode 7 (numpy)
        player0 hand(4x11)

        obs_mode 8 (numpy)
        player0 hand(11x4)

        obs_mode 9 (numpy)
        player0 hand(5x13)

        obs_mode 10 (numpy)
        player0 hand(13x5)

        obs_mode 11 (numpy)
        player0 hand, dora, remainig hand(13x(2xNUM_SAME_TILE+5))

        obs_mode 12 (numpy)
        player0 hand, dora(8), points(22P), round_num(4P), discard_tile(P) (13x(27xP+8))
        
        '''
        if self.obs_mode == 1 or self.obs_mode == 2:
            obs = {
                'dora': np.zeros(KIND_TILE_WITH_RED),
                'hand': np.zeros(self.obs_hand_shape),
                'discards': np.zeros((self.num_players, KIND_TILE_WITH_RED)),
            }
            obs['dora'][self.dora] = 1
            obs['hand'] = hand2plane(self.hands['player0'], self.obs_mode)
            for i in range(self.num_players):
                obs['discards'][i] = np.array(mahjong_utils.tiles2obs_hist(self.discarded_tiles[f'player{i}']))
            if self.debug:
                print('obs dora', obs['dora'])
                print('obs hand\n', obs['hand'])
                for i in range(self.num_players):
                    print(f'obs discard{i}', obs['discards'][i])
                print()
        elif self.obs_mode == 3:
            obs = hand2plane(self.hands['player0'], 1)
        elif self.obs_mode == 4:
            obs = np.zeros(self.obs_shape)
            if self.has_dealer:
                dealer_num = int(self.dealer[-1:])
                obs[0, :, dealer_num] = 1
            obs[1] = hand2plane([self.dora], 2)
            obs[2] = hand2plane(self.hands['player0'], 2)
            for i in range(self.num_players):
                obs[i+3] = hand2plane(self.discarded_tiles[f'player{i}'], 2)
        elif self.obs_mode == 5:
            obs = np.zeros(self.obs_shape)
            obs[0] = self.round_info2plane()
            obs[1] = hand2plane([self.dora], 2)
            obs[2:2+self.num_players] = mahjong_utils.points2planes(
                self.num_players,
                self.points,
                mode=1,
            )
            obs[2+self.num_players] = hand2plane(self.hands['player0'], 2)
            for i in range(self.num_players):
                obs[-(self.num_players-i)] = hand2plane(self.discarded_tiles[f'player{i}'], 2)
        elif self.obs_mode == 6:
            obs = np.zeros(self.obs_shape)
            obs[0] = mahjong_utils.dealer2plane(self.dealer)
            obs[1:3] = mahjong_utils.round_num2plane(self.round_num)
            obs[3] = hand2plane([self.dora], 2)
            obs[4:4+2*self.num_players] = mahjong_utils.points2planes(
                self.num_players,
                self.points,
                mode=2,
            )
            obs[4+2*self.num_players] = hand2plane(self.hands['player0'], 2)
            for i in range(self.num_players):
                obs[-(self.num_players-i)] = hand2plane(self.discarded_tiles[f'player{i}'], 2)
        elif self.obs_mode == 7:
            obs = hand2plane(self.hands['player0'], 2)
        elif self.obs_mode == 8:
            obs = hand2plane(self.hands['player0'], 2).T
        elif self.obs_mode == 9:
            obs = hand2plane(self.hands['player0'], 4)
        elif self.obs_mode == 10:
            obs = hand2plane(self.hands['player0'], 4).T
        elif self.obs_mode == 11:
            remaining_tiles = self.get_remaining_tiles('player0')
            remaining_plane = hand2plane(remaining_tiles, 4).T
            hand_plane = hand2plane(self.hands['player0'], 5, self.dora)
            obs = np.concatenate([hand_plane, remaining_plane], 1)
        elif self.obs_mode == 12:
            hand_dora_plane = hand2plane(self.hands['player0'], 5, self.dora)
            point_vec = mahjong_utils.points2vec(self.num_players, self.points)
            point_plane = np.tile(point_vec, 13).reshape(13, 22 * self.num_players)
            round_vec = np_utils.to_categorical(self.round_num, self.num_players * 4)
            round_plane = np.tile(round_vec, 13).reshape(13, self.num_players * 4)
            discards_plane = mahjong_utils.discards2exist_plane(self.discarded_tiles, self.players)
            obs = np.concatenate([hand_dora_plane, point_plane, round_plane, discards_plane], 1)
        return obs
 
    # 'player' draws a tile from the wall and calcurate point
    def tumo_and_calc_point(self, player):
        is_tumo = False
        tile = self.wall.pop()
        self.last_move = tile
        if self.debug:
            print('{} draws {} => "{}"'.format(
                player, tile, hand2show_str([tile])))
        self.hands[player].append(tile)
        self.hands[player] = sorted(self.hands[player])
        if self.debug:
            self.show_player_hand()
            print()
        win_point = mahjong_utils.calc_point(self.hands[player], self.dora)
        if win_point >= 5:
            is_tumo = True
            if player == self.dealer:
                win_point += 2
        if self.debug and is_tumo:
            print('tumo')
            print(f'tile {tile} => "{hand2show_str([tile])}"')            
        return is_tumo, win_point

    def ron_and_calc_point(self, tile, lose_player):
        self.last_move = tile
        players = self.players[:]
        is_ron = False
        win_points = []
        win_players = []
        players.remove(lose_player)
        for p in players:
            if tile in self.discarded_tiles[p]:
                continue
            temp_hand = self.hands[p][:]
            temp_hand.append(tile)
            temp_hand = sorted(temp_hand)
            point = mahjong_utils.calc_point(temp_hand, self.dora)
            if point >= 5:
                is_ron = True
                win_players.append(p)
                if p == self.dealer:
                    point += 2
                win_points.append(point)
        if self.debug and is_ron:
            print('ron')
            print(f'tile {tile} => "{hand2show_str([tile])}"')
        return is_ron, win_points, win_players

    def set_points_tumo(self, win_point, win_player):
        players = self.players[:]
        players.remove(win_player)
        divided_point = math.ceil(win_point / (self.num_players - 1))
        for p in players:
            self.points[p] -= divided_point
        self.points[win_player] += divided_point * (self.num_players - 1)
        if self.debug:
            print('show all player hand')
            self.show_player_hand()
            print('set_points_tumo', self.points)

    def set_points_ron(self, win_points, win_players, lose_player):
        for i in range(len(win_players)):
            self.points[win_players[i]] += win_points[i]
            self.points[lose_player] -= win_points[i]
        if self.debug:
            print('show all player hand')
            self.show_player_hand()
            print('set_points_ron', self.points)

    def get_discard_tile(self, player, gamma_str):
        hand = self.hands[player]
        #print(hand, self.dora, gamma_str)
        discard_tile = mahjong_utils.get_discard_tile(hand, self.dora, gamma_str)
        return discard_tile

    def remove_tile(self, discard_tile, player):
        self.hands[player].remove(discard_tile)
        self.discarded_tiles[player].append(discard_tile)
        if self.debug:
            print('{} discard {} => "{}"'.format(
                player, discard_tile, hand2show_str([discard_tile])))
        return
                
    def is_wall_remaining(self):
        if len(self.wall) == 0:
            if self.debug:
                print('there is no tiles in wall')
            return True
        return False

    def reset_env(self, complete_reset=True):
    # reset env and return done and player0's point
        player0_point = 0
        info = {}
        while True:
            if self.debug:
                print(f'\nreset_env complete_reset:{complete_reset}')
            # set dealer
            if not complete_reset and self.has_dealer:
                dealer_num = (int(self.dealer[-1]) + 1) % self.num_players
                self.dealer = f'player{dealer_num}'
            else:
                self.first_dealer_num = np.random.randint(self.num_players)
                dealer_num = self.first_dealer_num
                self.dealer = f'player{dealer_num}'
            if self.debug:
                if self.has_dealer:
                    print(f'first dealer is player{self.first_dealer_num}')
                print('dealer is', self.dealer)
            # set round_num
            if self.do_complete_game:
                if complete_reset:
                    self.round_num = 0
                else:
                    self.round_num += 1
                if self.debug:
                    print('round_num is', self.round_num)
            # reset the wall
            self.wall = mahjong_utils.make_wall()
            np.random.shuffle(self.wall)
            # reset discarded tiles and points
            for player in self.players:
                self.discarded_tiles[player] = []
                if complete_reset:
                    self.points[player] = 40
            # set before and after info
            if (self.reward_mode == 4 or self.reward_mode == 5) and self.do_complete_game:
                if complete_reset:
                    self.before_round_num = 0
                    self.before_round_point = 0
                    self.before_points = deepcopy(self.points)
                    self.before_dealer = self.dealer
                else:
                    self.before_round_num = self.after_round_num
                    self.before_points = deepcopy(self.after_points)
                    self.before_dealer = self.after_dealer
                self.after_round_num = self.round_num
                self.after_points = deepcopy(self.points)
                self.after_dealer = self.dealer
            if self.debug:
                print(f"points {self.points}")
            # dealing tiles
            self.draw_hands()
            # decide dora
            self.dora = self.wall.pop()
            if self.debug:
                print('dora {} => {}'.format(self.dora, hand2show_str([self.dora])))
            reset_info = self.make_reset_info()
            info = self.add_infos(info, reset_info)
            # dealer's first turn
            turn_players = self.players[dealer_num:] + self.players[:dealer_num]
            for player in turn_players:
                # winning from the wall
                is_tumo, win_point = self.tumo_and_calc_point(player)
                if is_tumo:
                    if self.debug:
                        print('tenho or tiho')
                        print(f'win point {win_point}')
                    self.set_points_tumo(win_point, win_player=player)
                    next_info = self.make_tumo_info(win_point, player)
                    info = self.add_infos(info, next_info)
                    div_point = math.ceil(win_point / (self.num_players - 1))
                    if player == 'player0':
                        player0_point += div_point * (self.num_players - 1)
                    else:
                        player0_point -= div_point
                    if self.do_complete_game and \
                       self.round_num == self.final_round_num:
                        return True, player0_point, info
                    break
                # return False only when player0's turn 
                if player == 'player0':
                    next_info = self.make_other_info('no')
                    info = self.add_infos(info, next_info)
                    return False, player0_point, info
                # discard optimal tile
                discard_tile = self.get_discard_tile(player, self.gamma_str)
                self.remove_tile(discard_tile, player)
                # winning from a discard which 'player' discarded
                is_ron, win_points, win_players = self.ron_and_calc_point(
                    tile=discard_tile, lose_player=player)
                if is_ron:
                    if self.debug:
                        print('renho')
                        print(f'points {win_points}')
                    self.set_points_ron(win_points, win_players, lose_player=player)
                    next_info = self.make_ron_info(win_points, win_players, player)
                    info = self.add_infos(info, next_info)                    
                    if 'player0' in win_players:
                        idx = win_players.index('player0')
                        player0_point += win_points[idx]
                    if self.do_complete_game and \
                       self.round_num == self.final_round_num:
                        return True, player0_point, info
                    break
            complete_reset = False
                
    def reset(self):
        while True:
            done, _, info = self.reset_env(complete_reset=True)
            if not done:
                break
        self.reset_info = info
        #print('reset', info)
        return self.make_obs()

    def rank2reward(self, rank):
        # rank 1 => get reward 1, worst rank => get reward -1
        return 1 + 2 * (1 - rank) / (self.num_players - 1)
                
    def make_reward(self, point, finish_kyoku=False):
        if self.debug:
            if point != 0:
                print('points', self.points)
        if self.reward_mode == 1: # reward is round score
            return point / self.reward_scale
        elif self.reward_mode == 2: # reward is round rank
            rank = get_rank(self.points, self.first_dealer_num, 'player0')
            if self.debug:
                print(f"player0 rank is {rank}")
            return self.rank2reward(rank) / self.rank_reward_scale
        elif self.reward_mode == 3: # reward is mixture of score and rank
            rank = get_rank(self.points, self.first_dealer_num, 'player0')
            #rank = self.get_rank('player0')
            if self.debug:
                print(f"player0 rank is {rank}")
            return (point / self.reward_scale + \
                    self.rank2reward(rank) / self.rank_reward_scale) / 2
        elif self.reward_mode == 4 or self.reward_mode == 5: # predict value reward
            self.after_round_point = self.after_points['player0'] - \
                                     self.before_points['player0']
            if finish_kyoku:
                before_x = reward_predict.make_features(
                    self.num_players,
                    self.model_type,
                    self.before_round_num,
                    self.before_round_point,
                    self.before_points,
                    self.before_dealer,
                )
                after_x = reward_predict.make_features(
                    self.num_players,
                    self.model_type,
                    self.after_round_num,
                    self.after_round_point,
                    self.after_points,
                    self.after_dealer,
                )
                if self.model_type == 'cnn':
                    before_x = before_x.reshape(
                        1, 5 + 2 * self.num_players, NUM_SAME_TILE, KIND_TILE)
                    after_x = after_x.reshape(
                        1, 5 + 2 * self.num_players, NUM_SAME_TILE, KIND_TILE)
                elif self.model_type == 'gru':
                    before_x = before_x.reshape(1, 1, 6 * self.num_players + 1)
                    after_x = after_x.reshape(1, 1, 6 * self.num_players + 1)
                elif self.model_type == 'dense':
                    before_x = before_x.reshape(1, 27 * self.num_players + 22)
                    after_x = after_x.reshape(1, 27 * self.num_players + 22)
                elif self.model_type == 'lstm' or self.model_type == 'rnn':
                    before_x = before_x.reshape(1, 1, 27 * self.num_players + 22)
                    after_x = after_x.reshape(1, 1, 27 * self.num_players + 22)
                before_pred_rank = self.model.predict(before_x)
                after_pred_rank = self.model.predict(after_x)
                diff = before_pred_rank - after_pred_rank
                diff = diff[0][0] / (self.num_players - 1)
                self.before_round_point = self.after_round_point
            else:
                diff = 0
            if self.reward_mode == 5:
                if point >= 0 or finish_kyoku:
                    point = 0
            return (point + diff) / self.reward_scale

    def look_ahead(self):
        if self.do_complete_game:
            if self.round_num == self.final_round_num:
                return True, 0, {}
            else:
                done, player0_point, info = self.reset_env(complete_reset=False)
                if self.debug:
                    print(f'look ahead done {done} player0_point {player0_point}')
                return done, player0_point, info
        else:
            return True, 0, {}

    def make_reset_info(self):
        reset_info = {}
        info = {}
        if self.do_complete_game:
            info["round_num"] = self.round_num
        info['first_dealer_num'] = self.first_dealer_num
        info['dealer'] = self.dealer
        info['wall'] = deepcopy(self.wall)
        info['dora'] = self.dora
        info['points'] = deepcopy(self.points)
        hand_dict = {}
        for i in range(self.num_players):
            hand_dict[f'player{i}'] = self.hands[f'player{i}'][:]
        info['hands'] = deepcopy(hand_dict)
        reset_info[0] = deepcopy(info)
        return reset_info
                
    def make_other_info(self, info_str):
        info = {}
        if hasattr(self, 'reset_info') and self.reset_info:
            info = deepcopy(self.reset_info)
            self.reset_info = {}
        info_dict = {
            'is_win': False,
            'dora': self.dora,
            'info': info_str,
            'dealer': self.dealer,
            'discards': deepcopy(self.discarded_tiles),
        }
        if self.do_complete_game:
            info_dict['first_dealer_num'] = self.first_dealer_num
            info_dict['round_num'] = self.round_num
            if info_str == 'no wall' and self.round_num == self.final_round_num:
                info_dict['final_point'] = deepcopy(self.points)
        info[len(info)] = deepcopy(info_dict)
        return info

    def make_tumo_info(self, point, win_player):
        info = {}
        if hasattr(self, 'reset_info') and self.reset_info:
            info = deepcopy(self.reset_info)
            self.reset_info = {}
        info_dict = {
            'is_win': True,
            'dora': self.dora,
            'dealer': self.dealer,
            'discards': deepcopy(self.discarded_tiles),
        }
        if self.do_complete_game:
            info_dict['first_dealer_num'] = self.first_dealer_num
            info_dict['round_num'] = self.round_num
        info_sub_dict = {}
        info_sub_dict['hands'] = deepcopy(self.hands)
        info_sub_dict['tile'] = self.last_move
        info_sub_dict['type'] = 'tumo'
        info_sub_dict['win_player'] = win_player
        div_point = math.ceil(point / (self.num_players - 1))
        info_sub_dict['point'] = div_point * (self.num_players - 1)
        info_sub_dict['turn'] = len(self.discarded_tiles[win_player]) + 1
        info_dict['info'] = deepcopy(info_sub_dict)
        #info.append(info_dict)
        info[len(info)] = info_dict
        if self.do_complete_game and self.round_num == self.final_round_num:        
            info[len(info) - 1]['final_point'] = deepcopy(self.points)
        return info

    def make_ron_info(self, points, win_players, lose_player):
        info = {}
        if hasattr(self, 'reset_info') and self.reset_info:
            info = deepcopy(self.reset_info)
            self.reset_info = {}
        for i in range(len(points)):
            info_dict = {
                'is_win': True,
                'dora': self.dora,
                'dealer': self.dealer,
                'discards': deepcopy(self.discarded_tiles),
            }
            if self.do_complete_game:
                info_dict['first_dealer_num'] = self.first_dealer_num
                info_dict['round_num'] = self.round_num
            info_sub_dict = {}
            info_sub_dict['hands'] = deepcopy(self.hands)
            info_sub_dict['tile'] = self.last_move
            info_sub_dict['type'] = 'ron'
            info_sub_dict['win_player'] = win_players[i]
            info_sub_dict['lose_player'] = lose_player
            info_sub_dict['point'] = points[i]
            info_sub_dict['turn'] = len(self.discarded_tiles[lose_player]) 
            info_dict['info'] = deepcopy(info_sub_dict)
            #info.append(info_dict)
            info[len(info)] = info_dict
        if self.do_complete_game and self.round_num == self.final_round_num:
            info[len(info) - 1]['final_point'] = deepcopy(self.points)
        return info
    
    def add_infos(self, info, next_info):
        idx = len(info)
        for i in range(len(next_info)):
            info[idx + i] = deepcopy(next_info[i])
        return info

    def step(self, act):
        # change 'int64' to 'int' for json
        act = int(act)
        if self.debug:
            print('act {} => "{}"'.format(act, hand2show_str([act])))
        # for debug
        #act = get_discard_tile('player0', '09')
        # can't discard the tile which don't have
        if act not in self.hands['player0']:
            if self.debug:
                hand = self.hands["player0"]
                print(f'no tile! hand {hand} => "{hand2show_str(hand)}"')
            self.points['player0'] -= 2 * (self.num_players - 1)
            for player in self.players[1:]:
                self.points[player] += 2
            if not self.eval_mode:
                return self.make_obs(), self.make_reward(-2 * (self.num_players - 1)), \
                False, self.make_other_info('no tile')
            else:
                rand_idx = np.random.randint(MAX_LEN_HAND - 1)
                act = self.hands['player0'][rand_idx]
        # discard 'act' tile
        self.remove_tile(act, 'player0')
        # winning from a tile which 'player0' discarded
        is_ron, win_points, win_players = self.ron_and_calc_point(
            tile=act, lose_player='player0')
        if is_ron:
            self.set_points_ron(win_points, win_players, lose_player='player0')
            info = self.make_ron_info(win_points, win_players, 'player0') 
            done, next_point, next_info = self.look_ahead()
            point = -sum(win_points) + next_point
            return self.make_obs(), self.make_reward(point, True), \
                done, self.add_infos(info, next_info)
        if self.is_wall_remaining():
            info = self.make_other_info('no wall')
            done, next_point, next_info = self.look_ahead()
            return self.make_obs(), self.make_reward(next_point, True), \
                done, self.add_infos(info, next_info)
        turn_players = self.players[1:] + self.players[:1]
        for player in turn_players:
            # winning from the wall
            is_tumo, win_point = self.tumo_and_calc_point(player)
            if is_tumo:
                self.set_points_tumo(win_point, win_player=player)
                info = self.make_tumo_info(win_point, win_player=player)
                div_point = math.ceil(win_point / (self.num_players - 1))
                done, next_point, next_info = self.look_ahead()
                if player == 'player0':
                    point = div_point * (self.num_players - 1) + next_point
                else:
                    point = -div_point + next_point
                return self.make_obs(), self.make_reward(point, True), \
                    done, self.add_infos(info, next_info)
            # break the loop when player0's turn
            if player == 'player0':
                point = 0
                if self.pseudo_reward:
                    hist = mahjong_utils.hand2hist(self.hands['player0'])
                    if mahjong_utils.has_1meld(hist):
                        point += 0.1
                return self.make_obs(), self.make_reward(point), \
                    False, self.make_other_info('no')
            # discard optimal tile
            discard_tile = self.get_discard_tile(player, self.gamma_str)
            self.remove_tile(discard_tile, player)
            # winning from a discard which 'player' discarded
            is_ron, win_points, win_players = self.ron_and_calc_point(
                tile=discard_tile, lose_player=player)
            if is_ron:
                self.set_points_ron(win_points, win_players, lose_player=player)
                info = self.make_ron_info(win_points, win_players, player)
                done, next_point, next_info = self.look_ahead()
                if 'player0' in win_players:
                    idx = win_players.index('player0')
                    point = win_points[idx] + next_point
                else:
                    point = next_point
                return self.make_obs(), self.make_reward(point, True), \
                        done, self.add_infos(info, next_info)
            if self.is_wall_remaining():
                info = self.make_other_info('no wall')
                done, next_point, next_info = self.look_ahead()
                return self.make_obs(), self.make_reward(next_point, True), \
                    done, self.add_infos(info, next_info)
