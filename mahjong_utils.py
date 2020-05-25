import itertools
import pickle
import math
import numpy as np

NUM_SAME_TILE = 4
KIND_TILE = 11
KIND_TILE_WITH_RED = 20
MAX_LEN_HAND = 6

hand_to_optimal_dicts06 = []
hand_to_optimal_dicts09 = []
for dora in range(KIND_TILE_WITH_RED):
    f06 = open('./hand2optimal/hand2optdict_{}_{}.pickle'.format('06', dora), 'rb')
    f09 = open('./hand2optimal/hand2optdict_{}_{}.pickle'.format('09', dora), 'rb')
    hand_to_optimal_dicts06.append(pickle.load(f06))
    hand_to_optimal_dicts09.append(pickle.load(f09))

# "len_hand" is MAX_LEN_HAND or MAX_LEN_HAND - 1
def generate_all_valid_hands(len_hand, dora=None):
    all_valid_hands = []
    for hand in itertools.combinations_with_replacement(range(KIND_TILE_WITH_RED), len_hand):
        if (is_valid(hand, dora)):
            all_valid_hands.append(hand)
    return all_valid_hands

# check each kind of tile
def is_valid(hand, dora=None):
    hand_set = set(hand)
    for tile in hand_set:
        if tile == dora:
            dora_count = 1
        else:
            dora_count = 0
            
        if tile == 18 or tile == 19: # 發 or 中
            if hand.count(tile) >= NUM_SAME_TILE + 1 - dora_count:
                return False
        elif tile % 2 == 0: # normal tile (not red Dora tile)
            if hand.count(tile) >= 4 - dora_count:
                return False
        else: # red Dora tile
            if hand.count(tile) >= 2 - dora_count:
                return False
    return True

def hand2show_str(hand):
    show_str = ''
    for tile in hand:
        if tile == 18:
            show_str += '發 '
            continue
        elif tile == 19:
            show_str += '中 '
            continue
        elif tile % 2 == 1:
            tile -= 1
            show_str += 'r'
        num = int((tile / 2) + 1)
        show_str += '{} '.format(num)
    return show_str[:-1]

def hand2plane(hand, mode):
    if mode == 1:
        plane = np.zeros(shape=(NUM_SAME_TILE, KIND_TILE_WITH_RED))
        for tile in range(KIND_TILE_WITH_RED):
            plane[4-hand.count(tile):4, tile] = 1
    elif mode == 2:
        plane = np.zeros(shape=(NUM_SAME_TILE, KIND_TILE))
        for tile in range(KIND_TILE_WITH_RED):
            if tile == 18 or tile == 19:
                #print(tile)
                plane[4-hand.count(tile):4, tile - 9] = 1
            elif tile % 2 == 0:
                #print(hand.count(tile))
                plane[3-hand.count(tile):3, tile // 2] = 1
            else:
                #print(hand.count(tile))
                if hand.count(tile) > 0:
                    plane[3, tile // 2] = 1
    return plane

def hand2hist(hand):
    # hist = [1索, 2索, ..., 9索, 0, 發, 0, 中]
    # hist[9] and hist[11] are always 0 due to using has_2melds function
    hist = [0] * (KIND_TILE + 2)
    for tile in hand: 
        if tile == 19:
            hist[12] += 1
        elif tile == 18:
            hist[10] += 1
        else:
            hist[tile // 2] += 1
    return hist

def hands2hists(hands):
    hists = []
    for hand in hands:
        hists.append(hand2hist(hand))
    return hists

def hist2hand(hist):
    hand = []
    for i, n in enumerate(hist):
        for j in range(n):
            hand.append(i)
    return hand

def has_1meld(hist):
    if hist.count(4) >= 1:
        return True
    if hist.count(3) >= 1:
        return True
    for i in range(KIND_TILE - 2):
        if hist[i] >= 1 and hist[i+1] >= 1 and hist[i+2] >= 1:
            return True
    return False
        
def has_2melds(hist):
    return has_2melds_sub(hist, n_melds=2)

def has_2melds_sub(hist, n_melds):
    if n_melds == 0:
        return True
    i = next(i for i, x in enumerate(hist) if x > 0)
    # Pong
    if hist[i] >= 3 and has_2melds_sub([x - 3 if i == j else x for j, x in enumerate(hist)], n_melds - 1):
        return True
    # Chows
    if i + 2 < len(hist) and hist[i + 1] > 0 and hist[i + 2] > 0 and has_2melds_sub([x - 1 if i <= j <= i + 2 else x for j, x in enumerate(hist)], n_melds - 1):
        return True
    return False

# yaku
one_nine_honer_tiles = [0, 1, 16, 17, 18, 19] # 1索, 9索, 發, 中
green_tiles = [2, 4, 6, 10, 14, 18] # 2索, 3索, 4索, 6索, 8索, 發

# count odd number
def count_red_tile(hand):
    count = 0
    for tile in hand:
        if tile % 2 == 1:
            count += 1
    return count

# include one_nine_honer_tile or not
def is_tanyao(hand):
    for tile in hand:
        if tile in one_nine_honer_tiles:
            return False
    return True

# count 1索, 9索, 發, 中
# chanta <=> has_2melds(hand) == True and (count_one_nine_honer == 2 or count_one_nine_honer == 4)
def is_chanta(hand):
    count_one_nine_honor = 0
    for tile in hand:
        if tile in one_nine_honer_tiles:
            count_one_nine_honor += 1
    if count_one_nine_honor == 2 or count_one_nine_honor == 4:
        return True
    return False

# include green tile or not
def is_all_green(hand):
    for tile in hand:
        if not tile in green_tiles:
            return False
    return True

# count 1索, 9索, 發, 中
def is_chinyao(hand):
    count_one_nine_honor = 0
    for tile in hand:
        if tile in one_nine_honer_tiles:
            count_one_nine_honor += 1
    if count_one_nine_honor == 6:
        return True
    return False

# count red tile
def is_super_red(hand):
    for tile in hand:
        if tile % 2 == 0: # even number is not red
            return False
    return True

def calc_point(hand, dora=None):
    hist = hand2hist(hand)
    if not has_2melds(hist):
        return 0
    else:
        if is_all_green(hand):
            return 10
        if is_chinyao(hand):
            return 15
        if is_super_red(hand):
            return 20
        
        point = count_red_tile(hand)
        if hist.count(3) == 2:
            point += 4 # 2 Pongs
        elif hist.count(3) == 1 or hist.count(4) == 1:
            point += 3 # 1 Pong and 1 Chow
        else:
            point += 2 # 2 Chows
        if is_tanyao(hand):
            point += 1
        elif is_chanta(hand):
            point += 2
        if dora is not None:
            if dora == 18 or dora == 19:
                point += hand.count(dora)
            else:
                if dora % 2 == 1:
                    point += hand.count(dora - 1)
                else:
                    point += hand.count(dora)
                    point += hand.count(dora + 1)
        return point

# wall = [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 5, ..., 16, 16, 16, 17, 18, 18, 18, 18, 19, 19, 19, 19]
# if set 'dora', remove dora from wall
def make_wall(dora=None):
    wall = list(range(KIND_TILE_WITH_RED))
    for i in range(0, 18, 2):
        for j in range(2):
            wall.append(i)
    for i in [18, 19]:
        for j in range(3):
            wall.append(i)
    if dora is not None: # remove dora from wall
        wall.remove(dora)
    return sorted(wall)

# careful: if discard_hand includes dora, wall.remove raise Error
def discard_hand_to_hand_idx_and_prob(discard_hand, valid_hands, dora=None):
    wall = make_wall(dora)
    for tile in discard_hand:
        wall.remove(tile)
    hand_idx_and_prob = []
    for tile in range(KIND_TILE_WITH_RED):
        if wall.count(tile) == 0:
            continue
        prob = wall.count(tile) / len(wall)
        next_hand = tuple(sorted(list(discard_hand) + [tile]))
        hand_idx = valid_hands.index(next_hand)
        hand_idx_and_prob.append((hand_idx, prob))
    return hand_idx_and_prob

def hand_to_discard_hands(hand):
    return list(set(tuple(hand[:i] + hand[i+1:]) for i in range(MAX_LEN_HAND)))

def get_discard_tile(hand, dora, gamma_str):
    #assert gamma_str == '06' or gamma_str == '09' or gamma_str == 'random', 'gamma_str must be "06" or "09" or "random", not "{}"'.format(gamma_str)
    if gamma_str == 'random':
        random_idx = np.random.randint(MAX_LEN_HAND)
        discard_tile = hand[random_idx]
    else:
        if gamma_str == '06':
            opt_tiles = hand_to_optimal_dicts06[dora][tuple(hand)]
        else:
            opt_tiles = hand_to_optimal_dicts09[dora][tuple(hand)]
        random_idx = np.random.randint(len(opt_tiles))
        discard_tile = opt_tiles[random_idx]
    return discard_tile

def has_win_hand(hand_idx_and_prob_i, valid_hands):
    for (idx, p) in hand_idx_and_prob_i:
        if calc_point(valid_hands[idx]) >= 5:
            return True
    return False

def tiles2obs_hist(tiles):
    obs_hist = [0] * KIND_TILE_WITH_RED
    for tile in tiles:
        obs_hist[tile] += 1
    return obs_hist

def plane2hand(plane, mode):
    hand = []
    if mode == 1:
        for i in range(KIND_TILE_WITH_RED):
            for j in range(int(plane[:, i].sum())):
                hand.append(i)
    if mode == 2:
        for i in range(KIND_TILE):
            if i <= 8:
                #print(int(plane[1:, i].sum()))
                for j in range(int(plane[:3, i].sum())):
                    hand.append(i * 2)
                if plane[3, i] == 1:
                    hand.append(i * 2 + 1)
            else:
                for j in range(int(plane[:, i].sum())):
                    hand.append(i+9)
    return hand

def sort_by_dealer(num_players, first_dealer_num, same_ranks):
    sorted_list = []
    score = same_ranks[0][1]
    players = [f'player{i}' for i in range(num_players)]
    sorted_players = players[first_dealer_num:] + players[:first_dealer_num]
    for player in sorted_players:
        if (player, score) in same_ranks:
            sorted_list.append(player)
    return sorted_list

def get_rank(points, first_dealer_num, player):
    num_players = len(points)
    sort_points = sorted(points.items(), key=lambda x:x[1])
    sorted_tuples = []
    p_s_tuple1 = sort_points.pop()
    while True:
        same_ranks = [p_s_tuple1]
        while len(sort_points) >= 1:
            p_s_tuple2 = sort_points.pop()
            if p_s_tuple1[1] > p_s_tuple2[1]:
                p_s_tuple1 = p_s_tuple2
                break
            same_ranks.append(p_s_tuple2)
        else:
            sorted_tuple = sort_by_dealer(num_players, first_dealer_num, same_ranks)
            sorted_tuples.append(sorted_tuple)
            break
        sorted_tuple = sort_by_dealer(num_players, first_dealer_num, same_ranks)
        sorted_tuples.append(sorted_tuple)
    rank_list = list(itertools.chain.from_iterable(sorted_tuples))
    return rank_list.index(player) + 1

def dealer2plane(dealer):
    plane = np.zeros((NUM_SAME_TILE, KIND_TILE))
    dealer_num = int(dealer[-1:])
    plane[:, dealer_num] = 1
    return plane

def round_num2plane(round_num):
    plane = np.zeros((2, NUM_SAME_TILE, KIND_TILE))
    if round_num <= 10:
        plane[0, :, round_num] = 1
    else:
        plane[1, :, round_num-11] = 1
    return plane

def points2planes(num_players, points, mode=2):
    if mode == 1:
        num_planes = num_players
    elif mode == 2:
        num_planes = 2 * num_players
    shape = (num_planes, NUM_SAME_TILE, KIND_TILE)
    planes = np.zeros(shape)
    for i in range(num_players):
        point = points[f'player{i}']
        if mode == 1:
            if point >= 0:
                bin_str = format(point, '011b')
            else:
                if point < -1024:
                    point = -1024
                bin_str = format(point & 0b11111111111, 'b')
            for j, s in enumerate(bin_str):
                planes[i, :, j] = int(s)
        elif mode == 2:
            if point < 0:
                planes[2*i, :, 0] = 1
            elif point >= 100:
                planes[2*i+1, :, 10] = 1
            else:
                num_row = point // 5 + 1
                if num_row <= 10:
                    planes[2*i, :, num_row] = 1
                else:
                    planes[2*i+1, :, num_row-11] = 1
    return planes

def points2vec(num_players, points):
    vec = np.zeros((22 * num_players))
    for i in range(num_players):
        point = points[f'player{i}']
        if point < 0:
            vec[i*22] = 1
        elif point >= 100:
            vec[i*22+21] = 1
        else:
            idx = point // 5 + 1
            vec[i*22+idx] = 1
    return vec

def round_point2plane(round_point):
    plane = np.zeros((2, NUM_SAME_TILE, KIND_TILE))
    if round_point >= 19:
        plane[0, :, 10] = 1
    elif round_point >= 0:
        num_row = math.ceil(round_point / 2)
        plane[0, :, num_row] = 1
    elif round_point <= -21:
        plane[1, :, 0] = 1
    else:
        num_row = 11 + math.floor(round_point / 2)
        plane[1, :, num_row-11] = 1
    return plane
    
def round_point2vec(round_point):
    vec = np.zeros((22))
    if round_point >= 19:
        vec[21] = 1
    elif round_point <= -21:
        vec[0] = 1
    elif round_point >= 0:
        idx = math.ceil(round_point / 2)
        vec[11+idx] = 1
    else:
        idx = math.floor(round_point / 2)
        vec[11+idx] = 1
    return vec
    
