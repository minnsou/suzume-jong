import json
import pprint
import pathlib
import os
import argparse

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers import Dense, GRU, Conv2D, Flatten, Masking, Dropout, SimpleRNN, LSTM

import mahjong_utils
from mahjong_utils import NUM_SAME_TILE, KIND_TILE, round_point2vec

def make_features(num_players, model_type, round_num, round_point, points, dealer):
    # [1 round_num, 2 round_point, 3 points, 4 dealer] to vector(or plane)
    if model_type == 'cnn':
        num_planes = 5 + 2 * num_players
        features = np.zeros((num_planes, NUM_SAME_TILE, KIND_TILE))
        features[0:2] = mahjong_utils.round_num2plane(round_num)
        features[2:4] = mahjong_utils.round_point2plane(round_point)
        features[4] = mahjong_utils.dealer2plane(dealer)
        features[-num_players*2:] = mahjong_utils.points2planes(
            num_players,
            points
        )
    elif model_type == 'gru':
        features = np.zeros((6 * num_players + 1))
        # round_num
        features[:4*num_players] = np_utils.to_categorical(
            round_num,
            num_players * 4
        )
        # round_point
        features[4*num_players] = round_point
        # point of each player
        for j in range(num_players):
            features[4*num_players+1+j] = points[f'player{j}']
        # dealer
        features[-num_players:] = np_utils.to_categorical(
            int(dealer[-1]),
            num_players
        )
    elif model_type == 'dense' or model_type == "rnn" or model_type == 'lstm':
        features = np.zeros((27 * num_players + 22))
        # round_num
        features[:4*num_players] = np_utils.to_categorical(
            round_num,
            num_players * 4
        )
        # round_point
        features[4*num_players:4*num_players+22] = round_point2vec(round_point)
        # point of each player
        features[4*num_players+22:26*num_players+22] = \
            mahjong_utils.points2vec(num_players, points)
        # dealer
        features[-num_players:] = np_utils.to_categorical(
            int(dealer[-1]),
            num_players
        )
    return features

def log2features(json_file, model_type):
    with open(json_file) as f:
        log_dict = json.load(f)
    #pprint.pprint(log_dict, width=40)
    num_players = log_dict['game_info']['num_players']
    if model_type == 'cnn':
        num_planes = 5 + 2 * num_players
        features = np.zeros((4 * num_players, num_planes, NUM_SAME_TILE, KIND_TILE))
    elif model_type == 'gru':
        features = np.zeros((4 * num_players, 6 * num_players + 1))
    elif model_type == 'dense' or model_type == 'rnn' or model_type == 'lstm':
        features = np.zeros((4 * num_players, 27 * num_players + 22))
    for i in range(num_players * 4):
        reset_dict = log_dict[f'{i} reset']
        kyoku_dict = log_dict[f'{i} kyoku']
        first_dealer_num = reset_dict['first_dealer_num']
        round_num = reset_dict['round_num']
        points = reset_dict['points']
        dealer = reset_dict['dealer']
        if 'no wall' in kyoku_dict:
            round_point = 0
            if 'final_point' in kyoku_dict['no wall']:
                final_point = kyoku_dict['no wall']['final_point']
        else:
            agari_num = 0
            round_point = 0
            while True:
                key = f'agari{agari_num}'
                if key in kyoku_dict:
                    if kyoku_dict[key]['win_player'] == 'player0':
                        round_point = kyoku_dict[key]['point']
                    elif kyoku_dict[key]['type'] == 'tumo':
                        round_point = - kyoku_dict[key]['point'] / (num_players - 1)
                    elif kyoku_dict[key]['lose_player'] == 'player0':
                        round_point -= kyoku_dict[key]['point']
                else:
                    break
                if 'final_point' in kyoku_dict[key]:
                    final_point = kyoku_dict[key]['final_point']
                agari_num += 1
        features[i] = make_features(
            num_players,
            model_type,
            round_num,
            round_point,
            points,
            dealer
        )
    rank = mahjong_utils.get_rank(final_point, first_dealer_num, 'player0')
    if model_type == 'gru':
        features = features.reshape(1, 4 * num_players, 6 * num_players + 1)
    elif model_type == 'lstm' or model_type == 'rnn':
        features = features.reshape(1, 4 * num_players,  27 * num_players + 22)
    return features, rank

def make_model(num_players, model_type):
    model = Sequential()
    if model_type == 'dense':
        input_dim = 27 * num_players + 22
        model.add(Dense(units=64, activation='relu', input_dim=input_dim))
        model.add(Dense(units=32, activation='relu'))
    elif model_type == 'gru':
        input_shape = (None, 6 * num_players + 1)
        # マスキングあり
        model.add(Masking(input_shape=input_shape, mask_value=-1.0))
        model.add(GRU(units=64, return_sequences=True))
        # マスキングなし
        # model.add(GRU(units=64, return_sequences=True, input_shape=input_shape))
        model.add(GRU(units=64))
    elif model_type == 'rnn':
        input_shape = (None, 27 * num_players + 22)
        model.add(SimpleRNN(units=64, input_shape=input_shape, return_sequences=True))
        model.add(SimpleRNN(units=64))
    elif model_type == 'lstm':
        input_shape = (None, 27 * num_players + 22)        
        model.add(LSTM(units=64, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(units=64))
    elif model_type == 'cnn':
        input_shape = (5 + 2 * num_players, NUM_SAME_TILE, KIND_TILE)
        model.add(Conv2D(filters=64, kernel_size=3, input_shape=input_shape))
        model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.summary()
    return model

def train(model_type, num_players, player_gamma, opponent_gamma):
    basename = f'{num_players}_{player_gamma}_{opponent_gamma}'
    train_name = f'{model_type}_{basename}'
    if os.path.exists(f'./save_rew_pred/x_{train_name}.npy'):
        print('load npy data')
        x = np.load(f'./save_rew_pred/x_{train_name}.npy')
        y = np.load(f'./save_rew_pred/y_{train_name}.npy')
    else:
        dirname = f'./log_json/{basename}'
        if model_type == 'cnn':
            x = np.zeros((1, 5 + 2 * num_players, NUM_SAME_TILE, KIND_TILE))
        elif model_type == 'gru':
            x = np.zeros((1, 4 * num_players, 6 * num_players + 1))
        elif model_type == 'dense':
            x = np.zeros((1, 27 * num_players + 22))
        elif model_type == 'lstm' or model_type == 'rnn':
            x = np.zeros((1, 4 * num_players, 27 * num_players + 22))
        y = []
        for i, name in enumerate(pathlib.Path(dirname).iterdir()):
            features, rank = log2features(name, model_type)
            x = np.vstack([x, features])
            if model_type == 'cnn' or model_type == 'dense':
                y += [rank for j in range(4 * num_players)]
            elif model_type == 'gru' or model_type == 'rnn' or model_type == 'lstm':
                y.append(rank)
        x = x[1:]
        np.save(f'./save_rew_pred/x_{train_name}', x)
        np.save(f'./save_rew_pred/y_{train_name}', y)
    model = make_model(num_players, model_type)
    epochs = 100
    batch_size = 128
    history = model.fit(
        x,
        y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1
    )
    model.save(f'{train_name}_{epochs}.h5')
    pd.DataFrame(history.history).to_csv(f'{train_name}_{epochs}.csv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--num_players", help="number of players (default 2)", type=int, default=2)
    parser.add_argument("-pg", "--player_gamma", help="player discount rate (default '09'), using when getting logs", type=str, default='09')    
    parser.add_argument("-og", "--opponent_gamma", help="opponent player discount rate (default '09')", type=str, default='09')    
    parser.add_argument("-m", "--model_type", help="model type (default 'dense')", type=str, default='dense')
    parser.add_argument('--play', default=False, action='store_true')

    args = parser.parse_args()
    num_players = args.num_players
    player_gamma = args.player_gamma
    opponent_gamma = args.opponent_gamma
    model_type = args.model_type

    log_name = f'{num_players}_{player_gamma}_{opponent_gamma}'

    if not args.play:
        train(model_type, num_players, player_gamma, opponent_gamma)
    else:
        model = load_model(f'./save_rew_pred/{model_type}_{log_name}_300.h5')
        if num_players == 2 and model_type == 'dense':
            x = np.zeros((3, 76))
            x[0, 0] = 1 # round_num 0
            x[0, 14] = 1 # round_point -10
            x[0, 39] = 1 # player0 point 40
            x[0, 61] = 1 # player1 point 40
            x[0, 75] = 1 # dealer 1
            x[1, 7] = 1 # round_num 7
            x[1, 14] = 1 # round_point -10
            x[1, 37] = 1 # player0 point 30
            x[1, 63] = 1 # player1 point 50
            x[1, 75] = 1 # dealer 1
            x[2, 7] = 1 # round_num 7
            x[2, 19] = 1 # round_point 0
            x[2, 41] = 1 # player0 point 50
            x[2, 59] = 1 # player1 point 30
            x[2, 75] = 1 # dealer 1
        elif num_players == 2 and model_type == 'gru':
            x = np.zeros((3, 1, 13))
            x[0, 0, 7] = 1 # round_num 7
            x[0, 0, 8] = 0 # round_point 0
            x[0, 0, 9] = 20 # player0 point 20
            x[0, 0, 10] = 60 # player1 point 60
            x[0, 0, 12] = 1 # dealer 0
            x[1, 0, 7] = 1 # round_num 7
            x[1, 0, 8] = 0 # round_point 0
            x[1, 0, 9] = 60 # player0 point 60
            x[1, 0, 10] = 20 # player1 point 20
            x[1, 0, 12] = 1 # dealer 0
            x[2, 0, 7] = 1 # round_num 7
            x[2, 0, 8] = 5 # round_point 5
            x[2, 0, 9] = 41 # player0 point 41
            x[2, 0, 10] = 39 # player1 point 39
            x[2, 0, 12] = 1 # dealer 0
        elif num_players == 2 and model_type == 'cnn':
            x = np.zeros((3, 9, 4, 11))
            x[0, 0, :, 0] = 1 # round_num 0
            x[0, 3, :, -5] = 1 # round_point -10
            x[0, 4, :, 1] = 1 # dealer 1
            x[0, 5, :, 9] = 1 # player0 point 40
            x[0, 7, :, 9] = 1 # player1 point 40
            x[1, 0, :, 7] = 1 # round_num 7
            x[1, 2, :, 3] = 1 # round_point 5
            x[1, 4, :, 1] = 1 # dealer 1
            x[1, 6, :, 0] = 1 # player0 point 50
            x[1, 7, :, 7] = 1 # player1 point 30
            x[2, 0, :, 7] = 1 # round_num 7
            x[2, 2, :, 0] = 1 # round_point 0
            x[2, 4, :, 1] = 1 # dealer 1
            x[2, 5, :, 7] = 1 # player0 point 30
            x[2, 8, :, 0] = 1 # player1 point 50
        elif num_players == 5 and model_type == 'dense':
            x = np.zeros((3, 157))
            x[0, 12] = 1 # round_num 12 (0~19)
            x[0, 24] = 1 # round_point -13 (20~41)
            x[0, 48] = 1 # player0 point 29 (42~63)
            x[0, 77] = 1 # player1 point 63 (64~85)
            x[0, 95] = 1 # player2 point 42 (86~107)
            x[0, 113] = 1 # player3 point 22 (108~129)
            x[0, 139] = 1 # player4 point 44 (130~151)
            x[0, 155] = 1 # dealer 3 (152~156)
            x[1, 19] = 1 # round_num 19 (0~19)
            x[1, 30] = 1 # round_point 0 (20~41)
            x[1, 55] = 1 # player0 point 60 (42~63)
            x[1, 75] = 1 # player1 point 50 (64~85)
            x[1, 95] = 1 # player2 point 42 (86~107)
            x[1, 111] = 1 # player3 point 10 (108~129)
            x[1, 139] = 1 # player4 point 44 (130~151)
            x[1, 155] = 1 # dealer 3 (152~156)
            x[2, 19] = 1 # round_num 19 (0~19)
            x[2, 30] = 1 # round_point 0 (20~41)
            x[2, 47] = 1 # player0 point 20 (42~63)
            x[2, 75] = 1 # player1 point 50 (64~85)
            x[2, 95] = 1 # player2 point 42 (86~107)
            x[2, 119] = 1 # player3 point 50 (108~129)
            x[2, 139] = 1 # player4 point 44 (130~151)
            x[2, 155] = 1 # dealer 3 (152~156)
        elif num_players == 5 and model_type == 'gru':
            # gru_5_09_09 まあまあ学習できている
            x = np.zeros((3, 1, 31))
            x[0, 0, 12] = 1 # round_num 12
            x[0, 0, 20] = 0 # round_point 0
            x[0, 0, 21] = 40 # player0 point 40
            x[0, 0, 22] = 20 # player1 point 20
            x[0, 0, 23] = 60 # player2 point 60
            x[0, 0, 24] = 30 # player3 point 30
            x[0, 0, 25] = 50 # player4 point 50
            x[0, 0, 29] = 1 # dealer 3
            x[1, 0, 19] = 1 # round_num 19
            x[1, 0, 20] = 0 # round_point 0
            x[1, 0, 21] = 63 # player0 point 63
            x[1, 0, 22] = 29 # player1 point 29
            x[1, 0, 23] = 42 # player2 point 42
            x[1, 0, 24] = 22 # player3 point 22
            x[1, 0, 25] = 44 # player4 point 44
            x[1, 0, 29] = 1 # dealer 3
            x[2, 0, 19] = 1 # round_num 19
            x[2, 0, 20] = 0 # round_point 0
            x[2, 0, 21] = 20 # player0 point 20
            x[2, 0, 22] = 63 # player1 point 63
            x[2, 0, 23] = 42 # player2 point 42
            x[2, 0, 24] = 31 # player3 point 31
            x[2, 0, 25] = 44 # player4 point 44
            x[2, 0, 29] = 1 # dealer 3
        elif num_players == 5 and model_type == 'cnn':
            x = np.zeros((1, 15, NUM_SAME_TILE, KIND_TILE))
            x[0, 1, :, 1] = 1 # round_num 12
            x[0, 3, :, 4] = 1 # round_point -13
            x[0, 4, :, 3] = 1 # dealer 3
            x[0, 5, :, 6] = 1 # player0 point 29
            x[0, 8, :, 2] = 1 # player1 point 63
            x[0, 9, :, 9] = 1 # player2 point 42
            x[0, 11, :, 5] = 1 # player3 point 22
            x[0, 13, :, 9] = 1 # player4 point 44
        print(model.predict(x))


if __name__ == '__main__':
    main()
