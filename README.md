## Description

__Suzume-Jong with reinforcement learning__

folder description

`baseline_results/` save 1player optimal(greedy) play (or random play) results

`baseline_run.py` run 1player optimal(greedy) play in Suzume-jan, and save log

`first_suzume.ipynb` this file made `hand2optimal`, `hand_idx_prob` and `value_iteration`

`hand2optimal/` npy or pickle files to get optimal discard tiles

`mahjong_networks.py` difine custom network(mlp, conv1d, conv2d)

`mahjong_utils.py` suzume-jan utility function such as is_win, calc_point and so on

`suzume_env/` Suzume-jan envirionment

`train_ppo.py` train and test ppo

`value_iteration/` save Q-values by using value iteration

## Requirements

OpenAI baselines, tensorflow==1.14, keras

## Usage

- test baseline player and save the result to `baseline_results` dir

`python baseline_run.py [-p num_player] [-t total_timesteps] [-og opponent_gamma] [-pg player_gamma] [-rs reward_scale] [-pr]`

`python baseline_run.py -p 2 -t 300000 -og 09 -pg random -rs 20 -ps`

- show the result 

`python baseline_run.py [-l pickle_name]`

`python baseline_run.py -l baseline_results/2_09_09_3000000.pickle`

- train ppo algorithm

`python train_ppo.py [--env env_name] [--save_path pickle_path] [-t num_timesteps]`

`python train_ppo.py --env Suzume2-v0 --save_path save_models/default.pkl -t 30000`

- show result (when --play, set num_timesteps to 0 and num_env to 1)

`python train_ppo.py --play [--load_path path]`

`python train_ppo.py --play --load_path save_models/ppo_2_5_1_99.pkl --pickle_name 09`

for more details, run `python train_ppo.py -h`

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/minnsou/suzume-jong/blob/master/LICENSE) for the full license text.

## Reference

[Suzume-Jong official site](https://sugorokuya.jp/p/suzume-jong/)

[Open AI baselines](https://github.com/openai/baselines)