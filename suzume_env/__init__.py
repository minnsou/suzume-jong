from gym.envs.registration import register

register(
    id='Suzume2-v0',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':2,
        'gamma_str':'09',
        'obs_mode':6,
        'has_dealer':True,
        'pseudo_reward':False,
        'reward_scale':20,
        'reward_mode':1,
        'do_complete_game':False, # fix
        'numpy_seed':0,
        'debug':False,
    }
)

register(
    id='Suzume2-v1',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':2,
        'gamma_str':'09',
        'obs_mode':10,
        'has_dealer':True, # fix
        'pseudo_reward':False,
        'reward_scale':1000,
        'reward_mode':4,
        'model_type': 'dense',
        'do_complete_game':True, # fix
        'rank_reward_scale':1000,
        'numpy_seed':0,
        'eval_mode':False, # fix
        'debug':False,
    }
)

# for eval
register(
    id='Suzume2-v2',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':2,
        'gamma_str':'09',
        'obs_mode':10,
        'has_dealer':True, # fix
        'pseudo_reward':False,
        'reward_scale':1000,
        'reward_mode':1, # anything is OK
        'do_complete_game':True, # fix
        'rank_reward_scale':1000,
        'numpy_seed':0,
        'eval_mode':True, # fix
        'debug':False,
    }
)

register(
    id='Suzume3-v0',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':3,
        'gamma_str':'06',
        'obs_mode':4,
        'has_dealer':False,
        'pseudo_reward':False,
        'reward_scale':20,
        'reward_mode':1,
        'do_complete_game':False, # fix
        'numpy_seed':0,
        'debug':False,
    }
)

register(
    id='Suzume3-v1',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':3,
        'gamma_str':'09',
        'obs_mode':5,
        'has_dealer':True, # fix
        'pseudo_reward':False,
        'reward_scale':1000,
        'reward_mode':3,
        'do_complete_game':True, # fix
        'rank_reward_scale':1000,
        'eval_mode':False, # fix        
        'numpy_seed':0,
        'debug':False,
    }
)

# for eval
register(
    id='Suzume3-v2',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':3,
        'gamma_str':'09',
        'obs_mode':5,
        'has_dealer':True, # fix
        'pseudo_reward':False,
        'reward_scale':1000,
        'reward_mode':1,
        'do_complete_game':True, # fix
        'rank_reward_scale':1000,
        'numpy_seed':0,
        'eval_mode':True, # fix
        'debug':False,
    }
)

register(
    id='Suzume4-v0',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':4,
        'gamma_str':'06',
        'obs_mode':4,
        'has_dealer':False,
        'pseudo_reward':False,
        'reward_scale':20,
        'reward_mode':1,
        'do_complete_game':False, # fix
        'numpy_seed':0,
        'debug':False,
    }
)

register(
    id='Suzume4-v1',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':4,
        'gamma_str':'06',
        'obs_mode':5,
        'has_dealer':True, # fix
        'pseudo_reward':False,
        'reward_scale':20,
        'reward_mode':2,
        'do_complete_game':True, # fix
        'rank_reward_scale':1000,
        'numpy_seed':0,
        'eval_mode':False, # fix        
        'debug':False,
    }
)

# for eval
register(
    id='Suzume4-v2',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':4,
        'gamma_str':'09',
        'obs_mode':6,
        'has_dealer':True, # fix
        'pseudo_reward':False,
        'reward_scale':1000,
        'reward_mode':1,
        'do_complete_game':True, # fix
        'rank_reward_scale':1000,
        'numpy_seed':0,
        'eval_mode':True, # fix
        'debug':False,
    }
)

register(
    id='Suzume5-v0',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':5,
        'gamma_str':'09',
        'obs_mode':4,
        'has_dealer':False,
        'pseudo_reward':False,
        'reward_scale':20,
        'reward_mode':1,
        'do_complete_game':False, # fix
        'numpy_seed':0,
        'debug':False,
    }
)

register(
    id='Suzume5-v1',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':5,
        'gamma_str':'09',
        'obs_mode':6,
        'has_dealer':True, # fix
        'pseudo_reward':False,
        'reward_scale':1000,
        'reward_mode':4,
        'model_type': 'gru',
        'do_complete_game':True, # fix
        'rank_reward_scale':1000,
        'numpy_seed':0,
        'eval_mode':False, # fix
        'debug':False,
    }
)

# for eval
register(
    id='Suzume5-v2',
    entry_point='suzume_env.main:SuzumeEnv',
    kwargs = {
        'num_players':5,
        'gamma_str':'09',
        'obs_mode':6,
        'has_dealer':True, # fix
        'pseudo_reward':False,
        'reward_scale':1000,
        'reward_mode':1,
        'do_complete_game':True, # fix
        'rank_reward_scale':1000,
        'numpy_seed':0,
        'eval_mode':True, # fix
        'debug':False,
    }
)
