# See installation instruction in the readme file.
register(
    id='Ataxx-v0',
    entry_point='gym.envs.ataxx:ReversiEnv',
    kwargs={
        'player_color': 'black', 
        'opponent': 'random', #"manual" is also available
        'observation_type': 'numpy3c',
        'illegal_place_mode': 'lose', #other options are raise and pass.
        'board_size': 5 #select board size
        
    }
)


