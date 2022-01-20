import ataxx
import ataxx.players
import joblib
import numpy as np
import time
import os
import sys
from minimax_utils import match
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)



if __name__ == '__main__':
    #List of all the test taken into account
    cases_list = [
        {
            "name": "alpha_alpha_norand.job",
            "depth_list":  [1, 2, 3],
            "seed": 3,
            "n_matchs": 10,
            "player_1": ataxx.players.alphabeta,
            "player_2": ataxx.players.alphabeta,

            "args1": {'alpha': -99999, 'beta': 999999,
                      'feval': ataxx.players.eval_fun, 'rand': False},
            "args2": {'alpha': -99999, 'beta': 999999,
                      'feval': ataxx.players.eval_fun, 'rand': False}
        },
        {
            "name": "alpha_alpha_rand.job",
            "depth_list":  [1, 2, 3],
            "seed": 3,
            "n_matchs": 10,
            "player_1": ataxx.players.alphabeta,
            "player_2": ataxx.players.alphabeta,

            "args1": {'alpha': -99999, 'beta': 999999,
                      'feval': ataxx.players.eval_fun, 'rand': True},
            "args2": {'alpha': -99999, 'beta': 999999,
                      'feval': ataxx.players.eval_fun, 'rand': True}
        },
        {
            "name": "alpha_alpha_alteval_noeval.job",
            "depth_list":  [1, 2, 3],
            "seed": 3,
            "n_matchs": 10,
            "player_1": ataxx.players.alphabeta,
            "player_2": ataxx.players.alphabeta,

            "args1": {'alpha': -99999, 'beta': 999999,
                      'feval': ataxx.players.alt_val_fun, 'rand': False},
            "args2": {'alpha': -99999, 'beta': 999999,
                      'feval': ataxx.players.eval_fun, 'rand': False}
        },
        
        {
            "name": "alpha_alpha_alteval_noeval_rand.job",
            "depth_list":  [1, 2, 3],
            "seed": 3,
            "n_matchs": 10,
            "player_1": ataxx.players.alphabeta,
            "player_2": ataxx.players.alphabeta,

            "args1": {'alpha': -99999, 'beta': 999999,
                      'feval': ataxx.players.alt_val_fun, 'rand': True},
            "args2": {'alpha': -99999, 'beta': 999999,
                      'feval': ataxx.players.eval_fun, 'rand': True}
        }



    ]

    for param in cases_list:
        match(param["depth_list"], param["seed"], param["n_matchs"],
              name=param["name"],
              player_1=param["player_1"],
              player_2=param["player_2"],
              args1=param["args1"], args2=param["args2"])
