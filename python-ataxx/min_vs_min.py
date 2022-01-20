import ataxx
import ataxx.players
import joblib
import numpy as np
import time
import os
import sys
from minimax_utils import match
# Antonio Ruiz Molero, 2022

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


depth_list = [1, 2, 3]


if __name__ == '__main__':
    cases_list = [
        #     {
        #         "depth_list":  [1, 2, 3],
        #         "seed": 3,
        #         "n_matchs": 2,
        #         "name": "hola.job",
        #         "player_1": ataxx.players.alphabeta,
        #         "player_2": ataxx.players.alphabeta,

        #         "args1": {'alpha': -99999, 'beta': 999999,
        #                   'feval': ataxx.players.eval_fun, 'rand': True},
        #         "args2": {'alpha': -99999, 'beta': 999999,
        #                   'feval': ataxx.players.eval_fun, 'rand': True}
        #     }
        # ,
        {
            "name": "mini_mini_norand.job",
            "depth_list":  [1, 2],
            "seed": 2,
            "n_matchs": 10,
            "name": "mini_mini_norand.job",
            "player_1": ataxx.players.minimax,
            "player_2": ataxx.players.minimax,

            "args1": {
                'feval': ataxx.players.eval_fun, 'rand': False},
            "args2": {
                'feval': ataxx.players.eval_fun, 'rand': False}
        },
        {
            "depth_list":  [1, 2],
            "seed": 2,
            "n_matchs": 10,
            "name": "mini_mini_yesrand.job",
            "player_1": ataxx.players.minimax,
            "player_2": ataxx.players.minimax,

            "args1": {
                'feval': ataxx.players.eval_fun, 'rand': True},
            "args2": {
                'feval': ataxx.players.eval_fun, 'rand': True}
        },
        {
            "depth_list":  [1, 2],
            "seed": 2,
            "n_matchs": 10,
            "name": "mini_mini_yesrand_mixed.job",
            "player_1": ataxx.players.minimax,
            "player_2": ataxx.players.minimax,

            "args1": {
                'feval': ataxx.players.eval_fun, 'rand': True},
            "args2": {
                'feval': ataxx.players.eval_fun, 'rand': False}
        },
        {
            "depth_list":  [1, 2],
            "seed": 2,
            "n_matchs": 10,
            "name": "mini_mini_norand_alt.job",
            "player_1": ataxx.players.minimax,
            "player_2": ataxx.players.minimax,

            "args1": {
                'feval': ataxx.players.alt_val_fun, 'rand': False},
            "args2": {
                'feval': ataxx.players.alt_val_fun, 'rand': False}
        },
        {
            "depth_list":  [1, 2],
            "seed": 2,
            "n_matchs": 10,
            "name": "mini_mini_yesrand_alt.job",
            "player_1": ataxx.players.minimax,
            "player_2": ataxx.players.minimax,

            "args1": {
                'feval': ataxx.players.alt_val_fun, 'rand': True},
            "args2": {
                'feval': ataxx.players.alt_val_fun, 'rand': True}
        },
        {
            "name": "mini_mini_yesrand_alt_noalt.job",
            "depth_list":  [1, 2],
            "seed": 2,
            "n_matchs": 10,

            "player_1": ataxx.players.minimax,
            "player_2": ataxx.players.minimax,

            "args1": {
                'feval': ataxx.players.alt_val_fun, 'rand': True},
            "args2": {
                'feval': ataxx.players.alt_val_fun, 'rand': True}
        }
    ]

    for param in cases_list:
        match(param["depth_list"], param["seed"], param["n_matchs"],
              name=param["name"],
              player_1=param["player_1"],
              player_2=param["player_2"],
              args1=param["args1"], args2=param["args2"])
