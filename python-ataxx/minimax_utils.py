
# Antonio Ruiz Molero, 2022
# Probability distributions of the three different difficulties and the function that
# select the depth given a the stage of the game
import numpy as np
import joblib
import time
import ataxx
# Antonio Ruiz Molero, 2022


easy = [[0.9, 0.1, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0]]

medium = [[0.5, 0.5, 0.0, 0.0],
          [0.3, 0.7, 0.0, 0.0],
          [0.3, 0.7, 0.0, 0.0]]

pro = [[0.35, 0.5, 0.1, 0.05],
       [0.2, 0.5, 0.2, 0.1],
       [0.1, 0.35, 0.35, 0.2]]


def select_depth(board, prob_list):
    gaps = board.num_gaps()
    if gaps < 49 and gaps > 20:
        dist = prob_list[0]
    if gaps <= 20 and gaps > 10:
        dist = prob_list[1]
    if gaps <= 10:
        dist = prob_list[2]

    return np.random.choice([1, 2, 3, 4], 1, 5, p=dist)


# We define the function that allows to test all the cases.

def match(depth_list, seed,  n_matchs, name, player_1,
          player_2, args1, args2):

    np.random.seed(seed)
    for d1 in depth_list:
        for d2 in depth_list:
            if d2 < d1:
                continue
            results = [0, 0]

            results_dict = []
            game_length = []
            game_time = []
            whostarts = []
            for i in range(n_matchs):
                start_temp = time.time()
                board = ataxx.Board()  # create the board
                who_starts_i = np.random.randint(0, 2)  # select starter player
                board.turn = who_starts_i
                turn = 0
                while not board.gameover():
                    if board.turn == ataxx.BLACK:
                        # execute movement of player black with the selected algorithm
                        move = player_1(board, depth=d1, **args1)
                    else:
                        # execute movement of player white with the selected algorithm
                        move = player_2(board, depth=d2,  **args2)
                    turn = turn + 1
                    board.makemove(move)
                end_temp = time.time()
                game_time.append(end_temp - start_temp)
                game_length.append(turn)
                whostarts.append(who_starts_i)

                aux = board.result().split("-")
                if aux[0] == "1/2" or aux[1] == "1/2":
                    aux[0], aux[1] = 0.33, 0.33

                results[0] += float(aux[0])
                results[1] += float(aux[1])
            # export results
            aux_dict = {"depth 1": d1,
                        "depth 2": d2,
                        "results": results,
                        "starting player": whostarts,
                        "length": game_length,
                        "time": game_time,
                        "name": name}
            print(aux_dict)
            results_dict.append(aux_dict)
            # we export the results in a job file
    joblib.dump(results_dict, name)
